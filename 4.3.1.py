import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import rasterio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import csv

warnings.filterwarnings("ignore")

# ===================== 全局配置（对齐论文） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 256
IN_CHANNELS = 249  # LuoJiaHSSR 249通道
OUT_CHANNELS = 1
BATCH_SIZE = 8  # 小样本下稍大batch稳定训练
EPOCHS = 150  # 增加epoch确保收敛
LEARNING_RATE = 1e-4  # 论文常用学习率
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-5  # L2正则化缓解过拟合
EARLY_STOP_PATIENCE = 30
EDGE_LOSS_WEIGHT = 0.3  # 边缘损失权重

# ===================== 路径配置（替换为您的真实路径） =====================
BASE_DIR = os.getcwd()
TRAIN_IMG_DIR = r"D:\unet\Satellite Images\img256_train_new"
TRAIN_LABEL_DIR = r"D:\unet\Satellite Images\label256_train_new"
VAL_IMG_DIR = r"D:\unet\Satellite Images\img256_val_new"
VAL_LABEL_DIR = r"D:\unet\Satellite Images\label256_val_new"

TEST_IMG_DIR = r"D:\unet\Satellite Images\img256_test_new" if os.path.exists(
    r"D:\unet\Satellite Images\img256_test_new") else VAL_IMG_DIR
TEST_LABEL_DIR = r"D:\unet\Satellite Images\label256_test_new" if os.path.exists(
    r"D:\unet\Satellite Images\label256_test_new") else VAL_LABEL_DIR

LOG_DIR = os.path.join(BASE_DIR, 'logs')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
VIS_DIR = os.path.join(BASE_DIR, 'vis')
RESULT_DIR = os.path.join(BASE_DIR, 'results')


def create_dirs():
    dirs = [LOG_DIR, WEIGHTS_DIR, VIS_DIR, RESULT_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ 目录初始化完成，输出保存在: {BASE_DIR}")


create_dirs()
print(f"✅ 使用设备: {DEVICE}")


# ===================== 论文复现：标准U-Net + 边缘检测分支 =====================
class DoubleConv(nn.Module):
    """(卷积 -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样（最大池化 -> DoubleConv）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样（转置卷积 -> 拼接 -> DoubleConv）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BuildingSegModel(nn.Module):
    """论文复现：带边缘检测分支的U-Net"""

    def __init__(self, n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 主分割分支解码器
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_seg = nn.Conv2d(64, n_classes, kernel_size=1)

        # 边缘检测分支（从编码器中间层提取特征）
        self.edge_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.edge_conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.edge_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.out_edge = nn.Conv2d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 主分割分支
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits_seg = self.out_seg(x)
        out_seg = self.sigmoid(logits_seg)

        # 边缘检测分支
        edge_feat1 = self.edge_conv1(x2)
        edge_feat2 = self.edge_conv2(x3)
        edge_feat = edge_feat1 + nn.functional.interpolate(edge_feat2, scale_factor=2, mode='bilinear')
        edge_feat = self.edge_up(edge_feat)
        logits_edge = self.out_edge(edge_feat)
        out_edge = self.sigmoid(logits_edge)

        return out_seg, out_edge


# ===================== 手写Dice Loss =====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice


# ===================== 修复后损失函数：强制float32避开autocast问题 =====================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred_seg, pred_edge, target_seg):
        # 🔴 核心修复：强制在float32下计算loss，避开autocast的数值不稳定性
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为float32
            pred_seg_fp32 = pred_seg.float()
            pred_edge_fp32 = pred_edge.float()
            target_seg_fp32 = target_seg.float()

            # 主分割损失：BCE + Dice
            loss_bce = self.bce(pred_seg_fp32, target_seg_fp32)
            loss_dice = self.dice(pred_seg_fp32, target_seg_fp32)
            main_loss = 0.5 * loss_bce + 0.5 * loss_dice

            # 边缘损失：用Sobel提取真实边缘，BCE监督
            target_edge = self._extract_edge(target_seg_fp32)
            loss_edge = self.bce(pred_edge_fp32, target_edge)

            total_loss = (1 - EDGE_LOSS_WEIGHT) * main_loss + EDGE_LOSS_WEIGHT * loss_edge

        return total_loss, main_loss, loss_edge

    def _extract_edge(self, x):
        """动态Sobel边缘提取"""
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        edge_x = nn.functional.conv2d(x, sobel_kernel_x, padding=1)
        edge_y = nn.functional.conv2d(x, sobel_kernel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edge / (edge.max() + 1e-8)


# ===================== 后处理（论文中提到的形态学+小连通域去除） =====================
def post_process(mask):
    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    # 形态学闭运算填充小空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 去除小连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            mask_bin[labels == i] = 0
    return mask_bin.astype(np.float32) / 255.0


# ===================== 论文指标：IoU、F1、精确率、召回率 =====================
def calculate_metrics(pred_mask, target_mask):
    pred_bin = (pred_mask > 0.5).float()
    target_bin = (target_mask > 0.5).float()

    # 基础统计
    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()
    TN = ((1 - pred_bin) * (1 - target_bin)).sum()

    # 计算指标
    precision = (TP + 1e-8) / (TP + FP + 1e-8)
    recall = (TP + 1e-8) / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = (TP + 1e-8) / (TP + FP + FN + 1e-8)

    return iou.item(), f1.item(), precision.item(), recall.item()


# ===================== 传统方法：Otsu阈值分割 =====================
def otsu_threshold_segmentation(img):
    """取前3通道均值，Otsu阈值分割"""
    img_gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.float32) / 255.0


# ===================== 传统方法：Canny边缘检测+形态学 =====================
def canny_edge_segmentation(img):
    """Canny边缘检测+闭运算+连通域填充"""
    img_gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
    edges = cv2.Canny(img_gray, 50, 150)
    # 闭运算连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 填充连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges_closed, connectivity=8)
    mask = np.zeros_like(edges_closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            mask[labels == i] = 255
    return mask.astype(np.float32) / 255.0


# ===================== 数据集类（严格对齐LuoJiaHSSR参数） =====================
class LuoJiaHSSRSegDataset(Dataset):
    def __init__(self, img_dir, label_dir, is_train=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.is_train = is_train
        self.cache = {}

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        label_files = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
        self.files = list(zip(img_files, label_files))
        print(f"✅ {'训练集' if is_train else '验证/测试集'}有效样本: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label_path = self.files[idx]
        fname = os.path.basename(img_path)
        if fname not in self.cache:
            # 读取249通道高光谱图像（int16，像素范围[-439, 8779]）
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)
                # 论文中提到的2%-98%百分位归一化
                for c in range(IN_CHANNELS):
                    p2, p98 = np.percentile(img[c], (2, 98))
                    img[c] = np.clip(img[c], p2, p98)
                    img[c] = (img[c] - p2) / (p98 - p2 + 1e-8)
                img = img.transpose(1, 2, 0)
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

            # 读取标签（uint8，像素范围[0,80]，二值化）
            with rasterio.open(label_path) as src:
                label = src.read(1).astype(np.float32)
                label = cv2.resize(label, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
                label = (label > 0).astype(np.float32)

            self.cache[fname] = (img, label)

        img, label = self.cache[fname]

        # 小样本下强数据增强
        if self.is_train:
            if np.random.random() > 0.5:
                img = np.flip(img, 1).copy()
                label = np.flip(label, 1).copy()
            if np.random.random() > 0.5:
                img = np.flip(img, 0).copy()
                label = np.flip(label, 0).copy()
            if np.random.random() > 0.5:
                k = np.random.choice([1, 2, 3])
                img = np.rot90(img, k, axes=(0, 1)).copy()
                label = np.rot90(label, k, axes=(0, 1)).copy()
            # 光谱随机扰动（缓解高光谱过拟合）
            if np.random.random() > 0.5:
                img = img + np.random.normal(0, 0.01, img.shape).astype(np.float32)
                img = np.clip(img, 0, 1)

        img = img.transpose(2, 0, 1)
        label = np.expand_dims(label, 0)
        return torch.tensor(img).float(), torch.tensor(label).float(), fname, img.transpose(1, 2, 0)  # 返回原图用于传统方法


# ===================== 可视化（论文风格） =====================
def visualize_result(img, label, pred_mask, pred_edge, save_path, img_name, metrics):
    img = img[:, :, :3]  # 取前3通道可视化
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    label = label.squeeze()
    pred_mask = pred_mask.squeeze()
    pred_edge = pred_edge.squeeze()
    error_map = np.abs(label - pred_mask)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 5, 1);
    plt.imshow(img);
    plt.title('卫星遥感原图');
    plt.axis('off')
    plt.subplot(1, 5, 2);
    plt.imshow(label, cmap='gray');
    plt.title('真实建筑掩码');
    plt.axis('off')
    plt.subplot(1, 5, 3);
    plt.imshow(pred_mask, cmap='gray');
    plt.title('预测建筑掩码');
    plt.axis('off')
    plt.subplot(1, 5, 4);
    plt.imshow(pred_edge, cmap='gray');
    plt.title('预测建筑边缘');
    plt.axis('off')
    plt.subplot(1, 5, 5);
    plt.imshow(error_map, cmap='jet');
    plt.title(f'误差热力图\nIoU={metrics["iou"]:.4f}');
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===================== 生成论文表4-1：不同方法对比 =====================
def generate_comparison_table(model, test_ds, save_path):
    print("\n📊 生成论文表4-1：不同建筑分割方法性能对比...")
    methods = ['阈值分割', '边缘检测', '本文U-Net模型']
    ious, f1s, precisions, recalls = [], [], [], []

    # 遍历测试集
    for idx in tqdm(range(len(test_ds)), desc="评估中"):
        img_tensor, label_tensor, fname, img_np = test_ds[idx]
        label_np = label_tensor.squeeze().numpy()

        # 1. 阈值分割
        mask_otsu = otsu_threshold_segmentation(img_np)
        iou_otsu, f1_otsu, prec_otsu, rec_otsu = calculate_metrics(torch.tensor(mask_otsu), torch.tensor(label_np))

        # 2. 边缘检测
        mask_canny = canny_edge_segmentation(img_np)
        iou_canny, f1_canny, prec_canny, rec_canny = calculate_metrics(torch.tensor(mask_canny), torch.tensor(label_np))

        # 3. 本文U-Net模型
        model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            pred_mask, pred_edge = model(img_tensor)
            pred_mask = pred_mask.cpu().numpy()[0, 0]
            pred_mask = post_process(pred_mask)
        iou_unet, f1_unet, prec_unet, rec_unet = calculate_metrics(torch.tensor(pred_mask), torch.tensor(label_np))

        # 保存当前样本指标
        ious.append([iou_otsu, iou_canny, iou_unet])
        f1s.append([f1_otsu, f1_canny, f1_unet])
        precisions.append([prec_otsu, prec_canny, prec_unet])
        recalls.append([rec_otsu, rec_canny, rec_unet])

    # 计算平均指标
    mean_iou = np.mean(ious, axis=0)
    mean_f1 = np.mean(f1s, axis=0)
    mean_prec = np.mean(precisions, axis=0)
    mean_rec = np.mean(recalls, axis=0)

    # 生成表格
    table = f"表4-1 不同建筑分割方法性能对比\n"
    table += f"{'方法名称':<15} {'IoU':<10} {'F1分数':<10} {'精确率':<10} {'召回率':<10}\n"
    table += "-" * 60 + "\n"
    for i, method in enumerate(methods):
        table += f"{method:<15} {mean_iou[i]:<10.4f} {mean_f1[i]:<10.4f} {mean_prec[i]:<10.4f} {mean_rec[i]:<10.4f}\n"

    # 保存并打印
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(table)
    print("\n" + table)
    print(f"✅ 表4-1已保存至: {os.path.relpath(save_path, BASE_DIR)}")
    return mean_iou, mean_f1, mean_prec, mean_rec


# ===================== 训练主函数 =====================
def run_training():
    # 1. 加载数据
    train_ds = LuoJiaHSSRSegDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, is_train=True)
    val_ds = LuoJiaHSSRSegDataset(VAL_IMG_DIR, VAL_LABEL_DIR, is_train=False)
    test_ds = LuoJiaHSSRSegDataset(TEST_IMG_DIR, TEST_LABEL_DIR, is_train=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. 初始化模型、损失、优化器
    model = BuildingSegModel().to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    best_iou = 0
    early_stop = 0
    log_path = os.path.join(LOG_DIR, 'training_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_IoU', 'Val_F1', 'Val_Precision', 'Val_Recall'])

    # 3. 训练循环
    print("\n🚀 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, labels, fnames, imgs_np in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_masks, pred_edges = model(imgs)
                loss, main_loss, edge_loss = criterion(pred_masks, pred_edges, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"总损失": f"{loss.item():.4f}", "主损失": f"{main_loss.item():.4f}",
                              "边缘损失": f"{edge_loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # 4. 验证
        model.eval()
        val_iou, val_f1, val_prec, val_rec = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels, fnames, imgs_np in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                pred_masks, pred_edges = model(imgs)
                preds = pred_masks.cpu().numpy()
                for b in range(preds.shape[0]):
                    preds[b, 0] = post_process(preds[b, 0])
                preds = torch.from_numpy(preds).to(DEVICE)

                # 计算指标
                batch_iou, batch_f1, batch_prec, batch_rec = 0, 0, 0, 0
                for b in range(preds.shape[0]):
                    i, f, p, r = calculate_metrics(preds[b], labels[b])
                    batch_iou += i
                    batch_f1 += f
                    batch_prec += p
                    batch_rec += r
                batch_iou /= preds.shape[0]
                batch_f1 /= preds.shape[0]
                batch_prec /= preds.shape[0]
                batch_rec /= preds.shape[0]
                val_iou += batch_iou
                val_f1 += batch_f1
                val_prec += batch_prec
                val_rec += batch_rec

        avg_iou = val_iou / len(val_loader)
        avg_f1 = val_f1 / len(val_loader)
        avg_prec = val_prec / len(val_loader)
        avg_rec = val_rec / len(val_loader)

        # 记录日志
        with open(log_path, 'a', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, round(avg_loss, 4),
                                    round(avg_iou, 4), round(avg_f1, 4),
                                    round(avg_prec, 4), round(avg_rec, 4)])

        print(f"\n📊 Epoch {epoch + 1} 结果:")
        print(f"  训练损失: {avg_loss:.4f} | 验证IoU: {avg_iou:.4f} | 验证F1: {avg_f1:.4f}")
        print(f"  精确率: {avg_prec:.4f} | 召回率: {avg_rec:.4f}")

        # 保存最优模型
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_model_path = os.path.join(WEIGHTS_DIR, "best_building_seg_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最优模型 (IoU: {best_iou:.4f})")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= EARLY_STOP_PATIENCE:
                print(f"🚀 早停触发，最佳IoU: {best_iou:.4f}")
                break

        scheduler.step(avg_iou)

    # 5. 生成论文表4-1
    print(f"\n🎉 训练完成！最佳IoU: {best_iou:.4f}")
    best_model_path = os.path.join(WEIGHTS_DIR, "best_building_seg_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    table_path = os.path.join(RESULT_DIR, "table4-1.txt")
    generate_comparison_table(model, test_ds, table_path)


if __name__ == "__main__":
    run_training()