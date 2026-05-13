from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from collections import defaultdict
import rasterio
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2
import random

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2  # 小样本调小batch，提升梯度稳定性
EPOCHS_FIRSTU = 20  # 增加训练轮次，保证拟合
EPOCHS_SECONDU = 20
LEARNING_RATE = 5e-5  # 调小学习率，避免震荡
# 路径配置（保持你的原有路径不变）
LABEL_DIR = r"D:\unet\Satellite Images\label256_val_new"
SAVE_DIR = r"D:\unet\RadioWNet_c_DPM_Thr2"
VIS_DIR = os.path.join(SAVE_DIR, "visualizations")
LOSS_LOG_PATH = os.path.join(SAVE_DIR, "loss_log.json")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ===================== 无线电传播物理模型 =====================
class RadioPropagationPhysics:
    def __init__(self, freq=2.1e9, tx_height=30, rx_height=1.5):
        self.freq = freq
        self.tx_height = tx_height
        self.rx_height = rx_height
        self.reference_power = 0.0
        self.path_loss_exponent = 3.8
        self.building_penetration_loss = 25.0  # 强化遮挡衰减，提升真值动态范围
        self.shadow_fading_std = 2.0

    def _has_line_of_sight(self, tx_pos, rx_pos, building_mask):
        h, w = building_mask.shape
        x0, y0 = tx_pos
        x1, y1 = rx_pos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        building_count = 0
        while True:
            if 0 <= x < w and 0 <= y < h:
                if building_mask[y, x] > 0.5:
                    building_count += 1
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return building_count

    def generate_radio_map(self, building_mask, tx_position):
        h, w = building_mask.shape
        radio_map = np.zeros((h, w), dtype=np.float32)
        tx_x, tx_y = tx_position
        for y in range(h):
            for x in range(w):
                distance = np.sqrt((x - tx_x)**2 + (y - tx_y)**2)
                distance = np.maximum(distance, 1.0)
                free_space_loss = 10 * self.path_loss_exponent * np.log10(distance)
                building_count = self._has_line_of_sight(tx_position, (x, y), building_mask)
                penetration_loss = building_count * self.building_penetration_loss
                shadow_fading = np.random.normal(0, self.shadow_fading_std)
                rx_power = self.reference_power - free_space_loss - penetration_loss + shadow_fading
                radio_map[y, x] = rx_power
        # 固定归一化范围，避免不同样本归一化后动态范围不一致
        radio_map = np.clip(radio_map, -150, 0)
        radio_map = (radio_map + 150) / 150  # 固定映射到[0,1]，1=最强信号
        return radio_map

# ===================== RadioWNet模型：添加Sigmoid激活，分阶段训练支持 =====================
class RadioWNet(nn.Module):
    def __init__(self, phase="firstU"):
        super(RadioWNet, self).__init__()
        self.phase = phase
        self.relu = nn.ReLU(inplace=True)
        # ---------------- First U-Net（粗预测） ----------------
        self.layer00 = nn.Conv2d(2, 64, 3, 1, 1)
        self.layer0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.layer10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.layer20 = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.layer4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer5 = nn.Conv2d(512, 512, 3, 2, 1)
        self.conv_up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv_up4 = nn.ConvTranspose2d(1024, 256, 2, 2)
        self.conv_up3 = nn.ConvTranspose2d(512, 128, 2, 2)
        self.conv_up2 = nn.ConvTranspose2d(256, 64, 2, 2)
        self.conv_up0 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv_up00 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up000 = nn.Conv2d(64, 1, 1)
        self.sigmoid1 = nn.Sigmoid()  # 【核心修复】强制输出到[0,1]

        # ---------------- Second U-Net（精修） ----------------
        self.Wlayer00 = nn.Conv2d(2 + 1, 64, 3, 1, 1)
        self.Wlayer0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Wlayer1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.Wlayer10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.Wlayer2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.Wlayer20 = nn.Conv2d(256, 256, 3, 1, 1)
        self.Wlayer3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.Wlayer4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.Wlayer5 = nn.Conv2d(512, 512, 3, 2, 1)
        self.Wconv_up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.Wconv_up4 = nn.ConvTranspose2d(1024, 256, 2, 2)
        self.Wconv_up3 = nn.ConvTranspose2d(512, 128, 2, 2)
        self.Wconv_up2 = nn.ConvTranspose2d(256, 64, 2, 2)
        self.Wconv_up0 = nn.Conv2d(128, 64, 3, 1, 1)
        self.Wconv_up00 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Wconv_up000 = nn.Conv2d(64, 1, 1)
        self.sigmoid2 = nn.Sigmoid()  # 【核心修复】强制输出到[0,1]

        # 权重初始化，避免梯度消失
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ============== 第一个 U-Net（粗预测） ==============
        x00 = self.relu(self.layer00(x))
        x0 = self.relu(self.layer0(x00))
        x1 = self.relu(self.layer1(x0))
        x10 = self.relu(self.layer10(x1))
        x2 = self.relu(self.layer2(x10))
        x20 = self.relu(self.layer20(x2))
        x3 = self.relu(self.layer3(x20))
        x4 = self.relu(self.layer4(x3))
        x5 = self.relu(self.layer5(x4))
        up5 = self.relu(self.conv_up5(x5))
        up5 = torch.cat([up5, x4], dim=1)
        up4 = self.relu(self.conv_up4(up5))
        up4 = torch.cat([up4, x20], dim=1)
        up3 = self.relu(self.conv_up3(up4))
        up3 = torch.cat([up3, x10], dim=1)
        up2 = self.relu(self.conv_up2(up3))
        up2 = torch.cat([up2, x0], dim=1)
        up0 = self.relu(self.conv_up0(up2))
        up00 = self.relu(self.conv_up00(up0))
        out1 = self.sigmoid1(self.conv_up000(up00))  # 【修复】加Sigmoid

        # ============== 第二个 U-Net（精修） ==============
        x_in = torch.cat([x, out1], dim=1)
        x00_2 = self.relu(self.Wlayer00(x_in))
        x0_2 = self.relu(self.Wlayer0(x00_2))
        x1_2 = self.relu(self.Wlayer1(x0_2))
        x10_2 = self.relu(self.Wlayer10(x1_2))
        x2_2 = self.relu(self.Wlayer2(x10_2))
        x20_2 = self.relu(self.Wlayer20(x2_2))
        x3_2 = self.relu(self.Wlayer3(x20_2))
        x4_2 = self.relu(self.Wlayer4(x3_2))
        x5_2 = self.relu(self.Wlayer5(x4_2))
        up5_2 = self.relu(self.Wconv_up5(x5_2))
        up5_2 = torch.cat([up5_2, x4_2], dim=1)
        up4_2 = self.relu(self.Wconv_up4(up5_2))
        up4_2 = torch.cat([up4_2, x20_2], dim=1)
        up3_2 = self.relu(self.Wconv_up3(up4_2))
        up3_2 = torch.cat([up3_2, x10_2], dim=1)
        up2_2 = self.relu(self.Wconv_up2(up3_2))
        up2_2 = torch.cat([up2_2, x0_2], dim=1)
        up0_2 = self.relu(self.Wconv_up0(up2_2))
        up00_2 = self.relu(self.Wconv_up00(up0_2))
        out2 = self.sigmoid2(self.Wconv_up000(up00_2))  # 【修复】加Sigmoid

        return out1, out2

    # 【修复】分阶段冻结参数
    def freeze_first_unet(self):
        """冻结第一U-Net，仅训练第二U-Net"""
        for name, param in self.named_parameters():
            if not name.startswith('Wlayer') and not name.startswith('Wconv_'):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_second_unet(self):
        """冻结第二U-Net，仅训练第一U-Net"""
        for name, param in self.named_parameters():
            if name.startswith('Wlayer') or name.startswith('Wconv_'):
                param.requires_grad = False
            else:
                param.requires_grad = True

# ===================== 数据集类：添加同步数据增强，保证输入-标签空间对应 =====================
class RadioMapDataset(Dataset):
    def __init__(self, label_dir, is_train=True):
        self.files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])
        self.label_dir = label_dir
        self.is_train = is_train
        self.physics_model = RadioPropagationPhysics()
        # 基站位置池，覆盖不同位置，保证模型泛化性
        self.tx_positions = [
            (64,64), (64,192), (192,64), (192,192), (128,128),
            (32,128), (128,32), (224,128), (128,224), (224,64)
        ]
        print(f"✅ 加载{'训练集' if is_train else '验证集'}样本数: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def _augment_data(self, building_mask, bs_mask, radio_gt):
        """【核心修复】同步数据增强 + 强制拷贝，彻底解决负步长问题"""
        # 随机水平翻转
        if random.random() > 0.5:
            building_mask = cv2.flip(building_mask, 1).copy()  # 每步都加 .copy()
            bs_mask = cv2.flip(bs_mask, 1).copy()
            radio_gt = cv2.flip(radio_gt, 1).copy()
        # 随机垂直翻转
        if random.random() > 0.5:
            building_mask = cv2.flip(building_mask, 0).copy()
            bs_mask = cv2.flip(bs_mask, 0).copy()
            radio_gt = cv2.flip(radio_gt, 0).copy()
        # 随机90/180/270度旋转
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            building_mask = np.rot90(building_mask, k=rot_k).copy()
            bs_mask = np.rot90(bs_mask, k=rot_k).copy()
            radio_gt = np.rot90(radio_gt, k=rot_k).copy()
        return building_mask, bs_mask, radio_gt

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 1. 读取建筑掩码
        with rasterio.open(os.path.join(self.label_dir, fname)) as src:
            building_mask = src.read(1).astype(np.float32)
        building_mask = cv2.resize(building_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        building_mask = (building_mask > 0).astype(np.float32)

        # 2. 生成基站位置
        if self.is_train:
            tx_pos = self.tx_positions[random.randint(0, len(self.tx_positions)-1)]
        else:
            tx_pos = self.tx_positions[idx % len(self.tx_positions)]
        tx_x, tx_y = tx_pos
        bs_mask = np.zeros_like(building_mask)
        bs_mask[max(0, tx_y-2):min(256, tx_y+2), max(0, tx_x-2):min(256, tx_x+2)] = 1.0

        # 3. 生成无线电地图真值
        radio_gt = self.physics_model.generate_radio_map(building_mask, tx_pos)

        # 4. 训练集执行同步数据增强
        if self.is_train:
            building_mask, bs_mask, radio_gt = self._augment_data(building_mask, bs_mask, radio_gt)

        # 5. 【终极修复】使用 np.ascontiguousarray 强制内存连续，100%兼容PyTorch
        input_tensor = np.stack([
            np.ascontiguousarray(building_mask),
            np.ascontiguousarray(bs_mask)
        ], axis=0)

        return (
            torch.tensor(np.ascontiguousarray(input_tensor)).float(),
            torch.tensor(np.ascontiguousarray(radio_gt)).float(),
            torch.tensor(np.ascontiguousarray(building_mask)).float(),
            torch.tensor(np.ascontiguousarray(bs_mask)).float(),
            fname,
            tx_pos
        )

# ===================== 损失函数与指标计算 =====================
def calc_radio_loss(pred, target, metrics):
    criterion = nn.MSELoss()
    pred = pred.squeeze(1)
    loss = criterion(pred, target)
    metrics['mse_loss'] += loss.item() * target.size(0)
    metrics['rmse'] += torch.sqrt(loss).item() * target.size(0)
    return loss

# ===================== 全局损失记录 =====================
global_loss_history = {
    "firstU_train": [], "firstU_val": [],
    "secondU_train": [], "secondU_val": []
}

# ===================== 训练函数：分阶段冻结参数，正确训练双U-Net =====================
def train_model(model, optimizer, scheduler, num_epochs, train_phase, dataloaders):
    best_loss = 1e10
    best_wts = copy.deepcopy(model.state_dict())
    # 分阶段冻结参数
    if train_phase == "firstU":
        model.freeze_second_unet()
        print("🔒 已冻结第二U-Net，仅训练第一U-Net（粗预测）")
    else:
        model.freeze_first_unet()
        print("🔒 已冻结第一U-Net，仅训练第二U-Net（精修）")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            metrics = defaultdict(float)
            count = 0
            # 全量收集预测和真值，用于计算全局R²
            all_preds = []
            all_gts = []

            for inp, gt_radio, _, _, _, _ in tqdm(dataloaders[phase], desc=f"{phase}进度"):
                inp, gt_radio = inp.to(DEVICE), gt_radio.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2 = model(inp)
                    pred = out1 if train_phase == "firstU" else out2
                    loss = calc_radio_loss(pred, gt_radio, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 收集数据用于R²计算
                all_preds.append(pred.squeeze(1).detach().cpu().numpy())
                all_gts.append(gt_radio.detach().cpu().numpy())
                count += inp.size(0)

            # 计算epoch级指标
            epoch_loss = metrics['mse_loss'] / count
            epoch_rmse = metrics['rmse'] / count
            # 【修复】全局R²计算，避免单批次失真
            all_preds = np.concatenate(all_preds).flatten()
            all_gts = np.concatenate(all_gts).flatten()
            ss_res = np.sum((all_gts - all_preds) ** 2)
            ss_tot = np.sum((all_gts - all_gts.mean()) ** 2)
            epoch_r2 = 1 - (ss_res / (ss_tot + 1e-8))

            print(f"{phase} | MSE损失: {epoch_loss:.6f} | RMSE: {epoch_rmse:.4f} | R²: {epoch_r2:.4f}")
            # 保存损失日志
            if train_phase == "firstU":
                global_loss_history[f"firstU_{phase}"].append(epoch_loss)
            else:
                global_loss_history[f"secondU_{phase}"].append(epoch_loss)
            # 保存最优模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_wts = copy.deepcopy(model.state_dict())
                print(f"✅ 保存最优模型，当前验证损失: {best_loss:.6f}")
        scheduler.step()
    model.load_state_dict(best_wts)
    return model

# ===================== 训练损失曲线可视化 =====================
def plot_training_loss():
    save_path = os.path.join(SAVE_DIR, "training_loss_curve.png")
    plt.figure(figsize=(12, 6))
    epochs1 = range(1, len(global_loss_history["firstU_train"]) + 1)
    plt.plot(epochs1, global_loss_history["firstU_train"], 'b-o', linewidth=2, label='第一阶段-训练损失', markersize=5)
    plt.plot(epochs1, global_loss_history["firstU_val"], 'r-o', linewidth=2, label='第一阶段-验证损失', markersize=5)
    epochs2 = range(len(epochs1), len(epochs1) + len(global_loss_history["secondU_train"]))
    plt.plot(epochs2, global_loss_history["secondU_train"], 'g-s', linewidth=2, label='第二阶段-训练损失', markersize=5)
    plt.plot(epochs2, global_loss_history["secondU_val"], 'm-s', linewidth=2, label='第二阶段-验证损失', markersize=5)
    plt.xlabel('训练轮次 Epoch', fontsize=12)
    plt.ylabel('MSELoss 损失值', fontsize=12)
    plt.title('RadioWNet 无线电预测模型 两阶段训练损失曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📈 训练损失曲线已保存至: {save_path}")

# ===================== 可视化函数：叠加建筑轮廓，直观验证遮挡效应 =====================
def visualize_radio_results(inp, building_mask, bs_mask, gt_radio, pred_radio, save_path, fname, tx_pos):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    inp_np = inp.squeeze().cpu().numpy()
    building_np = building_mask.squeeze().cpu().numpy()
    gt_np = gt_radio.squeeze().cpu().numpy()
    pred_np = pred_radio.squeeze().cpu().numpy()
    vmin, vmax = 0, 1  # 统一色阶，保证对比一致性

    # 1. 输入：建筑Mask + 基站位置
    axes[0].imshow(building_np, cmap='gray')
    axes[0].scatter(tx_pos[0], tx_pos[1], c='red', s=120, marker='*', label='基站位置')
    axes[0].set_title('输入: 建筑Mask + 基站位置', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].axis('off')

    # 2. 输入：基站位置掩码
    axes[1].imshow(bs_mask.squeeze().cpu().numpy(), cmap='hot')
    axes[1].set_title('输入: 基站位置掩码', fontsize=14)
    axes[1].axis('off')

    # 3. 真值：无线电强度（叠加建筑轮廓）
    im3 = axes[2].imshow(gt_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].contour(building_np, levels=[0.5], colors='white', linewidths=2, linestyles='-')
    axes[2].scatter(tx_pos[0], tx_pos[1], c='white', s=100, marker='*', edgecolors='black', linewidths=1.5)
    axes[2].set_title('真值: 无线电强度（含建筑遮挡）', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='归一化信号强度（1=最强）')
    axes[2].axis('off')

    # 4. 预测：无线电强度（叠加建筑轮廓）
    im4 = axes[3].imshow(pred_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[3].contour(building_np, levels=[0.5], colors='white', linewidths=2, linestyles='-')
    axes[3].scatter(tx_pos[0], tx_pos[1], c='white', s=100, marker='*', edgecolors='black', linewidths=1.5)
    axes[3].set_title('预测: 无线电强度', fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04, label='归一化信号强度（1=最强）')
    axes[3].axis('off')

    plt.suptitle(f'样本: {fname} | 基站坐标: {tx_pos}', fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ===================== 测试函数：修复R²计算逻辑 =====================
def test_radiounet(model_path):
    model = RadioWNet(phase="secondU").to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    test_dataset = RadioMapDataset(LABEL_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    rmses, maes, times = [], [], []
    all_preds, all_gts = [], []
    print(f"\n📸 可视化结果将保存至: {VIS_DIR}")

    with torch.no_grad():
        for idx, (inp, gt_radio, building_mask, bs_mask, fname, tx_pos) in enumerate(tqdm(test_loader, desc="测试进度")):
            inp, gt_radio = inp.to(DEVICE), gt_radio.to(DEVICE)
            tx_pos = (tx_pos[0].item(), tx_pos[1].item())
            fname = fname[0]

            t0 = time.time()
            _, pred = model(inp)
            t1 = time.time()

            # 收集数据计算全局R²
            p = pred.squeeze(1).cpu().numpy().flatten()
            t = gt_radio.cpu().numpy().flatten()
            all_preds.append(p)
            all_gts.append(t)

            # 单样本指标
            rmse = np.sqrt(np.mean((p - t) ** 2))
            mae = np.mean(np.abs(p - t))
            rmses.append(rmse)
            maes.append(mae)
            times.append(t1 - t0)

            # 可视化所有样本
            save_name = f"vis_{idx:02d}_{fname.replace('.tif', '.png')}"
            save_path = os.path.join(VIS_DIR, save_name)
            visualize_radio_results(inp, building_mask, bs_mask, gt_radio, pred, save_path, fname, tx_pos)

    # 全局R²计算
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    ss_res = np.sum((all_gts - all_preds) ** 2)
    ss_tot = np.sum((all_gts - all_gts.mean()) ** 2)
    total_r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 输出测试结果
    print("\n" + "=" * 80)
    print("📊 4.3.4 无线电预测模块实验结果（完全修复版）")
    print("=" * 80)
    print(f"平均 RMSE（归一化）: {np.mean(rmses):.4f}")
    print(f"平均 MAE（归一化）: {np.mean(maes):.4f}")
    print(f"全局 R²: {total_r2:.4f}")
    print(f"单张影像平均推理时间: {np.mean(times):.4f} 秒")
    print(f"\n📸 已保存 {len(test_dataset)} 张可视化图片至: {VIS_DIR}")
    print("=" * 80)

    # 保存指标结果
    with open(os.path.join(SAVE_DIR, "test_results.txt"), 'w', encoding='utf-8') as f:
        f.write("4.3.4 无线电预测模块实验结果（完全修复版）\n")
        f.write(f"平均 RMSE: {np.mean(rmses):.4f}\n")
        f.write(f"平均 MAE: {np.mean(maes):.4f}\n")
        f.write(f"全局 R²: {total_r2:.4f}\n")
        f.write(f"单张推理时间: {np.mean(times):.4f} 秒\n")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 加载数据集
    train_dataset = RadioMapDataset(LABEL_DIR, is_train=True)
    val_dataset = RadioMapDataset(LABEL_DIR, is_train=False)
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    # 2. 第一阶段训练（粗预测）
    print("\n🚀 开始第一阶段训练（First U-Net 粗预测）")
    model = RadioWNet(phase="firstU").to(DEVICE)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    model = train_model(model, opt, scheduler, EPOCHS_FIRSTU, "firstU", dataloaders)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "Trained_Model_FirstU.pt"))

    # 3. 第二阶段训练（精修）
    print("\n🚀 开始第二阶段训练（Second U-Net 精修）")
    model = RadioWNet(phase="secondU").to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "Trained_Model_FirstU.pt"), weights_only=True))
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    model = train_model(model, opt, scheduler, EPOCHS_SECONDU, "secondU", dataloaders)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "Trained_Model_SecondU.pt"))

    # 4. 绘制训练损失曲线
    plot_training_loss()

    # 5. 测试与可视化
    test_radiounet(os.path.join(SAVE_DIR, "Trained_Model_SecondU.pt"))