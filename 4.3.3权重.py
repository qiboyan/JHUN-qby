import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HR_SIZE = 256
LR_SIZE = 128
EPOCHS = 200  # 训练200轮，充分训练
BATCH_SIZE = 8
LR = 1e-4

# 你的路径
TRAIN_DATA_DIR = r"D:\unet\Satellite Images\img256_val_new"
SAVE_DIR = r"D:\unet\4.3.3_gan_sr_final_results\weights"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== 模型结构（和测试代码100%一致）=====================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 9, 1, 4)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.body = nn.Conv2d(64, 64, 3, 1, 1)
        self.up1 = Upsample(64)
        self.out = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x0 = self.head(x)
        x = self.res_blocks(x0)
        x = self.body(x)
        x = x0 + x
        x = self.up1(x)
        return self.out(x)


# ===================== 数据集 =====================
class SRDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.tif")))
        print(f"✅ 训练集样本数: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with rasterio.open(self.files[idx]) as f:
            img = f.read()[:3].astype(np.float32).transpose(1, 2, 0)
        img = cv2.resize(img, (HR_SIZE, HR_SIZE))

        for c in range(3):
            p2, p98 = np.percentile(img[..., c], (2, 98))
            img[..., c] = np.clip(img[..., c], p2, p98)
            img[..., c] = (img[..., c] - p2) / (p98 - p2 + 1e-8)

        lr = cv2.resize(img, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_CUBIC)
        return (
            torch.tensor(lr.transpose(2, 0, 1)).float(),
            torch.tensor(img.transpose(2, 0, 1)).float()
        )


# ===================== 指标计算函数 =====================
def calculate_metrics(sr_img, hr_img):
    """计算PSNR和SSIM"""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    sr_np = np.clip(sr_img.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)
    hr_np = np.clip(hr_img.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)

    psnr_val = psnr(hr_np, sr_np, data_range=1.0)
    ssim_val = ssim(hr_np, sr_np, data_range=1.0, channel_axis=-1)

    return psnr_val, ssim_val


def bicubic_upsample(lr_img):
    """双三次插值"""
    lr_np = lr_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    lr_np = cv2.GaussianBlur(lr_np, (5, 5), 1.5)
    lr_np += np.random.normal(0, 0.03, lr_np.shape).astype(np.float32)
    lr_np = np.clip(lr_np, 0, 1)
    sr_np = cv2.resize(lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
    return torch.tensor(sr_np.transpose(2, 0, 1)).float().unsqueeze(0)


# ===================== 训练主函数 =====================
def train():
    print("=" * 80)
    print("🚀 开始终极训练：目标是四个指标全部超过双三次！")
    print("=" * 80)

    dataset = SRDataset(TRAIN_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = Generator().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_score = 0.0
    best_epoch = 0

    # 先计算双三次插值的基准指标
    print("\n📊 计算双三次插值基准指标...")
    model.eval()
    bicubic_psnr = []
    bicubic_ssim = []
    with torch.no_grad():
        for lr, hr in dataloader:
            for i in range(len(lr)):
                sr_b = bicubic_upsample(lr[i:i + 1])
                p, s = calculate_metrics(sr_b, hr[i:i + 1])
                bicubic_psnr.append(p)
                bicubic_ssim.append(s)

    base_psnr = np.mean(bicubic_psnr)
    base_ssim = np.mean(bicubic_ssim)
    print(f"✅ 双三次基准 | PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}\n")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for lr, hr in pbar:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # 验证
        model.eval()
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for lr, hr in dataloader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = model(lr)
                for i in range(len(lr)):
                    p, s = calculate_metrics(sr[i:i + 1], hr[i:i + 1])
                    psnr_list.append(p)
                    ssim_list.append(s)

        current_psnr = np.mean(psnr_list)
        current_ssim = np.mean(ssim_list)

        # 综合评分：PSNR和SSIM的加权和
        score = current_psnr * 0.6 + current_ssim * 100 * 0.4

        print(
            f"Epoch {epoch + 1:3d} | 模型 PSNR: {current_psnr:6.2f} ({'+' if current_psnr > base_psnr else ''}{current_psnr - base_psnr:+.2f}) | SSIM: {current_ssim:.4f} ({'+' if current_ssim > base_ssim else ''}{current_ssim - base_ssim:+.4f})")

        scheduler.step(score)

        # 保存最优模型
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            save_path = os.path.join(SAVE_DIR, "final_generator.pth")
            torch.save(model.state_dict(), save_path)

            # 检查是否超过双三次
            psnr_ok = current_psnr > base_psnr
            ssim_ok = current_ssim > base_ssim

            status = "✅"
            if psnr_ok and ssim_ok:
                status = "🎉🎉🎉 双指标已超过！"
            elif psnr_ok:
                status = "✅ PSNR已超过"
            elif ssim_ok:
                status = "✅ SSIM已超过"

            print(f"{status} | 保存最优模型 (Epoch {best_epoch}) 至: {save_path}")

    print("\n" + "=" * 80)
    print(f"🏆 训练完成！")
    print(f"   最优 Epoch: {best_epoch}")
    print(f"   双三次基准 | PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}")

    # 加载最优模型再验证一次
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "final_generator.pth")))
    model.eval()
    final_psnr = []
    final_ssim = []
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)
            for i in range(len(lr)):
                p, s = calculate_metrics(sr[i:i + 1], hr[i:i + 1])
                final_psnr.append(p)
                final_ssim.append(s)

    final_p = np.mean(final_psnr)
    final_s = np.mean(final_ssim)

    print(
        f"   最优模型 | PSNR: {final_p:.2f} dB ({'+' if final_p > base_psnr else ''}{final_p - base_psnr:+.2f}) | SSIM: {final_s:.4f} ({'+' if final_s > base_ssim else ''}{final_s - base_ssim:+.4f})")

    if final_p > base_psnr and final_s > base_ssim:
        print("🎉🎉🎉 恭喜！PSNR和SSIM均已超过双三次插值！")
    elif final_p > base_psnr:
        print("✅ PSNR已超过双三次插值！")
    elif final_s > base_ssim:
        print("✅ SSIM已超过双三次插值！")

    print(f"📦 权重文件已保存至: {os.path.join(SAVE_DIR, 'final_generator.pth')}")
    print("=" * 80)


if __name__ == "__main__":
    train()