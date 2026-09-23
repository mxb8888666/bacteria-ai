# -*- coding: utf-8 -*-
"""
训练 U-Net 分割纸片（纸片检测双校验用）。

运行方式（先运行 make_disk_data.py 生成数据）：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" train_disk_unet.py

输出：
  disk_unet_best.pt   最佳模型（Dice 最高的一轮）
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unet import UNet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BATCH = 8
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegDataset(Dataset):
    """读取 (图, 掩膜) 对，转成张量。"""
    def __init__(self, split):
        self.dir = f"disk_data/{split}"
        self.ids = sorted(f[6:9] for f in os.listdir(self.dir)
                          if f.startswith("image"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img = cv2.imread(f"{self.dir}/image_{i}.png")          # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f"{self.dir}/mask_{i}.png", 0)       # 灰度掩膜
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # [0,1]
        return img, mask


def dice(pred, target):
    """Dice 系数：衡量分割重叠度，1=完全重合。"""
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-6)


def main():
    train_loader = DataLoader(SegDataset("train"), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SegDataset("val"), batch_size=BATCH)
    print(f"训练 {len(train_loader.dataset)} 张 / 验证 {len(val_loader.dataset)} 张 | 设备 {DEVICE}")

    model = UNet(in_c=3, out_c=1, base=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # pos_weight：纸片像素只占约 2%，给纸片像素加权，防止 U-Net 学成"全输出0"的偷懒解
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))

    hist = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0
        for img, mask in train_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            tl += loss.item() * img.size(0)
        tl /= len(train_loader.dataset)

        model.eval()
        vl, vd = 0.0, 0.0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                logits = model(img)
                vl += criterion(logits, mask).item() * img.size(0)
                vd += dice(torch.sigmoid(logits), mask).item() * img.size(0)
        vl /= len(val_loader.dataset)
        vd /= len(val_loader.dataset)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["val_dice"].append(vd)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss {tl:.4f} | "
              f"val_loss {vl:.4f} | val_dice {vd:.4f}")

        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), "disk_unet_best.pt")

    print(f"\n最佳验证 Dice: {best_dice:.4f}（1.0=完美分割），模型已保存 disk_unet_best.pt")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["val_dice"]); ax[1].set_title("Val Dice")
    plt.tight_layout(); plt.savefig("disk_unet_training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
