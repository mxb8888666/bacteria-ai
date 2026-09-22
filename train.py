# -*- coding: utf-8 -*-
"""
基于迁移学习的细菌图像分类 —— 训练脚本（完整可运行版）

思路（迁移学习）：
  1. 加载 ImageNet 预训练好的 ResNet18（它已经"认识"各种图像的纹理/形状）
  2. 只把最后一层分类头换成"我们的细菌种类数"
  3. 用细菌图片在它基础上微调，小数据也能训出高准确率

数据要求：
  data/ 下每个子文件夹 = 一个细菌种类，里面放该类图片。
  例：data/金黄色葡萄球菌/001.jpg  data/大肠杆菌/001.jpg ...
  脚本会自动按 8:2 划分训练集/验证集，不用手动分。

运行方式（终端）：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" train.py

运行结果：
  训练/验证的准确率曲线(training_curves.png)、混淆矩阵(confusion_matrix.png)
  最佳模型保存为 best_model.pt，类别信息保存为 model_info.json
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接把图存成文件
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 让 matplotlib 能用中文显示（否则混淆矩阵里的中文会变成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 可调参数（按需改这里） ====================
DATA_DIR = "data"                # 图片根目录
IMG_SIZE = 224                   # 输入尺寸
BATCH_SIZE = 16                  # 每批图片数
EPOCHS = 10                      # 训练轮数
LR = 1e-3                        # 学习率
FREEZE_BACKBONE = True           # True=只训练分类头(快)；False=整个网络微调(慢但更准)
VAL_RATIO = 0.2                  # 验证集比例
MODEL_NAME = "resnet18"          # 预训练模型：resnet18 / mobilenet_v2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================================================

# ImageNet 的均值/标准差，预训练模型自带的归一化约定，必须保持一致
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# 训练集：加随机翻转/旋转/颜色抖动做"数据增强"，防止过拟合
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# 验证集：只做固定预处理，不做增强
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageSet(Dataset):
    """把 [(图片路径, 类别编号), ...] 列表包装成 PyTorch 数据集。"""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_data():
    """读取 data/ 下按类分文件夹的图片，自动划分训练/验证集。"""
    full = datasets.ImageFolder(DATA_DIR)     # 自动把子文件夹名当成类别名
    labels = [s[1] for s in full.samples]
    # stratify 保证每类按相同比例划分，避免某个小类全被分进训练集
    train_s, val_s = train_test_split(
        full.samples, test_size=VAL_RATIO, stratify=labels, random_state=42)
    train_loader = DataLoader(ImageSet(train_s, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImageSet(val_s, val_transform),
                            batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, full.classes


def build_model(num_classes):
    """加载预训练模型，替换最后的分类层为我们的细菌种类数。"""
    if MODEL_NAME == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif MODEL_NAME == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier
    else:
        raise ValueError("MODEL_NAME 只支持 resnet18 / mobilenet_v2")

    if FREEZE_BACKBONE:
        # 冻结主干网络，只训练最后的分类层 —— CPU 上也能很快跑完
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    """训练一轮，返回平均损失和准确率。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    """验证集评估：返回损失、准确率、所有预测/真实标签（用于混淆矩阵）。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_history(hist):
    """画训练/验证的损失与准确率曲线，存成图片。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist["train_acc"], label="train")
    axes[1].plot(hist["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=120)


def plot_confusion(labels, preds, class_names):
    """画混淆矩阵，直观看出哪两类最容易混淆。"""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"找不到数据目录 {DATA_DIR}，请先运行 make_demo_data.py 生成演示数据")
        return

    train_loader, val_loader, class_names = load_data()
    print(f"类别数 {len(class_names)} -> {class_names}")
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张")
    print(f"设备: {DEVICE}")

    model = build_model(len(class_names))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc, preds, labels = evaluate(model, val_loader)
        hist["train_loss"].append(t_loss); hist["val_loss"].append(v_loss)
        hist["train_acc"].append(t_acc); hist["val_acc"].append(v_acc)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss {t_loss:.4f} "
              f"train_acc {t_acc:.3f} | val_loss {v_loss:.4f} val_acc {v_acc:.3f}")

        if v_acc > best_acc:               # 保存验证集准确率最高的模型
            best_acc = v_acc
            torch.save(model.state_dict(), "best_model.pt")

    # 保存类别信息，predict.py 预测时要读
    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": MODEL_NAME, "classes": class_names},
                  f, ensure_ascii=False)

    plot_history(hist)
    plot_confusion(labels, preds, class_names)
    print(f"\n最佳验证准确率: {best_acc:.3f}")
    print(classification_report(labels, preds, target_names=class_names))


if __name__ == "__main__":
    main()
