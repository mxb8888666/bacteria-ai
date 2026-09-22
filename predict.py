# -*- coding: utf-8 -*-
"""
用训练好的模型预测单张细菌图片属于哪一类。

运行方式：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" predict.py 图片路径.jpg

会输出这张图最可能是哪类细菌，以及前几名的置信度（概率）。
"""

import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, num_classes):
    """按训练时保存的模型类型，重建一模一样的网络结构。"""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"未知模型 {model_name}")
    return model


def main():
    img_path = sys.argv[1]                       # 命令行传入的图片路径
    info = json.load(open("model_info.json", encoding="utf-8"))
    class_names = info["classes"]

    model = build_model(info["model_name"], len(class_names))
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    # 和验证集相同的预处理
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.softmax(model(img), dim=1)[0]   # 每个类别的概率

    top = torch.topk(prob, k=min(3, len(class_names)))
    print("预测结果：")
    for i in range(len(top.values)):
        print(f"  {class_names[top.indices[i]]}: {top.values[i].item() * 100:.1f}%")


if __name__ == "__main__":
    main()
