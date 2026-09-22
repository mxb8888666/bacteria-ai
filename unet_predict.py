# -*- coding: utf-8 -*-
"""
用训练好的 U-Net 分割抑菌圈，测量直径并判读（替代传统阈值法）。

流程：输入培养皿图 -> U-Net 输出逐像素分割掩膜 -> 找连通域拟合圆
     -> 得抑菌圈直径 -> 对照 CLSI 判读 S/I/R

运行方式（先运行 train_unet.py 得到 unet_best.pt）：
  PYTHONIOENCODING=utf-8 "D:/Program_Code/python学习库/.venv/Scripts/python.exe" unet_predict.py

输出：
  终端打印每个纸片的 实测/真值/误差/判读结果
  unet_result/ 下保存分割结果标注图
"""

import os
import json
import torch
import numpy as np
import cv2
from unet import UNet
from clsi import judge

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PIXELS_PER_MM = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_from_mask(mask):
    """从分割掩膜找每个抑菌圈连通域，拟合圆，返回 (x, y, 直径mm) 列表。"""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    zones = []
    for lab in range(1, n):                # 0 是背景，跳过
        ys, xs = np.where(labels == lab)
        if len(xs) < 20:                   # 过滤噪声小连通域
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (x, y), r = cv2.minEnclosingCircle(pts)   # 最小外接圆拟合
        zones.append((int(x), int(y), 2 * r / PIXELS_PER_MM))
    return zones


def main():
    model = UNet(in_c=3, out_c=1, base=16)
    model.load_state_dict(torch.load("unet_best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    gt_all = json.load(open("unet_data/test_gt.json", encoding="utf-8"))
    os.makedirs("unet_result", exist_ok=True)

    for gt in gt_all:
        img_bgr = cv2.imread(f"unet_data/test/plate_{gt['plate']}.png")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(t.to(DEVICE)))[0, 0].cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8) * 255

        zones = measure_from_mask(mask)
        print(f"\n培养皿 {gt['plate']}（菌种：{gt['bacteria']}）")
        for d in gt["disks"]:
            if not zones:
                print(f"  {d['antibiotic']}: 未检测到")
                continue
            z = min(zones, key=lambda z: (z[0] - d["x"]) ** 2 + (z[1] - d["y"]) ** 2)
            err = z[2] - d["zone_mm"]
            verdict = judge(gt["bacteria"], d["antibiotic"], z[2])
            print(f"  {d['antibiotic']}: 实测 {z[2]:.1f}mm "
                  f"(真值 {d['zone_mm']}mm, 误差 {err:+.1f}mm) -> {verdict}")
            r_px = int(z[2] / 2 * PIXELS_PER_MM)
            cv2.circle(img_bgr, (z[0], z[1]), r_px, (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{d['antibiotic']} {z[2]:.1f}mm {verdict}",
                        (z[0] - 40, z[1] - r_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(f"unet_result/plate_{gt['plate']}_unet.png", img_bgr)

    print("\nU-Net 分割结果图已保存到 unet_result/（绿圈=分割出的抑菌圈）")


if __name__ == "__main__":
    main()
