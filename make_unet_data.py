# -*- coding: utf-8 -*-
"""
生成 U-Net 训练数据：合成药敏培养皿图 + 抑菌圈分割掩膜。

和 ast_pipeline 的差别：这里不仅生成图，还生成【每个像素的标签】——
抑菌圈区域标 1，其他标 0，这就是"语义分割"训练要的 (图, 掩膜) 对。

为了让 U-Net 有"用武之地"（而不是简单阈值就能搞定），加入了：
  1. 径向光照渐变（中心亮、边缘暗）—— 全局阈值法会失效
  2. 高斯噪声
  这样 U-Net 学到的是"理解空间结构"，比固定阈值更鲁棒。

运行方式：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" make_unet_data.py

输出：
  unet_data/train/  image_xxx.png + mask_xxx.png（160 对）
  unet_data/val/    image_xxx.png + mask_xxx.png（30 对）
  unet_data/test/   plate_x.png（6 张测试图）+ test_gt.json（真值）
"""

import os
import json
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

IMG = 256
PIXELS_PER_MM = 3.0          # 比例尺：256px ≈ 85mm 培养皿
CX, CY = 128, 128            # 培养皿圆心
PLATE_R = 118                # 培养皿半径(px)
DISK_R = int(3 * PIXELS_PER_MM)   # 纸片半径 3mm ≈ 9px
N_TRAIN, N_VAL = 160, 30     # 训练/验证样本数
ANTIBIOTICS = ["头孢他啶", "环丙沙星", "阿米卡星", "左氧氟沙星"]


def add_lawn(img, rng):
    """涂布菌苔：深色小点模拟细菌均匀生长。"""
    n = 12000
    xs = rng.integers(CX - PLATE_R, CX + PLATE_R, n)
    ys = rng.integers(CY - PLATE_R, CY + PLATE_R, n)
    inside = (xs - CX) ** 2 + (ys - CY) ** 2 < PLATE_R ** 2
    img[ys[inside], xs[inside]] = (90, 60, 40)


def add_illumination(img):
    """径向光照渐变：中心亮、边缘暗 25%，模拟真实拍照光照不均。"""
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / PLATE_R
    factor = 1.0 - 0.25 * dist
    return (img * factor[..., None]).astype(np.uint8)


def make_plate(rng):
    """生成一张培养皿图和对应掩膜，返回 (img, mask, disks_info)。"""
    img = np.full((IMG, IMG, 3), (200, 180, 145), np.uint8)
    cv2.circle(img, (CX, CY), PLATE_R, (225, 205, 165), -1)
    add_lawn(img, rng)

    mask = np.zeros((IMG, IMG), np.uint8)          # 分割掩膜：抑菌圈=255
    disks = []
    for _ in range(int(rng.integers(2, 5))):       # 每张 2~4 个纸片
        for _ in range(50):                        # 随机位置，尽量不重叠
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(30, PLATE_R - 50)
            x = int(CX + rad * np.cos(ang))
            y = int(CY + rad * np.sin(ang))
            if all((x - dx) ** 2 + (y - dy) ** 2 > 80 ** 2 for dx, dy, _ in disks):
                break
        zone_mm = rng.uniform(8, 22)               # 抑菌圈直径真值(mm)
        zone_r = int(zone_mm / 2 * PIXELS_PER_MM)  # 抑菌圈半径(px)
        cv2.circle(img, (x, y), zone_r, (225, 205, 165), -1)   # 抑菌圈(清亮)
        cv2.circle(img, (x, y), DISK_R, (250, 250, 250), -1)   # 纸片(白)
        cv2.circle(img, (x, y), DISK_R, (60, 60, 60), 1)
        cv2.circle(mask, (x, y), zone_r, 255, -1)              # 掩膜=整个抑菌圈
        disks.append((x, y, zone_mm))

    img = add_illumination(img)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    return img, mask, disks


def main():
    for split, n, base_seed in [("train", N_TRAIN, 0), ("val", N_VAL, 100000)]:
        os.makedirs(f"unet_data/{split}", exist_ok=True)
        for i in range(n):
            rng = np.random.default_rng(base_seed + i)
            img, mask, _ = make_plate(rng)
            cv2.imwrite(f"unet_data/{split}/image_{i:03d}.png", img)
            cv2.imwrite(f"unet_data/{split}/mask_{i:03d}.png", mask)
        print(f"已生成 {split} {n} 对 (图+掩膜)")

    # 测试集：6 张 + 真值（用于和传统方法对比判读精度）
    os.makedirs("unet_data/test", exist_ok=True)
    gt = []
    for i in range(6):
        rng = np.random.default_rng(200000 + i)
        img, mask, disks = make_plate(rng)
        cv2.imwrite(f"unet_data/test/plate_{i}.png", img)
        cv2.imwrite(f"unet_data/test/plate_{i}_mask.png", mask)
        gt.append({"plate": i, "bacteria": "大肠杆菌",
                   "disks": [{"antibiotic": ANTIBIOTICS[j % 4], "x": d[0], "y": d[1],
                              "zone_mm": round(d[2], 1)} for j, d in enumerate(disks)]})
    with open("unet_data/test_gt.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print("测试集 6 张 + 真值已生成到 unet_data/test/")


if __name__ == "__main__":
    main()
