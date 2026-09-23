# -*- coding: utf-8 -*-
"""
生成纸片分割 U-Net 训练数据：合成培养皿图 + 纸片位置掩膜。

用途：纸片检测【双校验】——传统阈值法 + U-Net 分割，两个方法都检出同一位置
才确认真是纸片，大幅降低真实照片的假阳性误检。

和 make_unet_data.py 的差别：掩膜只标【纸片圆】，不标抑菌圈。
菌苔颜色每张随机(6 种菌色)，让 U-Net 学到"白色圆=纸片"且与菌苔颜色无关。

运行方式：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" make_disk_data.py

输出：
  disk_data/train/  image_xxx.png + mask_xxx.png（160 对）
  disk_data/val/    image_xxx.png + mask_xxx.png（30 对）
"""

import os
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

IMG = 256
PIXELS_PER_MM = 3.0          # 比例尺：256px ≈ 85mm 培养皿（与 app.py 的 256 尺度一致）
CX, CY = 128, 128            # 培养皿圆心
PLATE_R = 118                # 培养皿半径(px)
DISK_R = int(3 * PIXELS_PER_MM)   # 纸片半径 3mm ≈ 9px
N_TRAIN, N_VAL = 160, 30     # 训练/验证样本数

# 菌苔颜色用 app.py 的 BACTERIA_COLORS（BGR），训练与部署分布一致
BACTERIA_COLORS = [(40, 150, 215), (168, 172, 172), (95, 170, 60),
                   (120, 180, 200), (60, 60, 190), (145, 168, 178)]


def make_plate(rng):
    """生成一张培养皿图和纸片掩膜，返回 (img, mask)。"""
    img = np.full((IMG, IMG, 3), (200, 180, 145), np.uint8)   # 背景(桌面)
    color = BACTERIA_COLORS[int(rng.integers(0, len(BACTERIA_COLORS)))]
    cv2.circle(img, (CX, CY), PLATE_R, color, -1)             # 菌苔(随机菌色)
    noise = rng.integers(-15, 16, (IMG, IMG, 3))              # 轻微纹理
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask = np.zeros((IMG, IMG), np.uint8)          # 分割掩膜：纸片=255
    disks = []
    for _ in range(int(rng.integers(2, 5))):       # 每张 2~4 个纸片
        for _ in range(50):                        # 随机位置，尽量不重叠
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(30, PLATE_R - 50)
            x = int(CX + rad * np.cos(ang))
            y = int(CY + rad * np.sin(ang))
            if all((x - dx) ** 2 + (y - dy) ** 2 > 80 ** 2 for dx, dy in disks):
                break
        zone_r = int(rng.uniform(8, 22) / 2 * PIXELS_PER_MM)  # 抑菌圈半径
        cv2.circle(img, (x, y), zone_r, (225, 205, 165), -1)  # 抑菌圈(琼脂亮)
        cv2.circle(img, (x, y), DISK_R, (250, 250, 250), -1)  # 纸片(纯白，无黑边)
        cv2.circle(mask, (x, y), DISK_R, 255, -1)             # 掩膜=纸片
        disks.append((x, y))

    # 径向光照渐变(中心亮边缘暗) + 高斯噪声，让 U-Net 比固定阈值更鲁棒
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / PLATE_R
    img = (img * (1.0 - 0.25 * dist)[..., None]).astype(np.uint8)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    # 负样本：随机加白色干扰块(标签/亮斑)，掩膜不标它们，
    # 让 U-Net 学会"只有培养皿内的圆形小纸片才是纸片"，其他白块不算
    if rng.random() < 0.6:
        for _ in range(int(rng.integers(1, 3))):
            kind = int(rng.integers(0, 3))
            if kind == 0:                    # 白色矩形标签
                x, y = int(rng.integers(0, IMG - 110)), int(rng.integers(0, IMG - 70))
                w, h = int(rng.integers(60, 130)), int(rng.integers(40, 80))
                cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 250), -1)
            elif kind == 1:                  # 大圆亮斑(比纸片大得多)
                x, y = int(rng.integers(20, IMG - 20)), int(rng.integers(20, IMG - 20))
                cv2.circle(img, (x, y), int(rng.integers(30, 60)), (245, 245, 245), -1)
            else:                            # 白色弧条(反光)
                x, y = int(rng.integers(0, IMG - 120)), int(rng.integers(0, IMG - 30))
                cv2.ellipse(img, (x, y), (60, 15), 0, 0, 360, (248, 248, 248), -1)
    return img, mask


def main():
    for split, n, base_seed in [("train", N_TRAIN, 0), ("val", N_VAL, 100000)]:
        os.makedirs(f"disk_data/{split}", exist_ok=True)
        for i in range(n):
            rng = np.random.default_rng(base_seed + i)
            img, mask = make_plate(rng)
            cv2.imwrite(f"disk_data/{split}/image_{i:03d}.png", img)
            cv2.imwrite(f"disk_data/{split}/mask_{i:03d}.png", mask)
        print(f"已生成 {split} {n} 对 (图+掩膜)")


if __name__ == "__main__":
    main()
