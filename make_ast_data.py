# -*- coding: utf-8 -*-
"""
生成合成"药敏试验培养皿"图像（K-B 纸片扩散法）+ 真值标注。

为什么要合成数据？
  真实药敏图（临床培养皿照片）涉及患者隐私、不易公开获取，
  科研里常用"合成数据 + 已知真值"先验证算法，再上真实数据。
  这里我们先生成带已知抑菌圈直径的图像，用来检验 ast_pipeline.py 判得准不准。

每张图结构：
  浅棕黄 Mueller-Hinton 琼脂培养皿 + 均匀菌苔(深色小点)
  + 4 个白色药敏纸片，每个纸片周围一个清亮无菌的"抑菌圈"

运行方式：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" make_ast_data.py

输出：
  ast_data/plate_*.png              培养皿图像
  ast_data/ground_truth.json        真值：每个纸片的抗生素名和抑菌圈直径(mm)
"""

import os
import json
import random
import numpy as np
import cv2

# 让脚本在自己的目录下运行，避免相对路径出错
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PIXELS_PER_MM = 8.0      # 比例尺：每毫米对应多少像素（关键：像素↔毫米换算）
IMG = 700                # 图像边长
CX, CY = 350, 350        # 培养皿圆心
PLATE_R = 330            # 培养皿半径(px) ≈ 82.5mm
DISK_D = 6.0             # 药敏纸片直径 6mm（K-B 法标准纸片）
NUM_PLATES = 6           # 生成几张培养皿

BACTERIA = "大肠杆菌"    # 本培养皿涂布的菌种（判读时要和 CLSI 表里的菌种对应）
ANTIBIOTICS = ["头孢他啶", "环丙沙星", "阿米卡星", "左氧氟沙星"]  # 4 种抗生素纸片


def add_lawn(img, seed):
    """涂布菌苔：在琼脂上撒满深色小点，模拟细菌均匀生长。"""
    rng = np.random.default_rng(seed)
    n = 40000
    xs = rng.integers(CX - PLATE_R, CX + PLATE_R, n)
    ys = rng.integers(CY - PLATE_R, CY + PLATE_R, n)
    inside = (xs - CX) ** 2 + (ys - CY) ** 2 < PLATE_R ** 2
    img[ys[inside], xs[inside]] = (90, 60, 40)   # 深色菌落点


def main():
    os.makedirs("ast_data", exist_ok=True)
    all_gt = []
    # 4 个纸片放在上下左右四个方位，避免互相重叠
    positions = [(CX, CY - 150), (CX, CY + 150), (CX - 150, CY), (CX + 150, CY)]

    for p in range(NUM_PLATES):
        img = np.full((IMG, IMG, 3), (200, 180, 145), np.uint8)  # 背景
        cv2.circle(img, (CX, CY), PLATE_R, (225, 205, 165), -1)  # 琼脂区
        add_lawn(img, seed=p)                                    # 涂菌苔

        gt_plate = {"plate": p, "bacteria": BACTERIA, "disks": []}
        for ab, (x, y) in zip(ANTIBIOTICS, positions):
            x += random.randint(-12, 12)            # 纸片位置轻微抖动，更真实
            y += random.randint(-12, 12)
            zone_mm = random.uniform(10, 24)        # 抑菌圈直径真值(mm)，覆盖 S/I/R 各区间
            zone_r = int(zone_mm / 2 * PIXELS_PER_MM)  # 抑菌圈半径(px)
            disk_r = int(DISK_D / 2 * PIXELS_PER_MM)   # 纸片半径(px)

            cv2.circle(img, (x, y), zone_r, (225, 205, 165), -1)  # 抑菌圈(清亮无菌)
            cv2.circle(img, (x, y), disk_r, (250, 250, 250), -1)  # 纸片(白色)
            cv2.circle(img, (x, y), disk_r, (60, 60, 60), 1)      # 纸片描边，便于检测

            gt_plate["disks"].append({
                "antibiotic": ab, "x": int(x), "y": int(y),
                "zone_mm": round(zone_mm, 1),
            })

        cv2.imwrite(f"ast_data/plate_{p}.png", img)
        all_gt.append(gt_plate)
        print(f"已生成 ast_data/plate_{p}.png")

    with open("ast_data/ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(all_gt, f, ensure_ascii=False, indent=2)
    print("\n真值标注保存在 ast_data/ground_truth.json")


if __name__ == "__main__":
    main()
