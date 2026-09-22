# -*- coding: utf-8 -*-
"""
K-B 纸片扩散法 药敏抑菌圈自动判读 —— 传统图像处理流水线（完整可运行版）

流程（对应复试要讲的算法思路）：
  1. 检测药敏纸片：白色圆形纸片 -> 亮度阈值分割 + 轮廓拟合圆
  2. 测量抑菌圈：从纸片中心向外做"径向亮度扫描"，找到菌苔边界 -> 得直径
  3. 判读：对照 CLSI 断点表 -> 敏感(S) / 中介(I) / 耐药(R)

运行方式（先运行 make_ast_data.py 生成数据）：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" ast_pipeline.py

输出：
  终端打印每个纸片的 实测直径 / 真值 / 误差 / 判读结果
  ast_result/ 下保存标注图（绿圈=抑菌圈边界，蓝圈=纸片）
"""

import os
import json
import numpy as np
import cv2
from clsi import judge

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PIXELS_PER_MM = 8.0
DISK_R_MM = 3.0
DISK_R_PX = int(DISK_R_MM * PIXELS_PER_MM)   # 纸片半径 24px


def detect_disks(gray):
    """定位纸片：纸片是培养皿里最亮的物体(纯白250)，用高阈值只留下纸片，再拟合圆。"""
    # 阈值分割：灰度 >235 的只剩白色纸片
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    # 找轮廓（兼容不同 OpenCV 版本的返回值个数）
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]

    disks = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)   # 最小外接圆拟合
        if DISK_R_PX - 6 <= r <= DISK_R_PX + 6:  # 半径≈24px 的才是纸片，过滤噪声
            disks.append((int(x), int(y), int(r)))
    return disks


def measure_zone_radius(gray, cx, cy, r_max=150):
    """
    径向亮度扫描：从纸片边缘向外，逐半径采样一圈的平均亮度。
    抑菌圈内(清亮无菌)亮度高，越过边界进入菌苔后亮度下降，
    亮度跌破中点的那个半径，就是抑菌圈边界。
    返回抑菌圈半径(px)，测不到则返回 None。
    """
    radii = np.arange(DISK_R_PX + 2, r_max)
    ang = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    H, W = gray.shape
    means = []
    for r in radii:
        xs = np.clip((cx + r * cos).astype(int), 0, W - 1)
        ys = np.clip((cy + r * sin).astype(int), 0, H - 1)
        means.append(gray[ys, xs].mean())
    means = np.array(means)

    hi = means[:10].mean()    # 靠近纸片处 = 抑菌圈(亮)
    lo = means[-10:].mean()   # 远处 = 菌苔(暗)
    thr = (hi + lo) / 2       # 边界取亮度中点，最稳
    below = np.where(means < thr)[0]
    if len(below) == 0:
        return None
    return int(radii[below[0]])


def match_antibiotic(gt, x, y):
    """按距离找真值里对应的抗生素（实际系统由纸片标记/贴片位置决定）。"""
    best, best_d = None, 1e9
    for d in gt["disks"]:
        dist = (d["x"] - x) ** 2 + (d["y"] - y) ** 2
        if dist < best_d:
            best_d, best = dist, d["antibiotic"]
    return best


def process_plate(img, gt, out_path):
    """处理一张培养皿：检测->测量->判读，并画标注图。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []
    for (x, y, r) in detect_disks(gray):
        r_zone = measure_zone_radius(gray, x, y)
        if r_zone is None:
            continue
        zone_mm = 2 * r_zone / PIXELS_PER_MM      # 像素直径 -> 毫米直径
        ab = match_antibiotic(gt, x, y)
        verdict = judge(gt["bacteria"], ab, zone_mm)
        results.append({"x": int(x), "y": int(y), "antibiotic": ab,
                        "zone_mm": round(zone_mm, 1), "verdict": verdict})

        # 画标注：绿圈=抑菌圈边界，蓝圈=纸片，红字=判读结果
        cv2.circle(img, (x, y), r_zone, (0, 255, 0), 2)
        cv2.circle(img, (x, y), DISK_R_PX, (255, 0, 0), 2)
        cv2.putText(img, f"{ab} {zone_mm:.1f}mm {verdict}",
                    (x - 45, y - r_zone - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(out_path, img)
    return results


def main():
    gt_all = json.load(open("ast_data/ground_truth.json", encoding="utf-8"))
    os.makedirs("ast_result", exist_ok=True)

    for gt in gt_all:
        img = cv2.imread(f"ast_data/plate_{gt['plate']}.png")
        results = process_plate(
            img, gt, f"ast_result/plate_{gt['plate']}_annotated.png")
        print(f"\n培养皿 {gt['plate']}（菌种：{gt['bacteria']}）")
        for r in results:
            true_mm = next(d["zone_mm"] for d in gt["disks"]
                           if d["antibiotic"] == r["antibiotic"])
            err = r["zone_mm"] - true_mm
            print(f"  {r['antibiotic']}: 实测 {r['zone_mm']}mm "
                  f"(真值 {true_mm}mm, 误差 {err:+.1f}mm) -> {r['verdict']}")

    print("\n标注图已保存到 ast_result/ 目录（绿圈=抑菌圈边界，蓝圈=纸片）")


if __name__ == "__main__":
    main()
