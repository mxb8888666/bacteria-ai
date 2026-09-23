# -*- coding: utf-8 -*-
"""
药敏抑菌圈智能判读系统 —— Streamlit 演示版(Demo)

功能：上传培养皿照片（或一键生成示例图）
     → 自动定位药敏纸片 → 测量抑菌圈直径 → 对照 CLSI 判读 S/I/R
     → 显示标注图（绿圈=抑菌圈边界，蓝圈=纸片）和结果表

两条技术路线可选：
  1. 传统图像处理（阈值分割 + 径向亮度扫描）
  2. U-Net 深度学习（语义分割）

运行：streamlit run app.py
"""

import os
import json
import hashlib
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
from clsi import judge, BREAKPOINTS
from unet import UNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 项目根目录(模型/配置都在这里)

# ================= 常量（传统法，700 尺度，与 ast_pipeline 一致） =================
DEFAULT_PIXELS_PER_MM = 8.0  # 默认比例尺：只用于「生成示例图」（合成图按此绘制）
IMG = 700                    # 培养皿图边长
CX, CY = 350, 350            # 培养皿圆心
PLATE_R = 330                # 培养皿半径(px)
DISK_R_PX = 24               # 示例图纸片半径(px) = 3mm × 8px/mm
DISK_DIAMETER_MM = 6.0       # K-B 药敏纸片标准直径(固定值，不随拍摄距离变)
DISK_R_MM = DISK_DIAMETER_MM / 2.0        # 纸片半径 3mm
DISK_R_MIN, DISK_R_MAX = 8, 80            # 检测纸片的像素半径宽松范围(覆盖不同拍摄距离)
ANTIBIOTICS = ["头孢他啶", "环丙沙星", "阿米卡星", "左氧氟沙星"]
POSITIONS = [(CX, CY - 150), (CX, CY + 150), (CX - 150, CY), (CX + 150, CY)]
ABBR = {"头孢他啶": "CAZ", "环丙沙星": "CIP", "阿米卡星": "AMK", "左氧氟沙星": "LEV"}

# 各菌种在培养基上的特征色(BGR)：与 make_demo_data 的 RGB 特征色一一对应。
# 约束：菌苔灰度必须明显低于琼脂(灰度195)，否则抑菌圈边界检测(亮琼脂→暗菌苔)
# 会失效。深色 4 种用原始鲜艳色；浅色 2 种调暗(灰度<175)并用色相区分。
BACTERIA_COLORS = {
    "金黄色葡萄球菌": (40, 150, 215),    # 金色
    "大肠杆菌":       (168, 172, 172),   # 中性灰白
    "铜绿假单胞菌":   (95, 170, 60),     # 绿色
    "枯草芽孢杆菌":   (120, 180, 200),   # 米黄
    "黏质沙雷菌":     (60, 60, 190),     # 红色
    "白色念珠菌":     (145, 168, 178),   # 奶油偏黄
}

# ================= 常量（U-Net，256 尺度，与 unet_predict 一致） =================
UNET_IMG = 256
UNET_PIXELS_PER_MM = DEFAULT_PIXELS_PER_MM * UNET_IMG / IMG   # 700图缩到256后的默认比例尺 ≈2.93


def antibiotics_for(bacteria):
    """返回某菌种在 CLSI 断点表里收录的抗生素列表（用于下拉框选项）。"""
    return [ab for (bac, ab) in BREAKPOINTS if bac == bacteria]


# ================= 传统法核心算法 =================
def detect_disks(gray):
    """定位纸片：找最亮圆形物体，用半径+圆度过滤误检，最多取 4 个。

    真实照片拍摄距离不定，纸片像素半径会变，所以用宽松的半径范围；
    再用「圆度」(轮廓面积/外接圆面积) 排除长条、不规则等噪声。
    """
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    disks = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)       # 外接圆：圆心 + 半径
        area = cv2.contourArea(c)
        circularity = area / (np.pi * r * r) if r > 0 else 0.0
        # 半径在宽松范围 + 圆度够高(接近圆) + 面积够大，才认定是纸片
        if DISK_R_MIN <= r <= DISK_R_MAX and circularity > 0.5 and area > 100:
            disks.append((x, y, r, area))
    disks.sort(key=lambda d: d[3], reverse=True)   # 按面积从大到小，最多取 10 个候选
    disks = disks[:10]                             # 双校验会进一步筛选，这里多留候选
    # r 保留浮点精度：动态比例尺靠它反推，int 截断会带来约 4% 系统误差
    return [(int(x), int(y), float(r)) for (x, y, r, _) in disks]


def detect_disks_dual(img_bgr, gray):
    """双校验检测纸片：传统阈值法 + U-Net 分割，两个方法都检出同一位置才确认。

    三层防线：
      1. 半径一致性：K-B 纸片直径固定 6mm，真纸片半径应该聚成一簇，
         明显偏大/偏小(相对中位数±40%)的是假阳性(标签/亮斑/杂物)
      2. 位置重合：传统和 U-Net 检出的圆心要重合
      3. 数量上限：确认后最多取 4 个
    U-Net 不可用或没检出时，退回传统结果(不因校验失效而漏检)。
    """
    disks_trad = detect_disks(gray)
    disks_unet = predict_disks_unet(img_bgr)
    if not disks_unet:
        return disks_trad[:4]
    r_med = float(np.median([r for (_, _, r) in disks_trad]))   # 纸片参考半径
    confirmed = []
    for (x, y, r) in disks_trad:
        if not (0.6 * r_med <= r <= 1.4 * r_med):
            continue                            # 大小明显异常的假阳性
        for (ux, uy, ur) in disks_unet:
            if (x - ux) ** 2 + (y - uy) ** 2 <= (r + ur) ** 2:   # 两圆位置重合
                confirmed.append((x, y, r))
                break
    confirmed.sort(key=lambda d: -d[2])
    return confirmed[:4]


def estimate_scale(disks):
    """动态比例尺：K-B 纸片直径固定 6mm，用检测到的纸片像素半径中位数反推 px/mm。

    每张图单独计算，不再写死常数——真实照片拍摄距离不同，比例尺随之改变。
    """
    if not disks:
        return None
    r_med = float(np.median([r for (_, _, r) in disks]))
    if r_med <= 0:
        return None
    return r_med / DISK_R_MM   # 纸片半径 3mm → px/mm


def measure_zone(gray, cx, cy, disk_r, r_max=150):
    """多方向测量抑菌圈：沿 N 个方向各扫一条径向亮度曲线，分别找边界。

    返回 (中位半径px, 各方向半径数组)；测不到返回 (None, None)。
    中位数比单一方向更抗噪、更能代表不规则圈；各方向半径数组用于判断形态是否规则。
    """
    radii = np.arange(disk_r + 2, r_max)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)   # 72 个方向
    H, W = gray.shape
    dir_radii = []
    for a in ang:
        xs = np.clip((cx + radii * np.cos(a)).astype(int), 0, W - 1)
        ys = np.clip((cy + radii * np.sin(a)).astype(int), 0, H - 1)
        profile = gray[ys, xs].astype(float)    # 该方向的亮度曲线
        # 滑动平均平滑(正确处理边缘)，降低菌苔噪点对边界定位的干扰
        half = 2
        profile = np.array([
            profile[max(0, i - half):min(len(profile), i + half + 1)].mean()
            for i in range(len(profile))
        ])
        hi = profile[:10].mean()                # 靠近纸片处(亮)
        lo = profile[-10:].mean()               # 远处(暗)
        thr = (hi + lo) / 2
        below = np.where(profile < thr)[0]
        if len(below):
            dir_radii.append(float(radii[below[0]]))
    if not dir_radii:
        return None, None
    arr = np.array(dir_radii)
    return float(np.median(arr)), arr


def match_antibiotic(x, y):
    """按纸片位置匹配抗生素（纸片放在上下左右四个方位）。"""
    best, best_d = None, 1e9
    for ab, (px, py) in zip(ANTIBIOTICS, POSITIONS):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d, best = d, ab
    return best


def make_sample_plate(bacteria="大肠杆菌"):
    """生成一张示例培养皿（菌苔按菌种特征色 + 4 个药敏纸片），返回 BGR 图。

    菌苔：均匀铺满培养皿、按菌种上色的菌落层 + 轻微纹理噪声
    （贴近真实涂布菌苔，菌种识别模型靠菌苔颜色认菌种）。
    """
    rng = np.random.default_rng()
    img = np.full((IMG, IMG, 3), (200, 180, 145), np.uint8)   # 背景(桌面)
    color = BACTERIA_COLORS.get(bacteria, (90, 60, 40))
    cv2.circle(img, (CX, CY), PLATE_R, color, -1)             # 菌苔(按菌种特征色)
    noise = rng.integers(-15, 16, (IMG, IMG, 3))              # 轻微纹理噪声
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for ab, (px, py) in zip(ANTIBIOTICS, POSITIONS):
        x, y = px + int(rng.integers(-10, 11)), py + int(rng.integers(-10, 11))
        zone_mm = float(rng.uniform(10, 24))
        zone_r = int(zone_mm / 2 * DEFAULT_PIXELS_PER_MM)
        cv2.circle(img, (x, y), zone_r, (225, 205, 165), -1)   # 抑菌圈(琼脂亮，无菌)
        cv2.circle(img, (x, y), DISK_R_PX, (250, 250, 250), -1)  # 纸片(纯白，无黑边)
    return img


def confidence_of(bacteria, ab, zone_mm, cv):
    """判读可信度：直径离 CLSI 断点越近、形态越不规则，可信度越低。

    高=直径远离断点(≥2mm)且形态规则；中=接近断点(1~2mm)；低=贴着断点(<1mm)或不规则。
    """
    b = BREAKPOINTS.get((bacteria, ab))
    if b:
        margins = [abs(zone_mm - v) for v in (b["S"], b["R"]) if v is not None]
        margin = min(margins) if margins else 99.0
    else:
        margin = 99.0
    if margin < 1.0 or cv > 0.15:
        return "低"
    if margin < 2.0:
        return "中"
    return "高"


def analyze(img_bgr, bacteria, ab_map=None):
    """传统法：检测纸片 → 动态算比例尺 → 量直径 → 判读，返回 (标注图, 结果列表)。

    ab_map：{纸片索引: 抗生素名}，由用户在页面手动绑定；
            None 时退回按位置猜(match_antibiotic)。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    disks = detect_disks_dual(img_bgr, gray)   # 双校验：传统+U-Net 都检出才确认
    if not disks:
        return img_bgr, []                     # 没检测到纸片：友好返回空，不崩溃
    scale = estimate_scale(disks)              # 动态比例尺 px/mm
    if scale is None or scale <= 0:
        return img_bgr, []
    r_max = int(scale * 20)                    # 扫描上限：抑菌圈直径按 40mm 计
    results = []
    for i, (x, y, r) in enumerate(disks):
        ab = ab_map.get(i) if ab_map is not None else match_antibiotic(x, y)
        if ab is None:
            continue                           # 该纸片未绑定抗生素，跳过
        r_med, r_arr = measure_zone(gray, x, y, disk_r=r, r_max=r_max)
        if r_med is None:
            continue
        zone_mm = 2 * r_med / scale

        # 形态不规则度：各方向半径的变异系数，过大标记复核
        cv = float(np.std(r_arr) / r_med) if r_med > 0 else 0.0
        note = "形态不规则，建议复核" if cv > 0.15 else ""
        conf = confidence_of(bacteria, ab, zone_mm, cv)   # 判读可信度

        verdict = judge(bacteria, ab, zone_mm)
        results.append({"抗生素": ab, "抑菌圈直径(mm)": round(zone_mm, 1),
                        "判读": verdict, "可信度": conf, "备注": note})
        cv2.circle(img_bgr, (x, y), int(r_med), (0, 255, 0), 2)
        cv2.circle(img_bgr, (x, y), int(r), (255, 0, 0), 2)   # 用检测到的实际半径画纸片圈
        cv2.putText(img_bgr, f"{ABBR.get(ab, ab)} {zone_mm:.1f}mm {verdict}",
                    (x - 60, y - int(r_med) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img_bgr, results


# ================= U-Net 深度学习法 =================
@st.cache_resource
def load_unet():
    """加载训练好的 U-Net（缓存，避免每次交互重新加载）；失败返回 None。"""
    try:
        model = UNet(in_c=3, out_c=1, base=16)
        model_path = os.path.join(BASE_DIR, "unet_best.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception:
        return None


@st.cache_resource
def load_disk_unet():
    """加载纸片分割 U-Net（纸片双校验用）；失败返回 None。"""
    try:
        model = UNet(in_c=3, out_c=1, base=16)
        model_path = os.path.join(BASE_DIR, "disk_unet_best.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception:
        return None


def predict_disks_unet(img_bgr):
    """U-Net 分割纸片：缩到 256 → 分割 → 连通域找纸片。

    返回 700 尺度的纸片列表 [(x, y, r)]；模型不可用或没检出返回 []。
    """
    model = load_disk_unet()
    if model is None:
        return []
    img256 = cv2.resize(img_bgr, (UNET_IMG, UNET_IMG))
    rgb = cv2.cvtColor(img256, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255

    n, labels = cv2.connectedComponents(mask)
    ratio = IMG / UNET_IMG                      # 256 坐标转 700 尺度
    disks = []
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        if len(xs) < 10:                        # 过滤噪声小连通域
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (x, y), r_enc = cv2.minEnclosingCircle(pts)
        r = float(r_enc * ratio)                # 转到 700 尺度
        circularity = len(xs) / (np.pi * r_enc * r_enc) if r_enc > 0 else 0.0
        # 形状过滤：半径范围 + 圆度（排除矩形标签/弧条等非圆白块）
        if DISK_R_MIN <= r <= DISK_R_MAX and circularity > 0.6:
            disks.append((float(x * ratio), float(y * ratio), r))
    return disks


def measure_zone_from_mask(mask, cx, cy, n_angles=72):
    """从掩膜连通域中心向各方向找外边界，返回 (中位半径, 各方向半径数组)。

    从中心先找第一个掩膜点(进入区域)，再找第一个背景点(离开=外边界)，
    对实心圆和环形(中间有纸片洞)都正确。
    """
    H, W = mask.shape
    max_r = int(np.sqrt(H * H + W * W))
    radii = np.arange(0, max_r)
    ang = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    dir_radii = []
    for a in ang:
        xs = np.clip((cx + radii * np.cos(a)).astype(int), 0, W - 1)
        ys = np.clip((cy + radii * np.sin(a)).astype(int), 0, H - 1)
        line = mask[ys, xs]                  # 该方向掩膜值序列(255=抑菌圈内,0=外)
        inside = np.where(line == 255)[0]
        if not len(inside):
            continue
        start = int(inside[0])               # 第一个掩膜点(进入区域)
        rest = line[start:]
        zero_after = np.where(rest == 0)[0]
        if len(zero_after):
            dir_radii.append(float(radii[start + zero_after[0]]))   # 外边界半径
    if not dir_radii:
        return None, None
    arr = np.array(dir_radii)
    return float(np.median(arr)), arr


def analyze_unet(img_bgr, bacteria, ab_map=None):
    """U-Net 法：动态比例尺 → 分割抑菌圈 → 72方向测直径(基于掩膜) → 判读。

    和传统法共用：动态比例尺、抗生素绑定、可信度列、异常容错。
    差别：抑菌圈区域用 U-Net 语义分割(对光照/噪声更鲁棒)，而非径向亮度扫描。
    ab_map：{纸片索引: 抗生素名}；None 时按位置猜。
    """
    model = load_unet()
    if model is None:
        return img_bgr, []

    # 动态比例尺：和传统法同一套(双校验定位纸片 → 反推 px/mm)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    disks = detect_disks_dual(img_bgr, gray)
    scale = estimate_scale(disks) if disks else None
    if scale is None or scale <= 0:
        return img_bgr, []

    # U-Net 分割抑菌圈（256 尺度）
    img256 = cv2.resize(img_bgr, (UNET_IMG, UNET_IMG))
    rgb = cv2.cvtColor(img256, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255

    # 每个连通域：质心 + 72方向测掩膜边界
    n, labels = cv2.connectedComponents(mask)
    ratio = IMG / UNET_IMG                       # 256 → 700 尺度
    results = []
    used = set()
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        if len(xs) < 20:                         # 过滤噪声小连通域
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        if mask[int(cy), int(cx)] == 0:
            continue                             # 质心在掩膜外(如背景环形误分割)，跳过
        r_med, r_arr = measure_zone_from_mask(mask, cx, cy)
        if r_med is None:
            continue
        zone_mm = 2 * r_med * ratio / scale      # 256尺度半径 → 700尺度 → mm

        # 抗生素：用户绑定(匹配最近纸片)或按位置猜
        cx700, cy700 = cx * ratio, cy * ratio
        if ab_map is not None:
            cand = [i for i in range(len(disks)) if i not in used]
            if not cand:
                continue
            best_i = min(cand, key=lambda i: (disks[i][0] - cx700) ** 2
                                              + (disks[i][1] - cy700) ** 2)
            ab = ab_map.get(best_i)
            used.add(best_i)
        else:
            ab = match_antibiotic(cx700, cy700)
        if ab is None:
            continue

        # 不规则度 + 可信度（和传统法一致）
        cv = float(np.std(r_arr) / r_med) if r_med > 0 else 0.0
        note = "形态不规则，建议复核" if cv > 0.15 else ""
        conf = confidence_of(bacteria, ab, zone_mm, cv)
        verdict = judge(bacteria, ab, zone_mm)
        results.append({"抗生素": ab, "抑菌圈直径(mm)": round(zone_mm, 1),
                        "判读": verdict, "可信度": conf, "备注": note})
        cv2.circle(img256, (int(cx), int(cy)), int(r_med), (0, 255, 0), 2)
        cv2.putText(img256, f"{ABBR.get(ab, ab)} {zone_mm:.1f}mm {verdict}",
                    (int(cx) - 50, int(cy) - int(r_med) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 放大回 700 显示，便于和传统法结果对比
    img_annotated = cv2.resize(img256, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
    return img_annotated, results


# ================= 菌种识别（迁移学习 ResNet18） =================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet 归一化约定
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@st.cache_resource
def load_bacteria_model():
    """加载训练好的菌种识别模型(ResNet18) + 类别列表，缓存避免重复加载。"""
    info = json.load(open(os.path.join(BASE_DIR, "model_info.json"), encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(info["classes"]))
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "best_model.pt"), map_location="cpu"))
    model.eval()
    return model, info["classes"]


def predict_bacteria(img_bgr):
    """识别菌种：取培养皿四角纯菌苔的平均色，按训练数据风格构造多个输入、
    平均概率(多种子集成)。纸片在上下左右±150px，四角离纸片中心≈212px，
    大于纸片+抑菌圈最大半径(约120px)，所以四角是纯菌苔。
    返回 (预测类别, 置信度, top3)；任何异常返回 (None, 0.0, [])，不崩页面。
    """
    try:
        model, classes = load_bacteria_model()

        half = 30                     # 每个小块 60x60
        corners = [(CX - 160, CY - 160), (CX + 160, CY - 160),
                   (CX - 160, CY + 160), (CX + 160, CY + 160)]
        colors = []
        for (px, py) in corners:
            crop = img_bgr[py - half:py + half, px - half:px + half]
            colors.append(crop.astype(float).mean(axis=(0, 1)))
        avg_rgb = np.mean(colors, axis=0)[::-1].astype(int)   # 四角平均菌苔色(RGB)

        # 多种子构造"平均色+稀疏噪声"输入(与训练数据同分布)，平均概率消除噪声随机性
        probs = []
        for seed in range(5):
            solid = np.full((224, 224, 3), avg_rgb, np.uint8).copy()
            rng = np.random.default_rng(seed)
            for _ in range(3000):
                x, y = rng.integers(0, 224), rng.integers(0, 224)
                solid[y, x] = np.clip(avg_rgb + rng.integers(-20, 21, 3), 0, 255)
            t = torch.from_numpy(solid.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            t = (t - MEAN) / STD
            with torch.no_grad():
                probs.append(torch.softmax(model(t), dim=1)[0].cpu().numpy())
        prob = np.mean(probs, axis=0)     # 多种子平均概率
        order = np.argsort(prob)[::-1][:3]
        top3 = [(classes[i], float(prob[i])) for i in order]
        return classes[order[0]], float(prob[order[0]]), top3
    except Exception:
        return None, 0.0, []          # 模型缺失/损坏等异常时优雅降级，不崩页面


# ================= 页面 =================
st.set_page_config(page_title="药敏抑菌圈智能判读系统", page_icon="🔬", layout="wide")

# 手机屏幕适配：窄屏时图片、按钮、标题自适应
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .stApp img { max-width: 100% !important; height: auto !important; }
    .stButton button { width: 100% !important; }
    .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    h1 { font-size: 1.5rem !important; }
}
</style>
""", unsafe_allow_html=True)

if "image" not in st.session_state:
    st.session_state.image = None
    st.session_state.results = None
    st.session_state.annotated = None
if "bacteria" not in st.session_state:
    st.session_state.bacteria = "大肠杆菌"
if "pred" not in st.session_state:
    st.session_state.pred = None   # (预测类别, 置信度, top3列表)

BACTERIA_OPTIONS = list(BACTERIA_COLORS.keys())

st.title("药敏抑菌圈智能判读系统")
st.caption("K-B 纸片扩散法 · 定位纸片 → 量抑菌圈 → 识别菌种 → 对照 CLSI 判读 S/I/R")


def style_results(df):
    """可疑结果(可信度=低)整行标黄高亮。"""
    def row_style(row):
        if row.get("可信度") == "低":
            return ["background-color: #FFF3CD"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)


def load_and_detect(img_bgr):
    """载入图片并自动识别菌种（高置信时自动填入下拉框）。"""
    st.session_state.image = img_bgr
    st.session_state.results = None
    st.session_state.annotated = None
    fp = hashlib.md5(img_bgr.tobytes()).hexdigest()
    if st.session_state.get("img_fp") != fp:   # 图片变了才重新识别，避免覆盖手动选择
        st.session_state.img_fp = fp
        pred_name, conf, top3 = predict_bacteria(img_bgr)
        st.session_state.pred = (pred_name, conf, top3)
        if pred_name and conf >= 0.6:
            st.session_state.bacteria = pred_name


# —— 图片来源：上传 或 生成示例 ——
col_up, col_btn = st.columns([3, 1], vertical_alignment="bottom")
with col_up:
    uploaded = st.file_uploader("上传培养皿照片（jpg/png）", type=["jpg", "jpeg", "png"])
with col_btn:
    if st.button("生成示例培养皿", icon=":material/science:", width="stretch"):
        load_and_detect(make_sample_plate(st.session_state.bacteria))

if uploaded is not None:
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.warning("图片解析失败，请上传 jpg/png 格式的照片。")
    else:
        if img_bgr.shape[:2] != (IMG, IMG):
            img_bgr = cv2.resize(img_bgr, (IMG, IMG))
        load_and_detect(img_bgr)

# —— 侧边栏：判读设置 ——
with st.sidebar:
    st.subheader("判读设置")
    bacteria = st.selectbox("菌种", BACTERIA_OPTIONS, key="bacteria")
    pred = st.session_state.get("pred")
    if pred and pred[0]:
        pname, conf, _ = pred
        if conf >= 0.6:
            st.success(f"AI 识别：{pname}（{conf * 100:.0f}%）")
        else:
            st.warning(f"AI 识别：{pname}（{conf * 100:.0f}%，置信度低，请人工确认）")
    st.caption("AI 识别菌种后自动填入；也可手动切换。")

# —— 判读与结果 ——
if st.session_state.image is not None:
    st.divider()
    method = st.segmented_control(
        "判读方法", ["传统图像处理", "U-Net 深度学习"], default="传统图像处理")

    ab_map = None
    disks = []
    # 检测纸片：两个分支都先定位纸片（传统法用它定位+测圈；U-Net 法用它算比例尺+匹配药）
    gray = cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2GRAY)
    disks = detect_disks_dual(st.session_state.image, gray)   # 双校验

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图")
        img_show = st.session_state.image.copy()
        if disks:
            # 图上标注纸片序号，与右侧绑定下拉框一一对应
            for i, (x, y, r) in enumerate(disks):
                cv2.putText(img_show, str(i + 1), (int(x) - 10, int(y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        st.image(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB), width="stretch")

    with col2:
        st.subheader("绑定抗生素")
        ab_options = antibiotics_for(bacteria)
        if not disks:
            st.warning("未检测到药敏纸片，请确认照片清晰、纸片为白色圆片。")
        elif not ab_options:
            st.warning("该菌种暂无 CLSI 断点数据，判读会显示「未知」。请切换其他菌种。")
        else:
            st.caption(f"检测到 {len(disks)} 个纸片，序号已标注在左侧图上，"
                       "请为每个纸片选择对应的抗生素：")
            ab_map = {}
            for i, (x, y, r) in enumerate(disks):
                default_ab = match_antibiotic(x, y)
                default_idx = (ab_options.index(default_ab)
                               if default_ab in ab_options else 0)
                ab_map[i] = st.selectbox(
                    f"纸片 {i + 1}（圆心 {int(x)}, {int(y)}）",
                    ab_options,
                    index=default_idx,
                    key=f"ab_{bacteria}_{i}")

    if st.button("开始判读", type="primary", icon=":material/search:"):
        try:
            img_copy = st.session_state.image.copy()
            if method == "U-Net 深度学习":
                annotated, results = analyze_unet(img_copy, bacteria, ab_map=ab_map)
            else:
                annotated, results = analyze(img_copy, bacteria, ab_map=ab_map)
            st.session_state.annotated = annotated
            st.session_state.results = results
        except Exception as e:
            st.error(f"判读出错：{e}，请换一张更清晰的照片重试。")

    if st.session_state.results is not None:
        st.subheader(f"标注图（{method}）")
        st.image(cv2.cvtColor(st.session_state.annotated, cv2.COLOR_BGR2RGB), width="stretch")
        df = pd.DataFrame(st.session_state.results)
        if df.empty:
            st.warning("未得到判读结果：请确认照片清晰、纸片可见，并为每个纸片绑定抗生素。")
        else:
            st.dataframe(style_results(df), hide_index=True)
            st.caption("黄色行 = 可信度低（直径贴近断点或形态不规则），建议人工复核。")
        st.caption("判读断点为 CLSI 示例值，实际应用以最新版 CLSI M100 为准。")
else:
    st.info("请上传培养皿照片，或点击「生成示例培养皿」开始演示。")
