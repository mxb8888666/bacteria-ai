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
import numpy as np
import cv2
import pandas as pd
import torch
import streamlit as st
from clsi import judge
from unet import UNet

# ================= 常量（传统法，700 尺度，与 ast_pipeline 一致） =================
PIXELS_PER_MM = 8.0          # 比例尺：每毫米对应像素数
IMG = 700                    # 培养皿图边长
CX, CY = 350, 350            # 培养皿圆心
PLATE_R = 330                # 培养皿半径(px)
DISK_R_PX = 24               # 药敏纸片半径(px)
ANTIBIOTICS = ["头孢他啶", "环丙沙星", "阿米卡星", "左氧氟沙星"]
POSITIONS = [(CX, CY - 150), (CX, CY + 150), (CX - 150, CY), (CX + 150, CY)]
ABBR = {"头孢他啶": "CAZ", "环丙沙星": "CIP", "阿米卡星": "AMK", "左氧氟沙星": "LEV"}

# ================= 常量（U-Net，256 尺度，与 unet_predict 一致） =================
UNET_IMG = 256
UNET_PIXELS_PER_MM = PIXELS_PER_MM * UNET_IMG / IMG   # 700图缩到256后的比例尺 ≈2.93


# ================= 传统法核心算法 =================
def detect_disks(gray):
    """定位纸片：纸片是培养皿里最亮的物体，阈值分割 + 轮廓拟合圆。"""
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    disks = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        if DISK_R_PX - 6 <= r <= DISK_R_PX + 6:
            disks.append((int(x), int(y), int(r)))
    return disks


def measure_zone_radius(gray, cx, cy, r_max=150):
    """径向亮度扫描：从纸片中心向外，亮度突降处就是抑菌圈边界，返回半径(px)。"""
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
    thr = (means[:10].mean() + means[-10:].mean()) / 2
    below = np.where(means < thr)[0]
    return int(radii[below[0]]) if len(below) else None


def match_antibiotic(x, y):
    """按纸片位置匹配抗生素（纸片放在上下左右四个方位）。"""
    best, best_d = None, 1e9
    for ab, (px, py) in zip(ANTIBIOTICS, POSITIONS):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d, best = d, ab
    return best


def make_sample_plate():
    """生成一张示例培养皿（菌苔 + 4 个药敏纸片），返回 BGR 图。"""
    rng = np.random.default_rng()
    img = np.full((IMG, IMG, 3), (200, 180, 145), np.uint8)
    cv2.circle(img, (CX, CY), PLATE_R, (225, 205, 165), -1)
    n = 40000
    xs = rng.integers(CX - PLATE_R, CX + PLATE_R, n)
    ys = rng.integers(CY - PLATE_R, CY + PLATE_R, n)
    inside = (xs - CX) ** 2 + (ys - CY) ** 2 < PLATE_R ** 2
    img[ys[inside], xs[inside]] = (90, 60, 40)
    for ab, (px, py) in zip(ANTIBIOTICS, POSITIONS):
        x, y = px + int(rng.integers(-10, 11)), py + int(rng.integers(-10, 11))
        zone_mm = float(rng.uniform(10, 24))
        zone_r = int(zone_mm / 2 * PIXELS_PER_MM)
        cv2.circle(img, (x, y), zone_r, (225, 205, 165), -1)
        cv2.circle(img, (x, y), DISK_R_PX, (250, 250, 250), -1)
        cv2.circle(img, (x, y), DISK_R_PX, (60, 60, 60), 1)
    return img


def analyze(img_bgr, bacteria):
    """传统法：检测纸片 → 量直径 → 判读，返回 (标注图, 结果列表)。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for (x, y, r) in detect_disks(gray):
        r_zone = measure_zone_radius(gray, x, y)
        if r_zone is None:
            continue
        zone_mm = 2 * r_zone / PIXELS_PER_MM
        ab = match_antibiotic(x, y)
        verdict = judge(bacteria, ab, zone_mm)
        results.append({"抗生素": ab, "抑菌圈直径(mm)": round(zone_mm, 1), "判读": verdict})
        cv2.circle(img_bgr, (x, y), r_zone, (0, 255, 0), 2)
        cv2.circle(img_bgr, (x, y), DISK_R_PX, (255, 0, 0), 2)
        cv2.putText(img_bgr, f"{ABBR[ab]} {zone_mm:.1f}mm {verdict}",
                    (x - 60, y - r_zone - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img_bgr, results


# ================= U-Net 深度学习法 =================
@st.cache_resource
def load_unet():
    """加载训练好的 U-Net（缓存，避免每次交互重新加载）。"""
    model = UNet(in_c=3, out_c=1, base=16)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet_best.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def analyze_unet(img_bgr, bacteria):
    """U-Net 法：缩放到 256 → 分割抑菌圈 → 量直径 → 判读，返回 (标注图, 结果列表)。"""
    model = load_unet()
    # 缩放到 U-Net 训练尺寸 256
    img256 = cv2.resize(img_bgr, (UNET_IMG, UNET_IMG))
    rgb = cv2.cvtColor(img256, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255

    # 从分割掩膜找连通域、拟合圆、量直径
    n, labels = cv2.connectedComponents(mask)
    results = []
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        if len(xs) < 20:                       # 过滤噪声小连通域
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (x, y), r = cv2.minEnclosingCircle(pts)
        zone_mm = 2 * r / UNET_PIXELS_PER_MM
        # 256 坐标换算回 700 尺度，用于匹配抗生素位置
        ab = match_antibiotic(x * IMG / UNET_IMG, y * IMG / UNET_IMG)
        verdict = judge(bacteria, ab, zone_mm)
        results.append({"抗生素": ab, "抑菌圈直径(mm)": round(zone_mm, 1), "判读": verdict})
        cv2.circle(img256, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(img256, f"{ABBR[ab]} {zone_mm:.1f}mm {verdict}",
                    (int(x) - 50, int(y) - int(r) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 放大回 700 显示，便于和传统法结果对比
    img_annotated = cv2.resize(img256, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
    return img_annotated, results


# ================= 页面 =================
st.set_page_config(page_title="药敏抑菌圈智能判读系统", page_icon="🔬", layout="wide")

if "image" not in st.session_state:
    st.session_state.image = None
    st.session_state.results = None
    st.session_state.annotated = None

st.title("药敏抑菌圈智能判读系统")
st.caption("K-B 纸片扩散法 · 自动定位纸片 → 测量抑菌圈直径 → 对照 CLSI 判读 S/I/R")

with st.sidebar:
    st.subheader("判读设置")
    bacteria = st.selectbox("菌种", ["大肠杆菌", "金黄色葡萄球菌"])
    st.caption("示例培养皿按大肠杆菌生成；切换菌种会改变 CLSI 判读断点。")

# —— 图片来源：上传 或 生成示例 ——
col_up, col_btn = st.columns([3, 1], vertical_alignment="bottom")
with col_up:
    uploaded = st.file_uploader("上传培养皿照片（jpg/png）", type=["jpg", "jpeg", "png"])
with col_btn:
    if st.button("生成示例培养皿", icon=":material/science:", width="stretch"):
        st.session_state.image = make_sample_plate()
        st.session_state.results = None

if uploaded is not None:
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        if img_bgr.shape[:2] != (IMG, IMG):
            img_bgr = cv2.resize(img_bgr, (IMG, IMG))
        st.session_state.image = img_bgr
        st.session_state.results = None

# —— 判读与结果 ——
if st.session_state.image is not None:
    st.divider()
    method = st.segmented_control(
        "判读方法", ["传统图像处理", "U-Net 深度学习"], default="传统图像处理")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图")
        st.image(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB), width="stretch")
    with col2:
        st.subheader("判读结果")
        if st.session_state.results is None:
            st.info("点击下方「开始判读」查看结果")
    if st.button("开始判读", type="primary", icon=":material/search:"):
        img_copy = st.session_state.image.copy()
        if method == "U-Net 深度学习":
            annotated, results = analyze_unet(img_copy, bacteria)
        else:
            annotated, results = analyze(img_copy, bacteria)
        st.session_state.annotated = annotated
        st.session_state.results = results
    if st.session_state.results is not None:
        st.subheader(f"标注图（{method}）")
        st.image(cv2.cvtColor(st.session_state.annotated, cv2.COLOR_BGR2RGB), width="stretch")
        st.dataframe(pd.DataFrame(st.session_state.results), hide_index=True)
        st.caption("判读断点为 CLSI 示例值，实际应用以最新版 CLSI M100 为准。")
else:
    st.info("请上传培养皿照片，或点击「生成示例培养皿」开始演示。")
