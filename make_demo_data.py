# -*- coding: utf-8 -*-
"""
生成演示用的"合成细菌图像"，用来先跑通整条训练流程。

为什么要这个脚本？
  真实数据集（DIBaS 等）需要去官网申请下载，比较麻烦。
  先用能立即生成的合成数据把 train.py 跑通、确认环境没问题，
  之后再替换成真实细菌显微图即可，训练代码不用改。

运行方式（终端，用你 venv 的完整 python 路径）：
  "D:/Program_Code/python学习库/.venv/Scripts/python.exe" make_demo_data.py

生成结果：
  data/ 目录下，每个细菌种类一个子文件夹，里面是 .jpg 图片。
"""

import os
import random
from PIL import Image, ImageDraw

# 6 种常见细菌/真菌的菌苔特征色(RGB)。金葡菌=金色、铜绿=绿色、黏质沙雷=红色。
# 约束：菌苔灰度必须明显低于琼脂(灰度195)，否则抑菌圈边界检测(亮琼脂→暗菌苔)
# 会失效。深色 4 种用原始鲜艳色；浅色 2 种调暗(灰度<175)并用色相区分：
# 大肠杆菌=中性灰白，白色念珠菌=奶油偏黄。
CLASSES = {
    "金黄色葡萄球菌": (215, 150, 40),
    "大肠杆菌":       (172, 172, 168),
    "铜绿假单胞菌":   (60, 170, 95),
    "枯草芽孢杆菌":   (200, 180, 120),
    "黏质沙雷菌":     (190, 60, 60),
    "白色念珠菌":     (178, 168, 145),
}

IMG_SIZE = 224          # 图片边长，和 train.py 的 Resize 一致
NUM_PER_CLASS = 50      # 每个菌种生成多少张（演示用 50 张足够）


def make_one_image(rgb, out_path):
    """生成一张 224x224 菌苔图：菌种特征色铺满 + 轻微纹理噪声（模拟涂布菌苔）。

    药敏平板(K-B)上菌悬液涂布后长成均匀菌苔，颜色由菌种决定；
    菌种识别模型靠菌苔颜色分辨菌种，所以这里生成"均匀色块+纹理"而非离散菌落。
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), rgb)   # 菌苔特征色铺满
    draw = ImageDraw.Draw(img)
    # 噪声密度/幅度每张随机，逼模型把颜色当主特征、忽略噪声纹理模式
    n = random.randint(500, 5000)
    amp = random.randint(8, 20)
    for _ in range(n):
        x = random.randint(0, IMG_SIZE - 1)
        y = random.randint(0, IMG_SIZE - 1)
        c = tuple(max(0, min(255, v + random.randint(-amp, amp))) for v in rgb)
        draw.point((x, y), fill=c)
    img.save(out_path, quality=92)


def main():
    base = os.path.join(os.path.dirname(__file__), "data")
    for name, rgb in CLASSES.items():
        cls_dir = os.path.join(base, name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(NUM_PER_CLASS):
            make_one_image(rgb, os.path.join(cls_dir, f"{i:03d}.jpg"))
        print(f"已生成 {name}: {NUM_PER_CLASS} 张")
    print(f"\n完成！图片在 {base}，共 {len(CLASSES)} 类。")


if __name__ == "__main__":
    main()
