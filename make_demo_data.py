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

# 6 种常见细菌/真菌，颜色模拟它们在培养基上的真实菌落特征色
# 金黄色葡萄球菌=金色、铜绿假单胞菌=绿色、黏质沙雷菌=红色，都是临床检验的真实特征
CLASSES = {
    "金黄色葡萄球菌": (215, 150, 40),
    "大肠杆菌": (235, 235, 225),
    "铜绿假单胞菌": (60, 170, 95),
    "枯草芽孢杆菌": (200, 180, 120),
    "黏质沙雷菌": (190, 60, 60),
    "白色念珠菌": (245, 245, 245),
}

IMG_SIZE = 224          # 图片边长，和 train.py 的 Resize 一致
NUM_PER_CLASS = 50      # 每个菌种生成多少张（演示用 50 张足够）


def make_one_image(rgb, out_path):
    """生成一张 224x224 图像：浅色培养基背景 + 若干该菌种的菌落圆点。"""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (245, 242, 230))  # 培养基浅黄背景
    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(5, 15)):          # 每张图随机 5~15 个菌落
        x = random.randint(10, IMG_SIZE - 10)       # 随机位置
        y = random.randint(10, IMG_SIZE - 10)
        r = random.randint(6, 20)                   # 随机大小
        # 对基准色做轻微抖动，让同类内部也有差异，更接近真实拍摄
        color = tuple(max(0, min(255, c + random.randint(-18, 18))) for c in rgb)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
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
