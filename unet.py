# -*- coding: utf-8 -*-
"""
U-Net 语义分割模型（从零手写实现）

U-Net 是 Ronneberger 2015 年提出的经典医学图像分割网络，
名字来自它 U 形的结构：
  左半(编码器)：逐层下采样，提取越来越抽象的特征
  右半(解码器)：逐层上采样，恢复空间分辨率
  中间"跳跃连接"：把编码器的浅层细节直接拼到解码器，保留边缘信息

这里用它来分割"抑菌圈"区域：输入培养皿图，输出每个像素是不是抑菌圈(0/1)。
"""

import torch
import torch.nn as nn


def conv_block(in_c, out_c):
    """U-Net 的基本单元：两个 3x3 卷积 + 批归一化 + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """经典 U-Net：编码器(4次下采样) + 瓶颈 + 解码器(4次上采样) + 跳跃连接。"""
    def __init__(self, in_c=3, out_c=1, base=16):
        super().__init__()
        # 编码器：每次 MaxPool 分辨率减半，通道数翻倍
        self.enc1 = conv_block(in_c, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈：最低分辨率、通道最多
        self.bottleneck = conv_block(base * 8, base * 16)

        # 解码器：转置卷积上采样，再和编码器对应层拼接(跳跃连接)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, out_c, 1)   # 1x1 卷积输出逐像素预测

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # 上采样 + 跳跃连接
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)   # 输出 logits，训练时接 BCEWithLogitsLoss
