"""
HW3 Task 1 - 从零搭建的 SimpleCNN
4 个卷积块 (Conv → BN → ReLU → MaxPool)，通道数 32 → 64 → 128 → 256
AdaptiveAvgPool2d(1) 压缩空间维度，Dropout + 全连接分类头
"""

from typing import List

import torch.nn as nn


class ConvBlock(nn.Module):
    """基础卷积块：Conv → BN → ReLU → MaxPool"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    用于猫狗二分类的简单卷积神经网络。
    默认 4 个卷积块，通道逐步翻倍：32 → 64 → 128 → 256。
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        blocks = []
        ch_in = in_channels
        for ch_out in channels:
            blocks.append(ConvBlock(ch_in, ch_out))
            ch_in = ch_out

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
