from typing import List

import torch.nn as nn


class MlpNet(nn.Module):
    """
    用于 MNIST 手写数字分类的多层感知机（MLP）。
    包含至少 2 个隐藏层，使用 ReLU 激活函数。
    """

    def __init__(
        self,
        input_size: int = 784,       # 28×28 展平
        hidden_sizes: List[int] = [256, 128],
        output_size: int = 10,        # 0-9 共 10 个类别
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size

        # 输出层不加激活（CrossEntropyLoss 内含 Softmax）
        layers.append(nn.Linear(in_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 28, 28) → 展平为 (B, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)
