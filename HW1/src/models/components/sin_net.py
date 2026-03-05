import torch.nn as nn

class SinNet(nn.Module):
    """
    一个包含单隐藏层的简单神经网络，用于拟合正弦函数。
    """
    def __init__(self, hidden_size: int = 64, activation: str = "tanh"):
        super().__init__()
        
        # 根据传入的参数动态选择激活函数
        activation = activation.lower()
        if activation == "tanh":
            act_layer = nn.Tanh()
        elif activation == "sigmoid":
            act_layer = nn.Sigmoid()
        elif activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "leaky_relu":
            act_layer = nn.LeakyReLU()
        elif activation == "prelu":
            act_layer = nn.PReLU()
        elif activation == "elu":
            act_layer = nn.ELU()
        elif activation == "gelu":
            act_layer = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 使用 Sequential 快速搭建：输入层(1) -> 隐藏层(hidden_size) -> 输出层(1)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            act_layer,
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)
