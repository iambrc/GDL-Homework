import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def plot_activation_functions():
    # 构造 x 轴输入，范围从 -5 到 5，共 1000 个取样点
    x_input = torch.linspace(-5, 5, 1000)
    x_np = x_input.numpy()

    # 准备要在图表中展示的激活函数模型
    activations = {
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "ReLU": nn.ReLU(),
        "Leaky ReLU (slope=0.1)": nn.LeakyReLU(negative_slope=0.1),
        # PReLU 默认 weight 也是 0.25 (这里展示一下效果)
        "PReLU (init=0.25)": nn.PReLU(num_parameters=1, init=0.25),
        "ELU (alpha=1.0)": nn.ELU(alpha=1.0),
        "GELU": nn.GELU()
    }

    # 设置matplotlib绘图配置 (设置图像大小和网格)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle('Common Activation Functions in Deep Learning', fontsize=18)
    
    # 将二维数组轴变平以方便遍历
    axes = axes.flatten()

    for idx, (name, act_fn) in enumerate(activations.items()):
        # 获取张量的结果并转化为 NumPy 用于绘图
        with torch.no_dict_warning() if hasattr(torch, "no_dict_warning") else torch.no_grad():
             y_np = act_fn(x_input).detach().numpy()
             
        ax = axes[idx]
        ax.plot(x_np, y_np, color='b', linewidth=2.5)
        ax.set_title(name, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 补充 x=0 和 y=0 的参考线
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        
        # 为了对比直观，统一坐标限制
        if name == "Sigmoid":
            ax.set_ylim(-0.1, 1.1)
        elif name == "Tanh":
            ax.set_ylim(-1.5, 1.5)
        else:
            ax.set_ylim(-2.0, 5.0)
            
    # 因为我们只有 7 个函数，但是画布是 2x4=8，所以删除第 8 个空白子图
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 给主标题留点空
    
    # 确保保存目录存在
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "activation_functions_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已成功保存至: {save_path}")
    
    plt.show() # 如果只需要存图就不弹出了

if __name__ == "__main__":
    plot_activation_functions()
