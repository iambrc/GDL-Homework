import os
import torch
import glob
import matplotlib.pyplot as plt

# 为了应对 PyTorch 2.6 的安全策略，由于保存了 Hydra 的超参配置，我们需要解锁 omegaconf 基础对象的禁止
import functools
import omegaconf
import typing
import collections
torch.serialization.add_safe_globals([
    functools.partial, 
    torch.optim.Adam, 
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode,
    omegaconf.nodes.StringNode,
    omegaconf.nodes.BooleanNode,
    omegaconf.nodes.IntegerNode,
    omegaconf.nodes.FloatNode,
    omegaconf.base.Container,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    list,
    dict,
    int,
    collections.defaultdict,
    omegaconf.base.Metadata
])

# 引入你的 LightningModule
from models.sin_module import SinLitModule
from models.components.sin_net import SinNet

def generate_predictions():
    # 构造你要预测的 X 参数区间 (与训练集的区间一致，例如 -2pi 到 2pi)
    x_input = torch.linspace(-6.283, 6.283, 500).unsqueeze(1)
    
    # 手动定义你的模型所在的 checkpoint 文件夹，从之前的历史记录映射：
    # （注：你刚才跑完训练的日志保存在 HW1/logs/train/runs 下）
    ckpt_dirs = {
        "Tanh": "Tanh",
        "ReLU": "ReLU",
        "Sigmoid": "Sigmoid",
        "Leaky ReLU": "Leaky_ReLU",
        "PReLU": "PReLU",
        "ELU": "ELU",
        "GELU": "GELU"
    }
    
    base_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "train", "runs")

    # 创建一块 2x4 的画板
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle('Fitting Results of Different Activation Functions', fontsize=16)
    axes = axes.flatten()

    for idx, (act_name, run_dir) in enumerate(ckpt_dirs.items()):
        ax = axes[idx]
        
        # 绘制背景的真实正弦函数
        y_true = torch.sin(x_input)
        ax.plot(x_input.numpy(), y_true.numpy(), color='black', linestyle='--', label='True Sin(x)', alpha=0.6)

        # 寻找对应运行记录文件夹中的检查点 (通常是 last.ckpt 或者 epoch_xxx.ckpt)
        ckpt_folder = os.path.join(base_log_dir, run_dir, "checkpoints")
        ckpt_files = glob.glob(os.path.join(ckpt_folder, "*.ckpt"))
        
        if not ckpt_files:
            print(f"[{act_name}] 未找到 checkpoint, 跳过...")
            ax.set_title(f"{act_name} (Missing ckpt)")
            continue
            
        # 选择最后一个或者最佳的 checkpoint（这里只取第一个找到的）
        ckpt_path = ckpt_files[0]
        
        # 使用 PyTorch Lightning 内置的加载方法加载权重
        print(f"加载 [{act_name}] 模型从: {run_dir}")
        net = SinNet(activation=run_dir.lower())
        model = SinLitModule.load_from_checkpoint(ckpt_path, net=net)
        
        # 将模型设置为评估模式并禁用梯度
        model.eval()
        with torch.no_grad():
            preds = model(x_input)

        # 把预测的结果画成红色的点状图
        ax.scatter(x_input.numpy(), preds.numpy(), color='red', s=5, label='Predicted', alpha=0.8)
        
        # 美化图表
        ax.set_title(f"Model Prediction: {act_name}", fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

    # 我们只有 7 个结果，将第 8 个空白子图删除
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 统一保存到 assets 下
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "fitting_results_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"已成功生成对比报告图: {save_path}")

if __name__ == "__main__":
    generate_predictions()