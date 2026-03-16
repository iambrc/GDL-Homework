"""
HW3 Task 3 - 特征图可视化

使用 Task 1 训练好的 SimpleCNN，通过 register_forward_hook 捕获
第 1、2、4 卷积块的中间特征图，展示浅层与深层特征的视觉差异。

运行方式：
    python HW3/src/visualize_features.py
    python HW3/src/visualize_features.py --ckpt path/to/checkpoint --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.components.simple_cnn import SimpleCNN


def parse_args():
    default_ckpt = (
        PROJECT_ROOT / "logs" / "train" / "runs" / "2026-03-15_09-31-18"
        / "checkpoints" / "epoch_028.ckpt"
    )
    default_image = PROJECT_ROOT / "data" / "dogs-vs-cats" / "train" / "cat.0.jpg"

    parser = argparse.ArgumentParser(description="HW3 Task3: Feature Map Visualization")
    parser.add_argument("--ckpt", type=str, default=str(default_ckpt),
                        help="模型检查点路径（Lightning .ckpt 或纯 .pth）")
    parser.add_argument("--image", type=str, default=str(default_image),
                        help="用于可视化的输入图像路径")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_channels", type=int, default=16,
                        help="每个卷积块最多展示的通道数")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "visualize"))
    return parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(ckpt_path: str) -> SimpleCNN:
    """从 Lightning checkpoint 或纯 state_dict 加载 SimpleCNN。"""
    model = SimpleCNN(in_channels=3, channels=[32, 64, 128, 256], num_classes=2, dropout=0.5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = {
            k.removeprefix("net."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
        }
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_and_preprocess(image_path: str, image_size: int):
    """加载原始图像并返回 (原图 ndarray, 预处理后的 tensor)。"""
    img = Image.open(image_path).convert("RGB")
    original = np.array(img.resize((image_size, image_size)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(img).unsqueeze(0)
    return original, tensor


def register_hooks(model: SimpleCNN, block_indices: list[int]):
    """在指定卷积块上注册前向钩子，返回 (激活字典, 钩子句柄列表)。"""
    activation = {}

    def make_hook(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    handles = []
    for idx in block_indices:
        name = f"block{idx + 1}"
        h = model.features[idx].register_forward_hook(make_hook(name))
        handles.append(h)
    return activation, handles


def visualize_feature_maps(original: np.ndarray, activation: dict,
                           block_names: list[str], max_channels: int,
                           save_path: str) -> None:
    """将原始图像与各卷积块的特征图绘制在同一张图中。"""
    n_blocks = len(block_names)
    n_cols = max(max_channels, 8)

    fig = plt.figure(figsize=(n_cols * 1.2 + 1, (n_blocks + 1) * 1.8))
    gs = gridspec.GridSpec(n_blocks + 1, n_cols + 1,
                           hspace=0.35, wspace=0.05,
                           left=0.02, right=0.98, top=0.92, bottom=0.02)

    fig.suptitle("Feature Map Visualization — SimpleCNN (Task 1)",
                 fontsize=14, fontweight="bold")

    # --- 原始图像 ---
    ax_img = fig.add_subplot(gs[0, :4])
    ax_img.imshow(original)
    ax_img.set_title("Input Image", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # --- 各卷积块的特征图 ---
    for row, name in enumerate(block_names, start=1):
        feat = activation[name].squeeze(0)  # (C, H, W)
        n_ch = min(feat.shape[0], max_channels)

        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.5, 0.5, name,
                      fontsize=10, fontweight="bold",
                      ha="center", va="center", rotation=0)
        ax_label.axis("off")

        ch_info = f"{feat.shape[0]}ch  {feat.shape[1]}×{feat.shape[2]}"
        ax_label.text(0.5, 0.15, ch_info, fontsize=7,
                      ha="center", va="center", color="gray")

        for c in range(n_ch):
            ax = fig.add_subplot(gs[row, c + 1])
            fmap = feat[c].cpu().numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"特征图可视化已保存至：{save_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型：{args.ckpt}")
    model = load_model(args.ckpt)

    print(f"加载图像：{args.image}")
    original, tensor = load_and_preprocess(args.image, args.image_size)

    block_indices = [0, 1, 3]  # 第 1、2、4 卷积块
    activation, handles = register_hooks(model, block_indices)

    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(1).item()
    label_name = "dog" if pred == 1 else "cat"
    print(f"模型预测：{label_name} (class {pred})")

    for h in handles:
        h.remove()

    block_names = [f"block{i + 1}" for i in block_indices]
    save_path = output_dir / "feature_maps.png"
    visualize_feature_maps(original, activation, block_names, args.max_channels, str(save_path))

    print("\n===== 浅层 vs. 深层特征分析 =====")
    for name in block_names:
        feat = activation[name].squeeze(0)
        print(f"  {name}: shape {tuple(feat.shape)}, "
              f"mean {feat.mean():.4f}, std {feat.std():.4f}, "
              f"sparsity {(feat == 0).float().mean() * 100:.1f}%")

    print("""
分析结论：
  - block1（浅层）：通道少、空间分辨率高，特征图保留了大量边缘、纹理等底层视觉信息，
    与原始图像的空间结构高度相似。
  - block2（中层）：空间尺寸减半，开始组合底层特征，捕捉更复杂的纹理模式和局部结构。
  - block4（深层）：通道数多、空间尺寸很小，特征高度抽象和稀疏（ReLU 后大量激活为零），
    编码了高级语义信息（如"耳朵""眼睛"等部件），与具体像素的对应关系变弱。
  浅层 → 深层：空间分辨率递减，语义抽象程度递增，稀疏度升高。
""")


if __name__ == "__main__":
    main()
