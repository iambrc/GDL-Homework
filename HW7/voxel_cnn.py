"""
HW7: 体素网格与 3D 卷积神经网络
================================

将 ModelNet10 点云体素化为 32×32×32 的二值占据网格，并训练一个 3D CNN
分类器（3 个 Conv3d 卷积块 + FC 分类头）。

用法：
    python voxel_cnn.py
    python voxel_cnn.py --epochs 30 --batch_size 32 --resolution 32
    python voxel_cnn.py --data_root ../HW6/data/ModelNet10  # 复用作业六缓存

产出（默认写入 assets/）：
    assets/voxel_vis.png         体素可视化图（4 个不同类别）
    assets/training_curves.png   训练 loss / accuracy 曲线
    assets/voxel_cnn.pt          训练好的 3D CNN 权重
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints


# =========================================================================== #
# 常量与工具
# =========================================================================== #
MODELNET10_CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================== #
# 任务一：点云 → 体素网格
# =========================================================================== #
def point_cloud_to_voxel(pos: torch.Tensor, resolution: int = 32) -> torch.Tensor:
    """把一个点云转成 (R, R, R) 二值占据网格。

    步骤：
        1) 把 pos 平移到边界框中心、再缩放到 [0, 1]^3；
        2) 离散化到 [0, R-1] 整数索引；
        3) 把所有有点落入的体素置 1.

    Args:
        pos: (N, 3) 点云坐标（float）。
        resolution: 体素网格分辨率 R。
    Returns:
        grid: (R, R, R) 的 float 张量，元素 ∈ {0, 1}。
    """
    pos = pos.float()

    # ------- Step 1: 归一化到 [0, 1]^3 -------
    # 用 axis-aligned bounding box（AABB）的中心 + 最长边作尺度
    # 这样保证形状不会被各向异性的拉伸压扁。
    p_min = pos.min(dim=0).values                  # (3,)
    p_max = pos.max(dim=0).values                  # (3,)
    center = (p_min + p_max) / 2.0
    extent = (p_max - p_min).max()                 # 最长边长度
    if extent < 1e-8:                              # 退化情形：所有点重合
        extent = torch.tensor(1.0, device=pos.device)
    pos = (pos - center) / extent + 0.5             # → 大致落在 [0, 1]^3

    # ------- Step 2: 离散化到网格索引 -------
    idx = (pos * resolution).long().clamp(0, resolution - 1)   # (N, 3)

    # ------- Step 3: 创建空网格并标记占据 -------
    grid = torch.zeros(resolution, resolution, resolution, dtype=torch.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


class VoxelDataset(Dataset):
    """把 PyG 的 ModelNet 数据集预先体素化，存到内存里供 DataLoader 取用。

    每个 sample 返回 ((1, R, R, R) 体素张量, 标签) ——单通道 3D 张量，
    与作业三的 (1, H, W) 单通道图像一一对应。
    """

    def __init__(self, pyg_dataset, resolution: int = 32, verbose: bool = True):
        self.resolution = resolution
        self.voxels: List[torch.Tensor] = []
        self.labels: List[int] = []

        n = len(pyg_dataset)
        report_every = max(n // 10, 1)
        for i in range(n):
            data = pyg_dataset[i]
            grid = point_cloud_to_voxel(data.pos, resolution)
            self.voxels.append(grid)
            self.labels.append(int(data.y))
            if verbose and (i + 1) % report_every == 0:
                print(f"  体素化进度: {i + 1}/{n}  ({(i + 1) / n * 100:.0f}%)")

        self.voxels = torch.stack(self.voxels, dim=0)              # (M, R, R, R)
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # (M,)

        # 顺手报告一下平均占据率，便于和报告中的 Q2 相互印证
        occ = (self.voxels > 0.5).float().mean().item()
        print(
            f"  完成：{len(self.labels)} 个样本  |  平均占据率 ρ = {occ * 100:.2f}% "
            f"(理论 ~1/R = {1.0 / resolution * 100:.2f}%)"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.voxels[idx].unsqueeze(0), self.labels[idx]      # (1, R, R, R)


def make_dataloaders(
    root: str,
    num_points: int,
    resolution: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, VoxelDataset, VoxelDataset]:
    """加载 ModelNet10、采样点云、体素化、封装 DataLoader。"""
    # 与作业六保持一致的 pre_transform，可直接复用 ./data/ModelNet10/processed 缓存
    pre_transform = Compose([SamplePoints(num=num_points), NormalizeScale()])

    train_pyg = ModelNet(root=root, name="10", train=True, pre_transform=pre_transform)
    test_pyg = ModelNet(root=root, name="10", train=False, pre_transform=pre_transform)

    print(f"PyG ModelNet10  训练: {len(train_pyg)}  测试: {len(test_pyg)}  类别: 10")

    print("体素化训练集 ...")
    train_ds = VoxelDataset(train_pyg, resolution=resolution)
    print("体素化测试集 ...")
    test_ds = VoxelDataset(test_pyg, resolution=resolution)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, test_loader, train_ds, test_ds


# =========================================================================== #
# 体素可视化
# =========================================================================== #
def visualize_voxels(
    dataset: VoxelDataset,
    save_path: Path,
    num_samples: int = 4,
    seed: int = 0,
) -> None:
    """从数据集中挑 num_samples 个不同类别的体素样本画 3D 散点。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 选 num_samples 个不同类别，每类挑第一个样本
    rng = random.Random(seed)
    classes_seen: Dict[int, int] = {}
    chosen: List[Tuple[int, int]] = []                       # (idx_in_dataset, class)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for i in indices:
        c = int(dataset.labels[i])
        if c not in classes_seen:
            classes_seen[c] = i
            chosen.append((i, c))
            if len(chosen) >= num_samples:
                break

    fig = plt.figure(figsize=(4.2 * num_samples, 4.5))
    fig.suptitle("Voxelized ModelNet10 Samples (32^3)", fontsize=14, fontweight="bold")
    for k, (idx, cls) in enumerate(chosen):
        grid = dataset.voxels[idx].numpy()                   # (R, R, R)
        occupied = np.argwhere(grid > 0.5)                   # (Nocc, 3)
        ax = fig.add_subplot(1, num_samples, k + 1, projection="3d")
        ax.scatter(
            occupied[:, 0], occupied[:, 1], occupied[:, 2],
            c=occupied[:, 2], cmap="viridis", marker="s", s=8, alpha=0.85,
        )
        R = dataset.resolution
        ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_zlim(0, R)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(
            f"{MODELNET10_CLASSES[cls]}\nN_occupied = {len(occupied)}",
            fontsize=11,
        )
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"体素可视化已保存到 {save_path}")


# =========================================================================== #
# 任务二：3D CNN 模型
# =========================================================================== #
class VoxelCNN(nn.Module):
    """3 层 Conv3d + FC 分类头。每个卷积块: Conv3d → BN3d → ReLU → MaxPool3d.

    输入: (B, 1, 32, 32, 32)  →  输出: (B, num_classes)

    参数量约 2.3M（绝大部分在 Linear(8192, 256)）。
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                              # 32 → 16
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                              # 16 → 8
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                              # 8 → 4
        )
        # 4 × 4 × 4 × 128 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# =========================================================================== #
# 训练 / 评估
# =========================================================================== #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for voxels, labels in loader:
        voxels = voxels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(voxels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for voxels, labels in loader:
        voxels = voxels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(voxels)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "epoch_time": [],
    }
    best_test_acc = 0.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        e_start = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, device)
        vl, va = evaluate(model, test_loader, device)
        scheduler.step()
        e_time = time.time() - e_start

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)
        history["epoch_time"].append(e_time)
        best_test_acc = max(best_test_acc, va)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={tl:.4f} acc={ta * 100:.2f}% | "
            f"test loss={vl:.4f} acc={va * 100:.2f}% | "
            f"best={best_test_acc * 100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.5f} | "
            f"{e_time:.1f}s"
        )

    elapsed = time.time() - t0
    print(
        f"训练完成（共 {elapsed / 60:.1f} 分钟）。"
        f"final test acc = {history['test_acc'][-1] * 100:.2f}%, "
        f"best test acc = {best_test_acc * 100:.2f}%."
    )
    history["best_test_acc"] = best_test_acc
    history["elapsed_sec"] = elapsed
    return history


# =========================================================================== #
# 训练曲线绘制
# =========================================================================== #
def plot_curves(history: Dict[str, List[float]], save_path: Path, title: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = list(range(1, len(history["train_loss"]) + 1))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["test_loss"], label="Test Loss", color="crimson")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    axes[1].plot(epochs, history["test_acc"], label="Test Acc", color="crimson")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02); axes[1].set_title("Accuracy")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


# =========================================================================== #
# 主流程
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW7: Voxel + 3D CNN on ModelNet10",
    )
    parser.add_argument("--data_root", type=str,
                        default=str(Path(__file__).parent.parent / "HW6/data/ModelNet10"),
                        help="ModelNet10 数据根目录（默认复用作业六的缓存）")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--assets_dir", type=str,
        default=str(Path(__file__).parent / "assets"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"设备: {device}  |  resolution={args.resolution}  "
        f"num_points={args.num_points}  batch_size={args.batch_size}  "
        f"epochs={args.epochs}"
    )

    # ---------- 1) 数据 ----------
    print("=" * 64)
    print("任务一：加载 ModelNet10 + 体素化")
    print("=" * 64)
    train_loader, test_loader, train_ds, _ = make_dataloaders(
        root=args.data_root,
        num_points=args.num_points,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"类别: {MODELNET10_CLASSES}")

    # ---------- 2) 体素可视化 ----------
    visualize_voxels(
        train_ds,
        save_path=assets_dir / "voxel_vis.png",
        num_samples=4,
        seed=args.seed,
    )

    # ---------- 3) 模型 + 训练 ----------
    print("\n" + "=" * 64)
    print("任务二：3D CNN 训练")
    print("=" * 64)
    model = VoxelCNN(num_classes=len(MODELNET10_CLASSES)).to(device)
    print(model)
    n_params = count_parameters(model)
    print(f"可训练参数量: {n_params:,}  (~{n_params / 1e6:.2f} M)")

    history = fit(
        model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr,
    )

    # ---------- 4) 训练曲线 + 权重 ----------
    plot_curves(
        history,
        save_path=assets_dir / "training_curves.png",
        title="3D CNN on ModelNet10 — Training Dynamics",
    )
    torch.save(model.state_dict(), assets_dir / "voxel_cnn.pt")
    print(f"模型权重已保存到 {assets_dir / 'voxel_cnn.pt'}")

    # ---------- 5) 单样本推理时间（便于报告对比）----------
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, args.resolution, args.resolution, args.resolution,
                        device=device)
        # warm up
        for _ in range(5):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        N_RUN = 50
        for _ in range(N_RUN):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_sample_ms = (time.time() - t0) / N_RUN * 1000.0

    avg_epoch = float(np.mean(history["epoch_time"]))
    print("\n" + "=" * 64)
    print("结果汇总")
    print("=" * 64)
    print(f"final test acc      : {history['test_acc'][-1] * 100:.2f}%")
    print(f"best test acc       : {history['best_test_acc'] * 100:.2f}%")
    print(f"参数量              : {n_params / 1e6:.2f} M")
    print(f"平均单 epoch 用时   : {avg_epoch:.1f} s")
    print(f"单样本推理时间      : {per_sample_ms:.2f} ms  (batch=1, {device})")
    print(
        f"输入数据大小        : "
        f"1 × {args.resolution}^3 = {args.resolution ** 3} 体素 / 样本"
    )


if __name__ == "__main__":
    main()
