"""
HW6: 点云与 PointNet / PointNet++
================================

从头实现 PointNet 和 PointNet++（SSG）分类网络，在 ModelNet10 上做 3D 形状分类。

用法：
    python pointnet_cls.py                     # 依次训练 PointNet 和 PointNet++
    python pointnet_cls.py --models pointnet   # 只训练 PointNet
    python pointnet_cls.py --models pointnetpp # 只训练 PointNet++
    python pointnet_cls.py --epochs 30 --batch_size 32 --num_points 1024

产出：
    assets/pointnet_curves.png   PointNet 的 loss / acc 曲线
    assets/pointnetpp_curves.png PointNet++ 的 loss / acc 曲线
    assets/comparison.png        两个模型的 test accuracy 对比
    assets/pointnet.pt           PointNet 权重（便于关键点可视化）
    assets/pointnetpp.pt         PointNet++ 权重
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
from torch.utils.data import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import (
    Compose,
    NormalizeScale,
    RandomJitter,
    RandomRotate,
    SamplePoints,
)


# =========================================================================== #
# 通用工具
# =========================================================================== #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================== #
# 数据加载：ModelNet10
# =========================================================================== #
MODELNET10_CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


def make_dataloaders(
    root: str,
    num_points: int,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """PyG 的 ModelNet 会从 mesh 中采样点云，返回 PyG Data。
    这里用原生 torch DataLoader + 自定义 collate：把每个样本 reshape 为 (N, 3)，
    再堆叠为 (B, N, 3) 张量，方便我们直接喂给 PointNet/PointNet++。"""
    # pre_transform: 每个样本只做一次（缓存到磁盘）—— 从 mesh 采样 + 归一化到单位球
    pre_transform = Compose([SamplePoints(num=num_points), NormalizeScale()])
    # transform: 每次 epoch 动态应用 —— 数据增强
    # RandomRotate 绕 Y 轴旋转（2D=Y axis）：axis=1
    train_transform = Compose([
        RandomRotate(degrees=180, axis=1),
        RandomJitter(translate=0.02),
    ])

    train_ds = ModelNet(
        root=root, name="10", train=True,
        pre_transform=pre_transform, transform=train_transform,
    )
    test_ds = ModelNet(
        root=root, name="10", train=False,
        pre_transform=pre_transform,
    )

    def collate(batch):
        # batch: list[torch_geometric.data.Data]
        pos = torch.stack([b.pos for b in batch], dim=0)      # (B, N, 3)
        y = torch.tensor([int(b.y) for b in batch], dtype=torch.long)
        return pos, y

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, drop_last=False,
    )
    return train_loader, test_loader


# =========================================================================== #
# PointNet
# =========================================================================== #
class TNet(nn.Module):
    """Spatial Transformer Network: 预测一个 k×k 变换矩阵。
    结构：Conv1d(k→64→128→1024) → MaxPool → FC(1024→512→256→k*k) → +I
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        # 初始化最后一层，让输出 = 预测偏移（初始接近 0），再加 I 得到单位矩阵
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, N) —— 逐点特征。返回 (B, k, k) 的变换矩阵。"""
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]            # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)                       # (B, k*k)
        I = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k)
        x = x + I
        return x.view(B, self.k, self.k)


class PointNetCls(nn.Module):
    """PointNet 分类网络：
        (B, N, 3)
        → InputTNet(3)  → MLP(3→64)
        → FeatureTNet(64) → MLP(64→128→1024)
        → MaxPool → FC(1024→512→256→num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.feature_tnet = TNet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, xyz: torch.Tensor, return_feat_trans: bool = True,
        return_pointwise: bool = False,
    ):
        """xyz: (B, N, 3)
        return_feat_trans=True: 同时返回 Feature T-Net 的 64x64 矩阵（用于正则损失）
        return_pointwise=True: 同时返回 Max Pool 前的逐点特征 (B, 1024, N)（用于关键点可视化）
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2)                        # (B, 3, N)
        trans3 = self.input_tnet(x)                     # (B, 3, 3)
        x = torch.bmm(trans3, x)                        # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))             # (B, 64, N)

        trans64 = self.feature_tnet(x)                  # (B, 64, 64)
        x = torch.bmm(trans64, x)                       # (B, 64, N)

        x = F.relu(self.bn2(self.conv2(x)))             # (B, 128, N)
        x = self.bn3(self.conv3(x))                     # (B, 1024, N)  —— 最后一层先不激活
        pointwise = x                                   # Max Pool 前的逐点特征
        x = F.relu(x)
        x = torch.max(x, dim=2)[0]                      # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)

        outputs = (logits,)
        if return_feat_trans:
            outputs = outputs + (trans64,)
        if return_pointwise:
            outputs = outputs + (pointwise,)
        return outputs if len(outputs) > 1 else outputs[0]


def feature_transform_reg(trans: torch.Tensor) -> torch.Tensor:
    """L_reg = || I - A A^T ||_F^2 的 batch 均值。trans: (B, k, k)。"""
    B, k, _ = trans.shape
    I = torch.eye(k, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    AAT = torch.bmm(trans, trans.transpose(1, 2))
    return ((I - AAT) ** 2).sum(dim=(1, 2)).mean()


# =========================================================================== #
# PointNet++ 核心算子
# =========================================================================== #
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """成对距离平方: src (B, N, 3), dst (B, M, 3) → (B, N, M)."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))          # (B, N, M)
    dist += (src ** 2).sum(dim=-1).view(B, N, 1)
    dist += (dst ** 2).sum(dim=-1).view(B, 1, M)
    return dist.clamp_min_(0.0)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS: 从 (B, N, 3) 选出 npoint 个索引 (B, npoint)，long。
    贪心：每次选与已选集合距离最远的点；distances 维护每个点到已选集合的最近距离。
    第一个点随机选取（训练时相当于隐式数据增强）。
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # 初始点随机（eval 时也保持随机—PointNet++ 论文如此）
    farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)
    batch_idx = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)     # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)              # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points: (B, N, C); idx: (B, M) 或 (B, M, K) → (B, M, C) 或 (B, M, K, C)."""
    B = points.size(0)
    # 构造 batch 索引，前缀 shape 与 idx 对齐
    view_shape = [1] * idx.dim()
    view_shape[0] = B
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def ball_query(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """球查询：xyz (B, N, 3), new_xyz (B, M, 3) → (B, M, nsample) 索引。
    在半径 radius 内找点；若不足 nsample 则用第一个邻居重复填充；超出则截断（排序靠前）。
    """
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape
    device = xyz.device
    sqrdist = square_distance(new_xyz, xyz)                     # (B, M, N)
    # 给半径外的点赋极大距离，排序后自然被踢到后面
    mask_out = sqrdist > radius ** 2
    sqrdist = sqrdist.masked_fill(mask_out, float("inf"))
    # 取每个中心点最近的 nsample 个点（半径外的被掩掉）
    _, idx = sqrdist.topk(nsample, dim=-1, largest=False)       # (B, M, K)
    # 对所有半径外的条目（topk 选到了 inf 的位置），用该中心点第一个有效邻居填充
    # idx[:, :, 0] 是最近点（若存在）；若连最近点都在半径外，说明该中心附近无点——
    # 此时直接用它自己（索引为第一个）。这里我们用 gather 回填：
    first_valid = idx[:, :, :1].clone()                          # (B, M, 1)
    # 判断每个 topk 结果是否为 inf（即半径外）
    group_dist = torch.gather(sqrdist, 2, idx)                   # (B, M, K)
    mask_inf = group_dist == float("inf")
    # 用 first_valid 填充这些无效位置
    idx = torch.where(mask_inf, first_valid.expand_as(idx), idx)
    return idx


class SetAbstraction(nn.Module):
    """PointNet++ 的 Set Abstraction: FPS + Ball Query + 共享 MLP + Max Pool.

    - group_all=True 时做全局聚合：不做 FPS/BQ，所有点作为一个组。
    - mlp 通道列表的第一个元素 = in_channel_plus_xyz，即上一层特征维度 + 3（坐标）
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        layers = []
        last = in_channel                # 注意：包括拼入的 3 个坐标维
        for out_c in mlp:
            layers.append(nn.Conv2d(last, out_c, 1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            last = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """xyz: (B, N, 3); features: (B, N, C) 或 None.
        返回: new_xyz (B, M, 3), new_features (B, M, C')."""
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)            # (B, 1, 3) —— 占位
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)  # 相对坐标 (B, 1, N, 3)
            if features is not None:
                grouped = torch.cat([grouped_xyz, features.unsqueeze(1)], dim=-1)
            else:
                grouped = grouped_xyz
            # (B, 1, N, C+3) → (B, C+3, 1, N) → MLP → MaxPool over N
            grouped = grouped.permute(0, 3, 1, 2)
            grouped = self.mlp(grouped)
            new_features = torch.max(grouped, dim=3)[0]         # (B, C', 1)
            new_features = new_features.transpose(1, 2)         # (B, 1, C')
            return new_xyz, new_features

        # 1) FPS 采样中心点
        fps_idx = farthest_point_sample(xyz, self.npoint)       # (B, M)
        new_xyz = index_points(xyz, fps_idx)                    # (B, M, 3)

        # 2) Ball Query 找邻居
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)   # (B, M, K)
        grouped_xyz = index_points(xyz, idx)                    # (B, M, K, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)        # 减去中心 → 局部坐标

        if features is not None:
            grouped_feat = index_points(features, idx)          # (B, M, K, C)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)   # (B, M, K, C+3)
        else:
            grouped = grouped_xyz                                # (B, M, K, 3)

        # 3) 共享 MLP + Max Pool
        grouped = grouped.permute(0, 3, 1, 2)                    # (B, C+3, M, K)
        grouped = self.mlp(grouped)                              # (B, C', M, K)
        new_features = torch.max(grouped, dim=3)[0]              # (B, C', M)
        new_features = new_features.transpose(1, 2)              # (B, M, C')
        return new_xyz, new_features


class PointNetPP(nn.Module):
    """PointNet++ SSG 分类网络。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        # SA1: mlp input = 3 (local xyz)；无额外特征
        self.sa1 = SetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[64, 64, 128], group_all=False,
        )
        # SA2: mlp input = 128 (SA1 输出特征) + 3 (local xyz)
        self.sa2 = SetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,
        )
        # SA3: 全局，mlp input = 256 + 3
        self.sa3 = SetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) → logits (B, num_classes)."""
        l1_xyz, l1_feat = self.sa1(xyz, None)             # (B, 512, 128)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)       # (B, 128, 256)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)       # (B, 1, 1024)
        x = l3_feat.view(l3_feat.size(0), -1)             # (B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


# =========================================================================== #
# 训练 / 评估循环
# =========================================================================== #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    reg_weight: float = 1e-3,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for pos, y in loader:
        pos = pos.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        if model_name == "pointnet":
            logits, trans64 = model(pos, return_feat_trans=True)
            loss_cls = F.cross_entropy(logits, y)
            loss_reg = feature_transform_reg(trans64)
            loss = loss_cls + reg_weight * loss_reg
        else:  # pointnetpp
            logits = model(pos)
            loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for pos, y in loader:
        pos = pos.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if model_name == "pointnet":
            logits, _ = model(pos, return_feat_trans=True)
        else:
            logits = model(pos)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples


def fit(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    reg_weight: float = 1e-3,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(
            model, train_loader, optimizer, device, model_name, reg_weight,
        )
        vl, va = evaluate(model, test_loader, device, model_name)
        scheduler.step()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)
        best_test_acc = max(best_test_acc, va)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"train loss={tl:.4f} acc={ta*100:.2f}% | "
            f"test loss={vl:.4f} acc={va*100:.2f}% | "
            f"best={best_test_acc*100:.2f}% | lr={optimizer.param_groups[0]['lr']:.5f}"
        )

    elapsed = time.time() - t0
    print(
        f"[{model_name}] 训练完成（{elapsed/60:.1f} 分钟）。"
        f"最终 test acc = {history['test_acc'][-1]*100:.2f}%，"
        f"最佳 test acc = {best_test_acc*100:.2f}%."
    )
    history["best_test_acc"] = best_test_acc
    history["elapsed_sec"] = elapsed
    return history


# =========================================================================== #
# 可视化
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
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    axes[1].plot(epochs, history["test_acc"], label="Test Acc", color="crimson")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02); axes[1].set_title("Accuracy")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


def plot_comparison(
    hist_pn: Dict[str, List[float]],
    hist_pp: Dict[str, List[float]],
    save_path: Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(hist_pn["test_acc"]) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist_pn["test_acc"], label=f"PointNet (best {max(hist_pn['test_acc'])*100:.2f}%)",
            color="steelblue", linewidth=2)
    ax.plot(epochs, hist_pp["test_acc"], label=f"PointNet++ (best {max(hist_pp['test_acc'])*100:.2f}%)",
            color="crimson", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.02); ax.set_title("PointNet vs PointNet++ on ModelNet10", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存到 {save_path}")


# =========================================================================== #
# 主流程
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="HW6: PointNet & PointNet++ on ModelNet10")
    parser.add_argument("--data_root", type=str, default="./data/ModelNet10")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=["pointnet", "pointnetpp"],
                        choices=["pointnet", "pointnetpp"])
    parser.add_argument("--assets_dir", type=str,
                        default=str(Path(__file__).parent / "assets"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device}  |  num_points={args.num_points}  batch_size={args.batch_size}")

    # ---------- 数据 ----------
    train_loader, test_loader = make_dataloaders(
        args.data_root, args.num_points, args.batch_size, args.num_workers,
    )
    print("=" * 60)
    print("ModelNet10 数据集")
    print("=" * 60)
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"类别: {MODELNET10_CLASSES}")
    print("=" * 60)

    histories: Dict[str, Dict[str, List[float]]] = {}

    # ---------- PointNet ----------
    if "pointnet" in args.models:
        print("\n" + "=" * 60)
        print("任务一：PointNet")
        print("=" * 60)
        model = PointNetCls(num_classes=10).to(device)
        print(model)
        print(f"可训练参数量: {count_parameters(model):,}")
        hist = fit(
            model, "pointnet", train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, reg_weight=args.reg_weight,
        )
        histories["pointnet"] = hist
        plot_curves(hist, assets_dir / "pointnet_curves.png",
                    "PointNet on ModelNet10 — Training Dynamics")
        torch.save(model.state_dict(), assets_dir / "pointnet.pt")
        print(f"PointNet 权重已保存到 {assets_dir / 'pointnet.pt'}")

    # ---------- PointNet++ ----------
    if "pointnetpp" in args.models:
        print("\n" + "=" * 60)
        print("任务二：PointNet++ (SSG)")
        print("=" * 60)
        model = PointNetPP(num_classes=10).to(device)
        print(model)
        print(f"可训练参数量: {count_parameters(model):,}")
        hist = fit(
            model, "pointnetpp", train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
        )
        histories["pointnetpp"] = hist
        plot_curves(hist, assets_dir / "pointnetpp_curves.png",
                    "PointNet++ (SSG) on ModelNet10 — Training Dynamics")
        torch.save(model.state_dict(), assets_dir / "pointnetpp.pt")
        print(f"PointNet++ 权重已保存到 {assets_dir / 'pointnetpp.pt'}")

    # ---------- 对比图 ----------
    if "pointnet" in histories and "pointnetpp" in histories:
        plot_comparison(
            histories["pointnet"], histories["pointnetpp"],
            assets_dir / "comparison.png",
        )

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    for name, h in histories.items():
        print(
            f"{name:12s} | final test acc = {h['test_acc'][-1]*100:.2f}%  "
            f"| best test acc = {max(h['test_acc'])*100:.2f}%  "
            f"| 训练耗时 = {h['elapsed_sec']/60:.1f} 分钟"
        )


if __name__ == "__main__":
    main()
