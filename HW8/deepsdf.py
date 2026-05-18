"""
HW8: 隐式表示 — SDF 与 DeepSDF
================================

用一个神经网络（DeepSDF 自解码器）参数化多个解析基元形状（球体 / 圆环 /
长方体）的 Signed Distance Function，并用 Marching Cubes 从训练好的网络
中提取三角网格。

用法：
    python deepsdf.py
    python deepsdf.py --epochs 500 --latent_dim 32 --resolution 64
    python deepsdf.py --no_interp                # 跳过隐空间插值（可选任务）

产出（默认写入 assets/）：
    assets/sdf_slices.png              三种形状的 SDF 切面热力图（z=0）
    assets/deepsdf_loss.png            训练 loss 曲线
    assets/deepsdf_reconstructions.png 三类代表形状的 Marching Cubes 重建
    assets/deepsdf_interpolation.png   （可选）隐空间插值结果
    assets/deepsdf.pt                  训练好的网络权重 + 隐编码
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# =========================================================================== #
# 通用工具
# =========================================================================== #
SHAPE_NAMES = ["sphere", "torus", "box"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================== #
# 任务一：解析 SDF 函数
# =========================================================================== #
def sdf_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """球体 SDF: f(x) = ||x - c|| - r."""
    return np.linalg.norm(points - center, axis=-1) - radius


def sdf_torus(points: np.ndarray, center: np.ndarray, R: float, r: float) -> np.ndarray:
    """圆环 SDF（环面位于 y=0 的 xz 平面）:
    q = sqrt(x^2 + z^2) - R,  f = sqrt(q^2 + y^2) - r."""
    p = points - center
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    q = np.sqrt(x * x + z * z) - R
    return np.sqrt(q * q + y * y) - r


def sdf_box(points: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """长方体 SDF（轴对齐）:
        外部距离 = ||max(q, 0)||,  内部距离 = min(max(qx,qy,qz), 0).
    """
    q = np.abs(points - center) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


# --------------------------------------------------------------------------- #
# 表面附近采样器（保证训练数据在 |SDF|≈0 的近表面区域有足够密度）
# --------------------------------------------------------------------------- #
def _sample_sphere_surface(num: int, center: np.ndarray, radius: float) -> np.ndarray:
    """在球面上均匀采样。"""
    u = np.random.randn(num, 3).astype(np.float32)
    u /= np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9
    return center + radius * u


def _sample_torus_surface(num: int, center: np.ndarray, R: float, r: float) -> np.ndarray:
    """在圆环面上采样（按 (R + r cosφ) 加权 φ 以保证均匀分布；这里近似用均匀）。"""
    theta = np.random.uniform(0.0, 2 * np.pi, size=num).astype(np.float32)
    phi = np.random.uniform(0.0, 2 * np.pi, size=num).astype(np.float32)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    z = (R + r * np.cos(phi)) * np.sin(theta)
    y = r * np.sin(phi)
    return center + np.stack([x, y, z], axis=-1)


def _sample_box_surface(num: int, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """在长方体六个面上按面积加权采样。"""
    bx, by, bz = half_extents
    areas = np.array([by * bz, by * bz, bx * bz, bx * bz, bx * by, bx * by])
    probs = areas / areas.sum()
    face_idx = np.random.choice(6, size=num, p=probs)

    pts = np.empty((num, 3), dtype=np.float32)
    for i, axis in enumerate([0, 0, 1, 1, 2, 2]):
        mask = face_idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        sign = 1.0 if i % 2 == 0 else -1.0
        # 该面所在轴固定 ±b, 其它两轴均匀
        coords = np.empty((n, 3), dtype=np.float32)
        for ax in range(3):
            if ax == axis:
                coords[:, ax] = sign * half_extents[ax]
            else:
                coords[:, ax] = np.random.uniform(-half_extents[ax], half_extents[ax], n)
        pts[mask] = coords
    return center + pts


def _mixed_sample(
    surface_fn,
    sdf_fn,
    num_pts: int,
    near_sigma: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """混合采样：50% 均匀 + 50% 表面附近高斯扰动。"""
    n_unif = num_pts // 2
    n_surf = num_pts - n_unif

    pts_unif = np.random.uniform(-1.0, 1.0, (n_unif, 3)).astype(np.float32)
    surf_pts = surface_fn(n_surf)
    noise = np.random.randn(n_surf, 3).astype(np.float32) * near_sigma
    pts_surf = (surf_pts + noise).astype(np.float32)

    pts = np.concatenate([pts_unif, pts_surf], axis=0)
    np.random.shuffle(pts)
    sdf = sdf_fn(pts).astype(np.float32)
    return pts, sdf


# --------------------------------------------------------------------------- #
# 数据集生成
# --------------------------------------------------------------------------- #
def generate_sdf_dataset(
    num_per_class: int = 15,
    num_pts: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """为三类基元形状（球 / 圆环 / 盒）生成带参数变化的 SDF 训练数据。

    Returns
    -------
    pts:  (num_shapes * num_pts, 3) 查询点（float32）
    sdf:  (num_shapes * num_pts,)    真实 SDF 值（float32）
    ids:  (num_shapes * num_pts,)    形状编号（int64）
    info: List[Dict]                 每个形状的元信息（type / label / params）
    """
    all_pts: List[np.ndarray] = []
    all_sdf: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    info: List[Dict] = []
    shape_id = 0

    # -------- 球体 --------
    for _ in range(num_per_class):
        r = float(np.random.uniform(0.25, 0.55))
        c = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        pts, sdf = _mixed_sample(
            surface_fn=lambda n, c=c, r=r: _sample_sphere_surface(n, c, r),
            sdf_fn=lambda x, c=c, r=r: sdf_sphere(x, c, r),
            num_pts=num_pts,
        )
        all_pts.append(pts)
        all_sdf.append(sdf)
        all_ids.append(np.full(num_pts, shape_id, dtype=np.int64))
        info.append(dict(type="sphere", label=0, center=c, radius=r, shape_id=shape_id))
        shape_id += 1

    # -------- 圆环 --------
    for _ in range(num_per_class):
        R = float(np.random.uniform(0.30, 0.50))
        r = float(np.random.uniform(0.08, 0.18))
        c = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        pts, sdf = _mixed_sample(
            surface_fn=lambda n, c=c, R=R, r=r: _sample_torus_surface(n, c, R, r),
            sdf_fn=lambda x, c=c, R=R, r=r: sdf_torus(x, c, R, r),
            num_pts=num_pts,
        )
        all_pts.append(pts)
        all_sdf.append(sdf)
        all_ids.append(np.full(num_pts, shape_id, dtype=np.int64))
        info.append(dict(type="torus", label=1, center=c, R=R, r=r, shape_id=shape_id))
        shape_id += 1

    # -------- 长方体 --------
    for _ in range(num_per_class):
        he = np.random.uniform(0.15, 0.50, size=3).astype(np.float32)
        c = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        pts, sdf = _mixed_sample(
            surface_fn=lambda n, c=c, he=he: _sample_box_surface(n, c, he),
            sdf_fn=lambda x, c=c, he=he: sdf_box(x, c, he),
            num_pts=num_pts,
        )
        all_pts.append(pts)
        all_sdf.append(sdf)
        all_ids.append(np.full(num_pts, shape_id, dtype=np.int64))
        info.append(dict(type="box", label=2, center=c, half_extents=he, shape_id=shape_id))
        shape_id += 1

    pts_arr = np.concatenate(all_pts, axis=0).astype(np.float32)
    sdf_arr = np.concatenate(all_sdf, axis=0).astype(np.float32)
    ids_arr = np.concatenate(all_ids, axis=0).astype(np.int64)
    return pts_arr, sdf_arr, ids_arr, info


# --------------------------------------------------------------------------- #
# SDF 切面可视化
# --------------------------------------------------------------------------- #
def visualize_sdf_slices(save_path: Path, grid_size: int = 200) -> None:
    """对三种形状（默认参数）画 z=0 平面的 SDF 热力图，并叠加零等值线。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    g = np.linspace(-1.0, 1.0, grid_size).astype(np.float32)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    pts = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)  # (H, W, 3)

    center = np.zeros(3, dtype=np.float32)
    shapes = [
        ("sphere", lambda p: sdf_sphere(p, center, 0.5)),
        ("torus",  lambda p: sdf_torus(p, center, 0.4, 0.15)),
        ("box",    lambda p: sdf_box(p, center, np.array([0.4, 0.3, 0.45], dtype=np.float32))),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Analytic SDF Slices at z = 0", fontsize=14, fontweight="bold")

    # 统一色标范围
    vmax = 1.0

    for ax, (name, fn) in zip(axes, shapes):
        sdf_grid = fn(pts.reshape(-1, 3)).reshape(grid_size, grid_size)
        cf = ax.contourf(
            gx, gy, sdf_grid,
            levels=40, cmap="RdBu", vmin=-vmax, vmax=vmax,
        )
        ax.contour(gx, gy, sdf_grid, levels=[0.0], colors="black", linewidths=2.0)
        ax.set_aspect("equal")
        ax.set_title(f"{name}", fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="SDF")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SDF 切面图已保存到 {save_path}")


# =========================================================================== #
# 任务二：DeepSDF 自解码器
# =========================================================================== #
class DeepSDF(nn.Module):
    """DeepSDF 自解码器。

    输入: z (B, latent_dim) ⊕ x (B, 3)  →  输出: s (B,)

    架构：
      - num_layers 个 Linear(hidden_dim) + ReLU 隐藏层，使用 weight_norm；
      - 在中间层（num_layers // 2）做 skip connection，将原始输入再次拼接；
      - 输出层 Linear → Tanh，把 SDF 预测压在 (-1, 1)，避免 clamp 饱和。
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        skip_at: int | None = None,
    ):
        super().__init__()
        if skip_at is None:
            skip_at = num_layers // 2

        self.latent_dim = latent_dim
        self.skip_at = skip_at
        self.num_layers = num_layers

        in_dim = latent_dim + 3
        layers = []
        for i in range(num_layers):
            d_out = hidden_dim
            if i == 0:
                d_in = in_dim
            elif i == skip_at:
                d_in = hidden_dim + in_dim                   # skip 连接处把原始输入再拼回来
            else:
                d_in = hidden_dim
            layers.append(nn.utils.weight_norm(nn.Linear(d_in, d_out)))
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, 1))

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim), x: (B, 3)  →  (B,)"""
        inp = torch.cat([z, x], dim=-1)             # (B, latent + 3)
        h = inp
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, inp], dim=-1)     # skip connection
            h = F.relu(layer(h))
        out = torch.tanh(self.out_layer(h))         # (B, 1)，限制在 (-1, 1)
        return out.squeeze(-1)


# =========================================================================== #
# 训练
# =========================================================================== #
def train_deepsdf(
    model: DeepSDF,
    latent_codes: nn.Embedding,
    pts_t: torch.Tensor,
    sdf_t: torch.Tensor,
    ids_t: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr_model: float,
    lr_latent: float,
    clamp_delta: float,
    lambda_reg: float,
) -> List[float]:
    """联合优化 DeepSDF 网络参数和隐编码（独立的两个 Adam）。"""
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr_model)
    optimizer_latent = torch.optim.Adam(latent_codes.parameters(), lr=lr_latent)

    scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_model, mode="min", factor=0.5, patience=30
    )
    scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_latent, mode="min", factor=0.5, patience=30
    )

    num_samples = pts_t.shape[0]
    losses: List[float] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_samples, device=pts_t.device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            pts = pts_t[idx].to(device, non_blocking=True)            # (B, 3)
            gt_sdf = sdf_t[idx].to(device, non_blocking=True)         # (B,)
            shape_ids = ids_t[idx].to(device, non_blocking=True)      # (B,)

            z = latent_codes(shape_ids)                               # (B, latent_dim)
            pred = model(z, pts)                                      # (B,)

            # Clamped L1 Loss —— 只对 target 做 clamp，避免梯度死锁
            target = torch.clamp(gt_sdf, -clamp_delta, clamp_delta)
            loss_recon = F.l1_loss(pred, target)

            # 隐编码 L2 正则化（按 batch 中出现的 z 加权）
            loss_reg = lambda_reg * (z.pow(2).sum(dim=-1)).mean()

            loss = loss_recon + loss_reg
            optimizer_model.zero_grad(set_to_none=True)
            optimizer_latent.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_model.step()
            optimizer_latent.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)
        scheduler_model.step(avg_loss)
        scheduler_latent.step(avg_loss)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.5f} | "
                f"lr_m={optimizer_model.param_groups[0]['lr']:.2e} "
                f"lr_z={optimizer_latent.param_groups[0]['lr']:.2e} | "
                f"{time.time() - t0:.1f}s"
            )

    print(f"训练完成（共 {(time.time() - t0) / 60.0:.2f} 分钟），最终 loss = {losses[-1]:.5f}")
    return losses


def plot_loss_curve(losses: List[float], save_path: Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(1, len(losses) + 1), losses, color="steelblue", linewidth=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Loss (Clamped L1 + λ‖z‖²)")
    ax.set_title("DeepSDF Training Loss")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


# =========================================================================== #
# 任务三：Marching Cubes 表面提取
# =========================================================================== #
@torch.no_grad()
def extract_mesh(
    model: DeepSDF,
    latent_code: torch.Tensor,
    device: torch.device,
    resolution: int = 64,
    chunk: int = 16384,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """从 DeepSDF 中提取零等值面网格。

    Args:
        latent_code: (latent_dim,) —— 单个隐编码张量
        resolution:  3D 网格的每一维分辨率
        chunk:       每次送入网络的点数，避免显存爆炸
    """
    model.eval()
    grid = np.linspace(-1.0, 1.0, resolution).astype(np.float32)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)      # (R^3, 3)

    sdf_pred = np.empty(grid_points.shape[0], dtype=np.float32)
    z_single = latent_code.to(device)
    for i in range(0, grid_points.shape[0], chunk):
        pts = torch.from_numpy(grid_points[i:i + chunk]).to(device)
        z = z_single.unsqueeze(0).expand(pts.shape[0], -1)
        sdf_pred[i:i + chunk] = model(z, pts).detach().cpu().numpy()

    sdf_volume = sdf_pred.reshape(resolution, resolution, resolution)

    # 若零等值面不存在（整个体积同号），返回空网格
    if sdf_volume.min() > 0.0 or sdf_volume.max() < 0.0:
        return None, None

    verts, faces, _, _ = marching_cubes(sdf_volume, level=0.0)
    # 网格坐标 [0, resolution-1] → 世界坐标 [-1, 1]
    verts = verts / (resolution - 1) * 2.0 - 1.0
    return verts.astype(np.float32), faces.astype(np.int64)


def _draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, title: str, color: str = "#6FA8DC") -> None:
    ax.set_title(title, fontsize=12)
    if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
        ax.text2D(0.5, 0.5, "(empty surface)", transform=ax.transAxes,
                  ha="center", va="center", color="red")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        return
    mesh = Poly3DCollection(verts[faces], alpha=0.85)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((0.2, 0.2, 0.2, 0.3))
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-60)


def visualize_reconstructions(
    model: DeepSDF,
    latent_codes: nn.Embedding,
    info: List[Dict],
    device: torch.device,
    save_path: Path,
    resolution: int = 64,
) -> None:
    """对三类形状各取一个代表实例，提取 mesh 并 3D 可视化。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 4.8))
    fig.suptitle("DeepSDF Reconstructions via Marching Cubes",
                 fontsize=14, fontweight="bold")

    colors = {"sphere": "#6FA8DC", "torus": "#F6B26B", "box": "#93C47D"}
    chosen: Dict[int, Dict] = {}
    for entry in info:
        if entry["label"] not in chosen:
            chosen[entry["label"]] = entry
        if len(chosen) >= 3:
            break

    for k, label in enumerate(sorted(chosen)):
        entry = chosen[label]
        z = latent_codes.weight[entry["shape_id"]].detach()
        verts, faces = extract_mesh(model, z, device, resolution=resolution)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        title = f"{entry['type']}  (shape #{entry['shape_id']})"
        _draw_mesh(ax, verts, faces, title, color=colors[entry["type"]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"重建结果已保存到 {save_path}")


# =========================================================================== #
# 任务四（可选）：隐空间插值
# =========================================================================== #
def visualize_interpolation(
    model: DeepSDF,
    latent_codes: nn.Embedding,
    info: List[Dict],
    device: torch.device,
    save_path: Path,
    resolution: int = 48,
    label_a: int = 0,
    label_b: int = 1,
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """在两个不同类别的形状之间线性插值 latent code，提取并展示网格。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 选两个端点形状
    entry_a = next(e for e in info if e["label"] == label_a)
    entry_b = next(e for e in info if e["label"] == label_b)
    z_a = latent_codes.weight[entry_a["shape_id"]].detach()
    z_b = latent_codes.weight[entry_b["shape_id"]].detach()

    fig = plt.figure(figsize=(3.5 * len(alphas), 4.5))
    fig.suptitle(
        f"Latent Space Interpolation: {entry_a['type']} → {entry_b['type']}",
        fontsize=14, fontweight="bold",
    )
    for k, a in enumerate(alphas):
        z = (1.0 - a) * z_a + a * z_b
        verts, faces = extract_mesh(model, z, device, resolution=resolution)
        ax = fig.add_subplot(1, len(alphas), k + 1, projection="3d")
        _draw_mesh(ax, verts, faces, f"α = {a:.2f}", color="#C27BA0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"隐空间插值结果已保存到 {save_path}")


# =========================================================================== #
# 主流程
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="HW8: DeepSDF for analytic primitives")
    parser.add_argument("--num_per_class", type=int, default=15)
    parser.add_argument("--num_pts", type=int, default=10000)

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr_model", type=float, default=1e-4)
    parser.add_argument("--lr_latent", type=float, default=1e-3)
    parser.add_argument("--clamp_delta", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)

    parser.add_argument("--resolution", type=int, default=64,
                        help="Marching Cubes 提取时的 3D 网格分辨率")
    parser.add_argument("--no_interp", action="store_true",
                        help="跳过任务四（隐空间插值）")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assets_dir", type=str,
                        default=str(Path(__file__).parent / "assets"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"设备: {device}  |  latent_dim={args.latent_dim}  "
        f"hidden_dim={args.hidden_dim}  num_layers={args.num_layers}  "
        f"epochs={args.epochs}  batch_size={args.batch_size}"
    )

    # ---------- 任务一：生成数据 + 切面可视化 ----------
    print("=" * 64)
    print("任务一：生成 SDF 训练数据 + 切面可视化")
    print("=" * 64)
    pts, sdf, ids, info = generate_sdf_dataset(
        num_per_class=args.num_per_class,
        num_pts=args.num_pts,
    )
    num_shapes = len(info)
    print(
        f"生成 {num_shapes} 个形状（每类 {args.num_per_class}），"
        f"总采样点数 = {pts.shape[0]:,}"
    )
    print(
        f"  SDF 统计: min={sdf.min():.3f}  max={sdf.max():.3f}  "
        f"|SDF|≤{args.clamp_delta} 的比例 = "
        f"{(np.abs(sdf) <= args.clamp_delta).mean() * 100:.1f}%"
    )

    visualize_sdf_slices(assets_dir / "sdf_slices.png", grid_size=200)

    # ---------- 任务二：训练 DeepSDF ----------
    print("\n" + "=" * 64)
    print("任务二：构建并训练 DeepSDF（联合优化网络 + 隐编码）")
    print("=" * 64)
    model = DeepSDF(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    print(model)
    print(f"模型可训练参数: {count_parameters(model):,}")

    latent_codes = nn.Embedding(num_shapes, args.latent_dim).to(device)
    nn.init.normal_(latent_codes.weight, mean=0.0, std=0.01)
    print(f"隐编码: nn.Embedding({num_shapes}, {args.latent_dim}), "
          f"参数数 = {num_shapes * args.latent_dim}")

    pts_t = torch.from_numpy(pts)                       # 留在 CPU（按 index 取后再 .to(device)）
    sdf_t = torch.from_numpy(sdf)
    ids_t = torch.from_numpy(ids)

    losses = train_deepsdf(
        model, latent_codes,
        pts_t, sdf_t, ids_t,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_model=args.lr_model,
        lr_latent=args.lr_latent,
        clamp_delta=args.clamp_delta,
        lambda_reg=args.lambda_reg,
    )
    plot_loss_curve(losses, assets_dir / "deepsdf_loss.png")

    # ---------- 任务三：Marching Cubes 重建 ----------
    print("\n" + "=" * 64)
    print(f"任务三：Marching Cubes 提取表面（resolution={args.resolution}）")
    print("=" * 64)
    visualize_reconstructions(
        model, latent_codes, info,
        device=device,
        save_path=assets_dir / "deepsdf_reconstructions.png",
        resolution=args.resolution,
    )

    # ---------- 任务四（可选）：隐空间插值 ----------
    if not args.no_interp:
        print("\n" + "=" * 64)
        print("任务四（可选）：隐空间线性插值 sphere → torus")
        print("=" * 64)
        visualize_interpolation(
            model, latent_codes, info,
            device=device,
            save_path=assets_dir / "deepsdf_interpolation.png",
            resolution=max(48, args.resolution // 2 + 16),
            label_a=0, label_b=1,
        )

    # ---------- 保存权重 + 隐编码 ----------
    ckpt_path = assets_dir / "deepsdf.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_codes": latent_codes.state_dict(),
            "info": info,
            "args": vars(args),
            "losses": losses,
        },
        ckpt_path,
    )
    print(f"\n模型权重 + 隐编码已保存到 {ckpt_path}")

    # ---------- 结果汇总 ----------
    print("\n" + "=" * 64)
    print("结果汇总")
    print("=" * 64)
    print(f"形状数        : {num_shapes}（每类 {args.num_per_class}）")
    print(f"训练点总数    : {pts.shape[0]:,}")
    print(f"模型参数      : {count_parameters(model):,}")
    print(f"隐编码参数    : {num_shapes * args.latent_dim}")
    print(f"final loss    : {losses[-1]:.5f}")
    print(f"min loss      : {min(losses):.5f}")


if __name__ == "__main__":
    main()
