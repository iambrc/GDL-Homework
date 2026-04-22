"""
HW6 任务三（可选）：PointNet 关键点（Critical Points）可视化。

思路：
  PointNet 的全局特征来自 Max Pool —— 对 1024 维特征的每一维，
  只有"贡献了最大值"的那个点被选中。这些点去重后就是**关键点集**。

用法：
    python critical_points.py                               # 默认用 assets/pointnet.pt + 测试集第 0 个样本
    python critical_points.py --idx 42 --ckpt assets/pointnet.pt
    python critical_points.py --num_samples 4               # 同时画多个样本

产出：
    assets/critical_points.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints

from pointnet_cls import MODELNET10_CLASSES, PointNetCls, set_seed


def visualize_critical_points(
    ax,
    points: torch.Tensor,          # (N, 3)
    critical_idx: torch.Tensor,    # (K,)
    title: str,
) -> None:
    pts = points.cpu().numpy()
    crit = critical_idx.cpu().numpy()

    mask = torch.zeros(points.size(0), dtype=torch.bool)
    mask[critical_idx] = True
    non_crit = pts[~mask.numpy()]
    crit_pts = pts[crit]

    ax.scatter(non_crit[:, 0], non_crit[:, 1], non_crit[:, 2],
               c="lightgray", s=4, alpha=0.35, label=f"points ({len(non_crit)})")
    ax.scatter(crit_pts[:, 0], crit_pts[:, 1], crit_pts[:, 2],
               c="crimson", s=14, alpha=0.95, label=f"critical ({len(crit_pts)})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)
    # 统一视角
    ax.view_init(elev=20, azim=45)


def main() -> None:
    parser = argparse.ArgumentParser(description="HW6 Bonus: Critical Points")
    parser.add_argument("--ckpt", type=str, default="assets/pointnet.pt")
    parser.add_argument("--data_root", type=str, default="./data/ModelNet10")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--idx", type=int, default=0, help="测试集中样本的索引")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="画几个测试样本（从 idx 开始）")
    parser.add_argument("--save_path", type=str, default="assets/critical_points.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试集（不带增强）
    dataset = ModelNet(
        root=args.data_root, name="10", train=False,
        pre_transform=Compose([SamplePoints(num=args.num_points), NormalizeScale()]),
    )

    # 加载训练好的 PointNet
    model = PointNetCls(num_classes=10).to(device)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"没找到 PointNet 权重: {ckpt}. 先跑 `python pointnet_cls.py --models pointnet`。"
        )
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 画多个样本
    n = max(1, args.num_samples)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

    for i in range(n):
        sample = dataset[args.idx + i]
        points = sample.pos.to(device).unsqueeze(0)     # (1, N, 3)
        y_true = int(sample.y)

        with torch.no_grad():
            logits, _, pointwise = model(
                points, return_feat_trans=True, return_pointwise=True,
            )
            # pointwise: (1, 1024, N) —— Max Pool 前的逐点特征
            pred = int(logits.argmax(dim=1))
            # 每个特征维度对应的贡献最大值的点的索引
            argmax_idx = pointwise.squeeze(0).argmax(dim=1)   # (1024,)
            critical = argmax_idx.unique()                    # 去重

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        title = (
            f"idx={args.idx + i} | true={MODELNET10_CLASSES[y_true]} | "
            f"pred={MODELNET10_CLASSES[pred]} | {len(critical)} critical"
        )
        visualize_critical_points(ax, points.squeeze(0), critical, title)

    fig.suptitle("PointNet Critical Points on ModelNet10", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"关键点可视化已保存到 {save_path}")


if __name__ == "__main__":
    main()
