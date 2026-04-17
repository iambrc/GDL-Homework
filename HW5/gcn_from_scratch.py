"""
HW5 任务二：从头实现 GCN（不使用 GCNConv）。

核心公式：
    A_hat  = D_tilde^{-1/2}  @  (A + I)  @  D_tilde^{-1/2}
    H^{l+1} = sigma( A_hat @ H^{l} @ W^{l} )

只使用 torch 的矩阵运算和 nn.Linear，在 Cora 上应能达到与
PyG 的 GCNConv 同量级的测试准确率（~81%）。

用法：
    python gcn_from_scratch.py
    python gcn_from_scratch.py --use_wandb
    python gcn_from_scratch.py --dense_adj --epochs 300
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from utils import (
    CORA_CLASS_NAMES,
    WandbRun,
    count_parameters,
    plot_curves,
    set_seed,
    train_node_classifier,
    tsne_plot,
)


# --------------------------------------------------------------------------- #
# 构建对称归一化邻接矩阵  A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
# --------------------------------------------------------------------------- #
def build_norm_adj_dense(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """稠密版：2708^2 ≈ 28 MB，Cora 规模下完全 OK，最直观。"""
    A = torch.zeros(num_nodes, num_nodes, device=device)
    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = 1.0  # Cora 的 edge_index 已经是对称的（双向边），保险起见再显式对称化
    A = torch.maximum(A, A.t())

    # 加自环
    A_tilde = A + torch.eye(num_nodes, device=device)

    # 计算度  D_tilde_{ii} = sum_j A_tilde_{ij}
    deg = A_tilde.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # D^{-1/2} @ A_tilde @ D^{-1/2}
    # 等价于：  A_hat[i, j] = A_tilde[i, j] / sqrt(deg[i] * deg[j])
    A_hat = A_tilde * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1)
    return A_hat


def build_norm_adj_sparse(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    稀疏 COO 版：对于更大的图（如 OGB）更省内存。
    返回 torch.sparse_coo_tensor，可以直接 `A_hat @ X` 做聚合。
    """
    # 加自环
    self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index_aug = torch.cat([edge_index, self_loops], dim=1)

    # 度（含自环）
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(
        0, edge_index_aug[0], torch.ones(edge_index_aug.size(1), device=device)
    )
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 每条边的归一化权重
    row, col = edge_index_aug[0], edge_index_aug[1]
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A_hat = torch.sparse_coo_tensor(
        indices=edge_index_aug,
        values=values,
        size=(num_nodes, num_nodes),
    ).coalesce()
    return A_hat


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    """  H_out = A_hat @ H_in @ W   （可选 bias；原论文无 bias）"""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        # Kaiming / Glorot 初始化（PyG 的默认也是 Glorot）
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)      # H @ W
        # `A_hat @ x` 对稀疏 / 稠密矩阵都适用
        return torch.sparse.mm(A_hat, x) if A_hat.is_sparse else A_hat @ x


class GCNFromScratch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gc1 = GCNLayer(in_channels, hidden_channels)
        self.gc2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, A_hat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gc2(h, A_hat)
        return out

    @torch.no_grad()
    def extract_hidden(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.relu(self.gc1(x, A_hat))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="HW5 Task 2: GCN from scratch on Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dense_adj",
        action="store_true",
        default=True,
        help="使用稠密 A_hat（默认 True；Cora 规模下最直观）",
    )
    parser.add_argument(
        "--sparse_adj",
        dest="dense_adj",
        action="store_false",
        help="改用稀疏 COO 版 A_hat",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=str(Path(__file__).parent / "assets"),
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="GDL-Homework")
    parser.add_argument("--wandb_name", type=str, default="hw5_gcn_from_scratch")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # ---------------- 加载数据集 ----------------
    dataset = Planetoid(root=args.data_root, name="Cora")
    data = dataset[0].to(device)

    print("=" * 60)
    print("Cora 数据集信息（任务二：从头实现）")
    print("=" * 60)
    print(f"num_nodes={data.num_nodes}, num_edges={data.num_edges}, "
          f"num_features={data.num_node_features}, num_classes={dataset.num_classes}")
    print(f"train/val/test = {int(data.train_mask.sum())}/"
          f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

    # ---------------- 构建 A_hat ----------------
    if args.dense_adj:
        A_hat = build_norm_adj_dense(data.edge_index, data.num_nodes, device)
        print(f"A_hat（稠密） shape={tuple(A_hat.shape)}, "
              f"非零元素≈{int((A_hat != 0).sum())}")
    else:
        A_hat = build_norm_adj_sparse(data.edge_index, data.num_nodes, device)
        print(f"A_hat（稀疏 COO） shape={tuple(A_hat.shape)}, nnz={A_hat._nnz()}")

    # 数值验证：A_hat 行和 ≤ 1（对无孤立节点的连通成分，严格 <1 除自环聚合外）
    row_sum = (torch.sparse.sum(A_hat, dim=1).to_dense()
               if A_hat.is_sparse else A_hat.sum(dim=1))
    print(f"A_hat 行和统计: min={row_sum.min().item():.4f}, "
          f"max={row_sum.max().item():.4f}, mean={row_sum.mean().item():.4f}")

    # ---------------- 构建模型 ----------------
    model = GCNFromScratch(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)
    print(model)
    print(f"可训练参数量：{count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---------------- wandb ----------------
    wandb_run = WandbRun(
        enabled=args.use_wandb,
        project=args.wandb_project,
        name=args.wandb_name,
        group="HW5",
        tags=["hw5", "cora", "gcn", "from_scratch"],
        config={
            "task": "gcn_from_scratch",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_params": count_parameters(model),
            "dense_adj": args.dense_adj,
        },
    )

    # ---------------- 训练 ----------------
    forward_fn = lambda m: m(data.x, A_hat)
    history, summary = train_node_classifier(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=args.epochs,
        forward_fn=forward_fn,
        log_every=args.log_every,
        wandb_run=wandb_run,
    )

    # ---------------- 可视化 ----------------
    assets_dir = Path(args.assets_dir)
    plot_curves(
        history,
        save_path=assets_dir / "gcn_scratch_curves.png",
        title="GCN (from scratch) on Cora — Training Dynamics",
    )

    hidden = model.extract_hidden(data.x, A_hat).cpu().numpy()
    labels = data.y.cpu().numpy()
    tsne_plot(
        embeddings=hidden,
        labels=labels,
        save_path=assets_dir / "gcn_scratch_tsne.png",
        title="t-SNE of GCN-from-scratch hidden features (64-d)",
        class_names=CORA_CLASS_NAMES,
        seed=args.seed,
    )

    wandb_run.finish()

    print("\n" + "=" * 60)
    print("任务二（从头实现）完成")
    print("=" * 60)
    print(f"best_val_acc   : {summary['best_val_acc']*100:.2f}%")
    print(f"best_test_acc  : {summary['best_test_acc']*100:.2f}%")
    print(f"final_test_acc : {summary['final_test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
