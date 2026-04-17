"""
HW5 任务一：使用 PyTorch Geometric 的 GCNConv 实现 Cora 半监督节点分类。

用法：
    python gcn_cora.py                    # 默认 200 epoch
    python gcn_cora.py --use_wandb        # 同时用 wandb 记录曲线
    python gcn_cora.py --epochs 300 --hidden_dim 64

产出：
    assets/gcn_curves.png   训练 loss / acc 曲线
    assets/gcn_tsne.png     GCN 第一层隐藏表示的 t-SNE 图
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

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
# 模型
# --------------------------------------------------------------------------- #
class GCN(torch.nn.Module):
    """
    标准两层 GCN：
        x → GCNConv → ReLU → Dropout → GCNConv → logits
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

    @torch.no_grad()
    def extract_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """专用接口：不经过 dropout，拿到第一层的干净表示。"""
        self.eval()
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        return h


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="HW5 Task 1: GCN on Cora (PyG)")
    parser.add_argument("--data_root", type=str, default="./data", help="Cora 数据集缓存路径")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=str(Path(__file__).parent / "assets"),
        help="图片输出目录",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否启用 wandb 记录")
    parser.add_argument("--wandb_project", type=str, default="GDL-Homework")
    parser.add_argument("--wandb_name", type=str, default="hw5_gcn_cora")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # ---------------- 加载数据集 ----------------
    dataset = Planetoid(root=args.data_root, name="Cora")
    data = dataset[0].to(device)

    print("=" * 60)
    print("Cora 数据集信息")
    print("=" * 60)
    print(f"节点数 num_nodes         : {data.num_nodes}")
    print(f"边数 num_edges           : {data.num_edges}")
    print(f"特征维度 num_features    : {data.num_node_features}")
    print(f"类别数 num_classes       : {dataset.num_classes}")
    print(f"训练节点 train_mask      : {int(data.train_mask.sum())}")
    print(f"验证节点 val_mask        : {int(data.val_mask.sum())}")
    print(f"测试节点 test_mask       : {int(data.test_mask.sum())}")
    print("=" * 60)

    # ---------------- 构建模型 ----------------
    model = GCN(
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
        tags=["hw5", "cora", "gcn", "pyg"],
        config={
            "task": "gcn_cora",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_params": count_parameters(model),
        },
    )

    # ---------------- 训练 ----------------
    forward_fn = lambda m: m(data.x, data.edge_index)
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
        save_path=assets_dir / "gcn_curves.png",
        title="GCN (PyG) on Cora — Training Dynamics",
    )

    # t-SNE：用 eval 模式下第一层输出
    hidden = model.extract_hidden(data.x, data.edge_index).cpu().numpy()
    labels = data.y.cpu().numpy()
    tsne_plot(
        embeddings=hidden,
        labels=labels,
        save_path=assets_dir / "gcn_tsne.png",
        title="t-SNE of GCN hidden features (64-d)",
        class_names=CORA_CLASS_NAMES,
        seed=args.seed,
    )

    wandb_run.finish()

    print("\n" + "=" * 60)
    print("任务一（PyG GCN）完成")
    print("=" * 60)
    print(f"best_val_acc       : {summary['best_val_acc']*100:.2f}%")
    print(f"best_test_acc      : {summary['best_test_acc']*100:.2f}%")
    print(f"final_test_acc     : {summary['final_test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
