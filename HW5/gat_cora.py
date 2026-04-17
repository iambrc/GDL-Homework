"""
HW5 任务三（可选）：GAT 与 GCN 在 Cora 上的对比实验。

GAT（Graph Attention Network）将邻居聚合权重从 GCN 的静态归一化常数
替换为基于 Query-Key 的动态注意力分数：

    alpha_{ij} = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
    h_i'       = sigma( sum_{j in N(i) ∪ {i}} alpha_{ij} W h_j )

用法：
    python gat_cora.py
    python gat_cora.py --use_wandb
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from utils import (
    CORA_CLASS_NAMES,
    WandbRun,
    count_parameters,
    plot_curves,
    set_seed,
    train_node_classifier,
    tsne_plot,
)


class GAT(torch.nn.Module):
    """
    参考 Veličković et al. 2018：
        layer 1: GATConv(in, hidden, heads=8, concat=True) → ELU → Dropout
        layer 2: GATConv(hidden*heads, num_classes, heads=1, concat=False)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    @torch.no_grad()
    def extract_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.elu(self.conv1(x, edge_index))


def main() -> None:
    parser = argparse.ArgumentParser(description="HW5 Task 3: GAT on Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=str(Path(__file__).parent / "assets"),
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="GDL-Homework")
    parser.add_argument("--wandb_name", type=str, default="hw5_gat_cora")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    dataset = Planetoid(root=args.data_root, name="Cora")
    data = dataset[0].to(device)

    model = GAT(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_classes,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    print(model)
    print(f"可训练参数量：{count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    wandb_run = WandbRun(
        enabled=args.use_wandb,
        project=args.wandb_project,
        name=args.wandb_name,
        group="HW5",
        tags=["hw5", "cora", "gat"],
        config={
            "task": "gat_cora",
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_params": count_parameters(model),
        },
    )

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

    assets_dir = Path(args.assets_dir)
    plot_curves(
        history,
        save_path=assets_dir / "gat_curves.png",
        title="GAT on Cora — Training Dynamics",
    )

    hidden = model.extract_hidden(data.x, data.edge_index).cpu().numpy()
    labels = data.y.cpu().numpy()
    tsne_plot(
        embeddings=hidden,
        labels=labels,
        save_path=assets_dir / "gat_tsne.png",
        title="t-SNE of GAT hidden features",
        class_names=CORA_CLASS_NAMES,
        seed=args.seed,
    )

    wandb_run.finish()

    print("\n" + "=" * 60)
    print("任务三（GAT）完成")
    print("=" * 60)
    print(f"best_val_acc   : {summary['best_val_acc']*100:.2f}%")
    print(f"best_test_acc  : {summary['best_test_acc']*100:.2f}%")
    print(f"final_test_acc : {summary['final_test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
