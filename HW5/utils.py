"""
HW5 - 公共工具
- 训练 / 评估循环
- 训练曲线绘制
- t-SNE 节点嵌入可视化
- 可选的 wandb 记录开关
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# 随机种子
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# wandb 可选开关
# --------------------------------------------------------------------------- #
class WandbRun:
    """wandb 的薄封装，--use_wandb=false 时变为 no-op。"""

    def __init__(
        self,
        enabled: bool,
        project: str = "GDL-Homework",
        name: Optional[str] = None,
        group: str = "HW5",
        config: Optional[dict] = None,
        tags: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(
                project=project,
                name=name,
                group=group,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] 初始化失败（{exc}），降级为 no-op。")
            self.enabled = False

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self._wandb.finish()


# --------------------------------------------------------------------------- #
# 通用训练循环（任务一 / 任务二 / 任务三 共用）
# --------------------------------------------------------------------------- #
def train_node_classifier(
    model: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    forward_fn: Callable[[torch.nn.Module], torch.Tensor],
    log_every: int = 10,
    wandb_run: Optional[WandbRun] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    通用半监督节点分类训练循环。
    forward_fn(model) -> logits (num_nodes, num_classes)，
    让调用方决定传 edge_index 还是 A_hat。
    """
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    best = {"val_acc": 0.0, "test_acc": 0.0, "epoch": 0}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ------- train -------
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # ------- eval -------
        model.eval()
        with torch.no_grad():
            logits_eval = forward_fn(model)
            pred = logits_eval.argmax(dim=1)
            accs = {}
            for split, mask in [
                ("train", data.train_mask),
                ("val", data.val_mask),
                ("test", data.test_mask),
            ]:
                accs[split] = (pred[mask] == data.y[mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(accs["train"])
        history["val_acc"].append(accs["val"])
        history["test_acc"].append(accs["test"])

        # early tracking：以验证集为准选择最好的 test
        if accs["val"] > best["val_acc"]:
            best.update(
                val_acc=accs["val"], test_acc=accs["test"], epoch=epoch
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": loss.item(),
                    "train/acc": accs["train"],
                    "val/acc": accs["val"],
                    "test/acc": accs["test"],
                },
                step=epoch,
            )

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"train={accs['train']*100:.2f}% "
                f"val={accs['val']*100:.2f}% "
                f"test={accs['test']*100:.2f}%"
            )

    elapsed = time.time() - t0
    print(
        f"\n训练完成（{elapsed:.1f}s）。最佳 val epoch={best['epoch']}，"
        f"对应 val={best['val_acc']*100:.2f}%，test={best['test_acc']*100:.2f}%"
    )
    print(f"最终（第 {epochs} 轮）test={history['test_acc'][-1]*100:.2f}%")

    summary = {
        "best_val_acc": best["val_acc"],
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
        "final_test_acc": history["test_acc"][-1],
        "elapsed_sec": elapsed,
    }
    if wandb_run is not None:
        wandb_run.log({f"summary/{k}": v for k, v in summary.items()})
    return history, summary


# --------------------------------------------------------------------------- #
# 训练曲线绘制
# --------------------------------------------------------------------------- #
def plot_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "GCN on Cora",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss (on 140 labeled nodes)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    ax.plot(epochs, history["val_acc"], label="Val Acc", color="orange")
    ax.plot(epochs, history["test_acc"], label="Test Acc", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Train / Val / Test Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


# --------------------------------------------------------------------------- #
# t-SNE 节点嵌入可视化
# --------------------------------------------------------------------------- #
CORA_CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE of GCN hidden features",
    class_names: Optional[List[str]] = None,
    seed: int = 42,
) -> None:
    """对节点隐藏表示做 t-SNE 降维，按类别着色。"""
    from sklearn.manifold import TSNE

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在对 {embeddings.shape[0]} 个节点做 t-SNE 降维（{embeddings.shape[1]} → 2）...")
    # 兼容 sklearn 不同版本：新版本 TSNE 的参数名为 `max_iter`，旧版本为 `n_iter`
    tsne_kwargs = dict(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    try:
        tsne = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=1000, **tsne_kwargs)
    emb_2d = tsne.fit_transform(embeddings)

    num_classes = int(labels.max()) + 1
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(num_classes):
        mask = labels == c
        name = class_names[c] if class_names is not None else f"Class {c}"
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=12,
            alpha=0.75,
            color=cmap(c),
            label=name,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE 图已保存至 {save_path}")


# --------------------------------------------------------------------------- #
# 参数量统计
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
