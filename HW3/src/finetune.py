"""
HW3 Task 2 - 基于预训练 ResNet18 的迁移学习（猫狗分类）

两阶段训练策略：
  阶段一（热身）：冻结全部骨干网络，只训练新的分类头  lr=1e-3  5 epochs
  阶段二（微调）：解冻全部参数，端到端微调            fc 1e-3 / 骨干 1e-4  10 epochs

运行方式：
    python HW3/src/finetune.py                          # 使用默认参数
    python HW3/src/finetune.py --data_dir path/to/data   # 指定数据目录
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.catdog_datamodule import CatDogDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="HW3 Task2: ResNet18 Transfer Learning")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data"),
                        help="数据目录路径")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "finetune"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(num_classes: int = 2) -> nn.Module:
    """加载预训练 ResNet18 并替换分类头。"""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    nn.init.kaiming_normal_(resnet.fc.weight)
    nn.init.zeros_(resnet.fc.bias)
    return resnet


def freeze_backbone(model: nn.Module) -> None:
    """冻结除 fc 以外的所有参数。"""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """解冻全部参数。"""
    for param in model.parameters():
        param.requires_grad = True


def get_backbone_params(model: nn.Module):
    """返回除 fc 以外的所有参数（骨干网络参数）。"""
    fc_ids = set(id(p) for p in model.fc.parameters())
    return [p for p in model.parameters() if id(p) not in fc_ids and p.requires_grad]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def plot_finetune_curves(history: dict, phase1_epochs: int, save_path: str) -> None:
    """绘制训练曲线，竖线标注两阶段分界点。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Dogs vs. Cats — ResNet18 Transfer Learning", fontsize=14, fontweight="bold")

    boundary = phase1_epochs + 0.5

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="orange", linestyle="--")
    ax.axvline(x=boundary, color="red", linestyle=":", linewidth=1.5, label="Phase boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="steelblue")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="green", linestyle="--")
    ax.axvline(x=boundary, color="red", linestyle=":", linewidth=1.5, label="Phase boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim([0, 100])

    # --- Learning Rate ---
    ax = axes[2]
    ax.plot(epochs, history["lr"], label="Learning Rate", color="purple")
    ax.axvline(x=boundary, color="red", linestyle=":", linewidth=1.5, label="Phase boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存至：{save_path}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================== 数据准备 ========================
    dm = CatDogDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # ======================== 构建模型 ========================
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [],
    }
    total_epochs = args.phase1_epochs + args.phase2_epochs

    # ================== 阶段一：冻结骨干，训练分类头 ==================
    print("=" * 60)
    print(f"阶段一（热身）：冻结骨干网络，只训练分类头  [epoch 1-{args.phase1_epochs}]")
    print(f"  学习率：{args.lr_head}")
    print("=" * 60)

    freeze_backbone(model)
    optimizer_p1 = optim.Adam(model.fc.parameters(), lr=args.lr_head)

    for epoch in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_p1, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer_p1.param_groups[0]["lr"])

        print(f"[Phase1] Epoch {epoch:2d}/{total_epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.1f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.1f}% | "
              f"{elapsed:.1f}s")

    # ================== 阶段二：解冻全部，端到端微调 ==================
    print("=" * 60)
    print(f"阶段二（微调）：解冻全部参数，端到端微调  "
          f"[epoch {args.phase1_epochs + 1}-{total_epochs}]")
    print(f"  分类头学习率：{args.lr_head}  骨干学习率：{args.lr_backbone}")
    print("=" * 60)

    unfreeze_all(model)
    optimizer_p2 = optim.Adam([
        {"params": model.fc.parameters(), "lr": args.lr_head},
        {"params": get_backbone_params(model), "lr": args.lr_backbone},
    ])

    best_val_acc = 0.0
    for epoch in range(args.phase1_epochs + 1, total_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_p2, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer_p2.param_groups[0]["lr"])

        print(f"[Phase2] Epoch {epoch:2d}/{total_epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.1f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.1f}% | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = output_dir / "resnet18_finetune_best.pth"
            torch.save(model.state_dict(), save_path)

    # ======================== 保存 & 绘图 ========================
    final_path = output_dir / "resnet18_finetune_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n最终模型已保存至：{final_path}")
    print(f"最佳验证准确率：{best_val_acc * 100:.2f}%")

    curve_path = output_dir / "finetune_curves.png"
    plot_finetune_curves(history, args.phase1_epochs, str(curve_path))


if __name__ == "__main__":
    main()
