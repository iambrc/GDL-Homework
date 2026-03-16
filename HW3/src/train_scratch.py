"""
HW3 Task 1 - 从零搭建 CNN 训练猫狗分类
使用 PyTorch Lightning + Hydra 框架

运行方式：
    python HW3/src/train_scratch.py experiment=task1_scratch
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

import collections
import functools
import typing

import omegaconf

torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.Adam,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode,
    omegaconf.nodes.StringNode,
    omegaconf.nodes.BooleanNode,
    omegaconf.nodes.IntegerNode,
    omegaconf.nodes.FloatNode,
    omegaconf.base.Container,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    list,
    dict,
    int,
    collections.defaultdict,
    omegaconf.base.Metadata,
])


def plot_training_curves(model: LightningModule, output_dir: str) -> None:
    """从 LightningModule 的 history 字典读取训练指标，绘制并保存曲线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = getattr(model, "history", {})
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])

    if not train_loss and not val_loss:
        log.warning("无训练历史数据，跳过绘图。")
        return

    epochs_train = list(range(1, len(train_loss) + 1))
    epochs_val = list(range(1, len(val_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dogs vs. Cats — CNN from Scratch", fontsize=14, fontweight="bold")

    ax = axes[0]
    if train_loss:
        ax.plot(epochs_train, train_loss, label="Train Loss", color="steelblue")
    if val_loss:
        ax.plot(epochs_val, val_loss, label="Val Loss", color="orange", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    if val_acc:
        ax.plot(epochs_val, [a * 100 for a in val_acc], label="Val Accuracy", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    save_path = Path(output_dir) / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"训练曲线已保存至：{save_path}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """训练 CNN 猫狗分类模型，测试，保存权重，绘图。"""

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("train"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        else:
            ckpt_path = cfg.get("ckpt_path")
            if not ckpt_path:
                log.warning("train=False 且未指定 ckpt_path，将使用随机初始化权重！")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Testing ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    output_dir = Path(cfg.paths.output_dir)

    best_ckpt = (
        trainer.checkpoint_callback.best_model_path
        if cfg.get("train")
        else cfg.get("ckpt_path")
    )
    if best_ckpt:
        ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        net_state_dict = {
            k[len("net."):]: v
            for k, v in ckpt_data["state_dict"].items()
            if k.startswith("net.")
        }
        save_path = output_dir / "catdog_cnn.pth"
        torch.save(net_state_dict, save_path)
        log.info(f"模型权重已保存至：{save_path}")
    else:
        log.warning("未找到最佳检查点，跳过权重保存。")

    plot_training_curves(model, str(output_dir))

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Hydra 主入口"""
    extras(cfg)

    metric_dict, _ = train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )
    return metric_value


if __name__ == "__main__":
    main()
