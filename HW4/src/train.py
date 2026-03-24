"""
HW4 - 微调 MarianMT 英中翻译模型
使用 PyTorch Lightning + Hydra + wandb

运行方式：
    python HW4/src/train.py experiment=enzh_translation
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


def plot_training_curves(model: LightningModule, output_dir: str) -> None:
    """绘制训练曲线（loss 和 BLEU）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = getattr(model, "history", {})
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_bleu = history.get("val_bleu", [])

    if not train_loss and not val_loss:
        log.warning("无训练历史数据，跳过绘图。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("English → Chinese Translation — MarianMT Fine-tuning", fontsize=14, fontweight="bold")

    ax = axes[0]
    epochs_t = list(range(1, len(train_loss) + 1))
    epochs_v = list(range(1, len(val_loss) + 1))
    if train_loss:
        ax.plot(epochs_t, train_loss, label="Train Loss", color="steelblue")
    if val_loss:
        ax.plot(epochs_v, val_loss, label="Val Loss", color="orange", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    if val_bleu:
        epochs_b = list(range(1, len(val_bleu) + 1))
        ax.plot(epochs_b, val_bleu, label="Val BLEU", color="green", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU Score")
    ax.set_title("Validation BLEU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = Path(output_dir) / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"训练曲线已保存至：{save_path}")


def print_translation_samples(model: LightningModule, datamodule: LightningDataModule, n: int = 5) -> None:
    """打印翻译样例对比。"""
    model.eval()
    device = next(model.parameters()).device
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))

    input_ids = batch["input_ids"][:n].to(device)
    attention_mask = batch["attention_mask"][:n].to(device)
    labels = batch["labels"][:n].clone()
    labels[labels == -100] = model.tokenizer.pad_token_id

    with torch.no_grad():
        generated = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=model.hparams.max_length,
            num_beams=model.hparams.num_beams,
        )

    sources = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    references = model.tokenizer.batch_decode(labels.to(device), skip_special_tokens=True)
    predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("翻译样例对比")
    print("=" * 80)
    for i in range(min(n, len(sources))):
        print(f"\n[样例 {i+1}]")
        print(f"  源文(EN) : {sources[i]}")
        print(f"  参考(ZH) : {references[i]}")
        print(f"  模型输出 : {predictions[i]}")
    print("=" * 80 + "\n")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """训练英中翻译模型。"""

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

    # 微调前评估基线 BLEU
    if cfg.get("eval_baseline", True):
        log.info("评估微调前基线 BLEU...")
        trainer.validate(model=model, datamodule=datamodule)
        baseline_bleu = trainer.callback_metrics.get("val/bleu")
        if baseline_bleu is not None:
            log.info(f"微调前基线 BLEU: {baseline_bleu:.2f}")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # 微调后评估
    if cfg.get("test", True):
        log.info("微调后评估...")
        ckpt_path = None
        if cfg.get("train"):
            best = trainer.checkpoint_callback.best_model_path
            if best:
                ckpt_path = best
                log.info(f"加载最佳检查点: {ckpt_path}")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        final_bleu = trainer.callback_metrics.get("val/bleu")
        if final_bleu is not None:
            log.info(f"微调后 BLEU: {final_bleu:.2f}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    output_dir = Path(cfg.paths.output_dir)

    # 保存最佳模型（validate 已加载最佳 checkpoint）
    save_dir = output_dir / "best_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(save_dir))
    model.tokenizer.save_pretrained(str(save_dir))
    log.info(f"最佳模型已保存至：{save_dir}")

    # 打印翻译样例
    print_translation_samples(model, datamodule)

    # 绘制训练曲线
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
