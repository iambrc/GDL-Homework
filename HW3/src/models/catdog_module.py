"""
HW3 - 猫狗分类 LightningModule
支持 Task 1（从零 CNN）和 Task 2（迁移学习），通过 Hydra 注入不同的 net。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy


class CatDogLitModule(LightningModule):
    """
    猫狗二分类的 PyTorch Lightning 模块。
    使用交叉熵损失和 torchmetrics 追踪准确率。
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)
        self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        loss = self.trainer.callback_metrics.get("train/loss")
        if loss is not None:
            self.history["train_loss"].append(loss.item())

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None:
            self.history["val_loss"].append(val_loss.item())
        if val_acc is not None:
            self.history["val_acc"].append(val_acc.item())

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)
        self.test_acc(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
