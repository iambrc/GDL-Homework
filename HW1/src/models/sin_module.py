from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule

class SinLitModule(LightningModule):
    """
    用于训练 SinNet 的 PyTorch Lightning 模块
    包含训练、验证和测试逻辑。
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        super().__init__()
        
        # 保存传入的超参数，不包含网络参数 itself
        self.save_hyperparameters(logger=False, ignore=["net"])
        
        # 初始化模型
        self.net = net

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        preds = self.forward(x)
        # 用均方误差(MSE)计算损失
        loss = F.mse_loss(preds, y)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, y = self.step(batch)
        # 记录训练损失
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, y = self.step(batch)
        # 记录验证损失
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, y = self.step(batch)
        # 记录测试损失
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
