from typing import Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MnistDataModule(LightningDataModule):
    """
    MNIST 数据集的 DataModule。
    自动下载 MNIST，并将训练集拆分为训练/验证集，测试集使用官方测试集。
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        train_val_split: float = 0.9,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # 标准 MNIST 归一化参数（均值=0.1307, 标准差=0.3081）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self) -> None:
        """下载 MNIST（只需触发一次，不在此处赋值）"""
        datasets.MNIST(root=self.hparams.data_dir, train=True, download=True)
        datasets.MNIST(root=self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """加载数据集并拆分训练/验证集"""
        if not self.data_train and not self.data_val and not self.data_test:
            train_full = datasets.MNIST(
                root=self.hparams.data_dir,
                train=True,
                transform=self.transform,
                download=False,
            )
            self.data_test = datasets.MNIST(
                root=self.hparams.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

            train_size = int(len(train_full) * self.hparams.train_val_split)
            val_size = len(train_full) - train_size

            self.data_train, self.data_val = random_split(
                dataset=train_full,
                lengths=[train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
