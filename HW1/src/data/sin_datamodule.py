import math
from typing import Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

class SinDataModule(LightningDataModule):
    """
    负责生成正弦函数数据 f(x) = sin(x) 的 DataModule。
    支持自定义样本数和区间。
    """
    def __init__(
        self,
        batch_size: int = 64,
        num_samples: int = 10000,
        x_min: float = -2 * math.pi,
        x_max: float = 2 * math.pi,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        
        # 这一步会将所有的 init 参数存储在 self.hparams 中
        self.save_hyperparameters(logger=False)

        self.dataset: Optional[TensorDataset] = None
        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None):
        """生成数据并随机划分为训练集、验证集和测试集"""
        if not self.data_train and not self.data_val and not self.data_test:
            # 1. 在 [x_min, x_max] 范围内生成连续的特征 x 并增加一个维度 (N, 1)
            x = torch.linspace(self.hparams.x_min, self.hparams.x_max, self.hparams.num_samples).unsqueeze(1)
            
            # 2. 生成目标变量 y = sin(x)
            y = torch.sin(x)
            
            # 构建一个基础的 TensorDataset
            self.dataset = TensorDataset(x, y)
            
            # 计算切分长度
            train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
            train_len = int(train_ratio * self.hparams.num_samples)
            val_len = int(val_ratio * self.hparams.num_samples)
            test_len = self.hparams.num_samples - train_len - val_len
            
            # 随机划分为 train/validation/test (设置 generator 增加复现性)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
