"""
HW3 - Dogs vs. Cats 数据模块
自动从目录读取猫狗图像，根据文件名前缀解析标签，按 80/20 划分训练/验证集。
支持自动从 Kaggle 下载数据。
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class CatDogDataset(Dataset):
    """
    自定义猫狗数据集。
    从目录读取图像，根据文件名前缀（cat / dog）自动解析标签（0 / 1）。
    """

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class CatDogDataModule(LightningDataModule):
    """
    Dogs vs. Cats 数据模块。
    数据目录应包含命名格式为 cat.xxxx.jpg / dog.xxxx.jpg 的图像文件。
    按 80/20 比例自动划分训练集和验证集，测试集与验证集保持一致。
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: str = "data/",
        image_size: int = 128,
        batch_size: int = 64,
        train_ratio: float = 0.8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _find_image_dir(self) -> Path:
        """定位包含 cat.*.jpg / dog.*.jpg 的图像目录。"""
        base = Path(self.hparams.data_dir)
        candidates = [
            base / "dogs-vs-cats" / "train",
            base / "train",
            base,
        ]
        for d in candidates:
            if d.is_dir() and any(d.glob("cat.*")):
                return d
        raise FileNotFoundError(
            f"未找到猫狗图像。请将 Kaggle Dogs vs. Cats 数据集解压至 {base}/dogs-vs-cats/train/\n"
            f"下载方式：kaggle competitions download -c dogs-vs-cats -p {base}\n"
            f"然后解压：unzip {base}/dogs-vs-cats.zip -d {base}/dogs-vs-cats && "
            f"unzip {base}/dogs-vs-cats/train.zip -d {base}/dogs-vs-cats/"
        )

    def _scan_images(self, image_dir: Path) -> Tuple[List[str], List[int]]:
        """扫描目录，返回图像路径列表和对应标签列表。"""
        paths, labels = [], []
        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if fname.startswith("cat"):
                labels.append(0)
                paths.append(str(image_dir / fname))
            elif fname.startswith("dog"):
                labels.append(1)
                paths.append(str(image_dir / fname))
        return paths, labels

    def prepare_data(self) -> None:
        """尝试自动下载 Kaggle 数据集（需要 kaggle API 已配置）。"""
        base = Path(self.hparams.data_dir)
        target = base / "dogs-vs-cats" / "train"
        if target.is_dir() and any(target.glob("cat.*")):
            return

        zip_path = base / "dogs-vs-cats.zip"
        if zip_path.exists():
            extract_dir = base / "dogs-vs-cats"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            inner_zip = extract_dir / "train.zip"
            if inner_zip.exists():
                with zipfile.ZipFile(inner_zip, "r") as zf:
                    zf.extractall(extract_dir)
            return

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            base.mkdir(parents=True, exist_ok=True)
            api.competition_download_files("dogs-vs-cats", path=str(base))
            if zip_path.exists():
                extract_dir = base / "dogs-vs-cats"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                inner_zip = extract_dir / "train.zip"
                if inner_zip.exists():
                    with zipfile.ZipFile(inner_zip, "r") as zf:
                        zf.extractall(extract_dir)
        except Exception:
            pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        image_dir = self._find_image_dir()
        paths, labels = self._scan_images(image_dir)
        assert len(paths) > 0, f"在 {image_dir} 中未找到任何猫狗图像"

        n_total = len(paths)
        n_train = int(n_total * self.hparams.train_ratio)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_total, generator=generator).tolist()

        train_paths = [paths[i] for i in indices[:n_train]]
        train_labels = [labels[i] for i in indices[:n_train]]
        val_paths = [paths[i] for i in indices[n_train:]]
        val_labels = [labels[i] for i in indices[n_train:]]

        self.data_train = CatDogDataset(train_paths, train_labels, self.train_transform)
        self.data_val = CatDogDataset(val_paths, val_labels, self.val_transform)
        self.data_test = self.data_val

        print(f"数据集划分：训练 {len(self.data_train)} 张，验证 {len(self.data_val)} 张")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
