"""
HW4 - WMT19 英中翻译数据模块
从 WMT19 zh-en 加载平行语料，子采样后按 9:1 划分训练/验证集，
使用 MarianTokenizer 编码。
"""

from typing import Optional

import torch
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import MarianTokenizer


class TranslationDataModule(LightningDataModule):

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-zh",
        dataset_name: str = "wmt19",
        dataset_config: str = "zh-en",
        lang_prefix: str = ">>cmn<< ",
        num_samples: int = 50000,
        max_length: int = 128,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
            split="train",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        raw = load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
            split="train",
        )

        n = min(self.hparams.num_samples, len(raw))
        generator = torch.Generator().manual_seed(self.hparams.seed)
        indices = torch.randperm(len(raw), generator=generator)[:n].tolist()
        raw = raw.select(indices)

        split = raw.train_test_split(test_size=0.1, seed=self.hparams.seed)

        tokenizer = self.tokenizer
        lang_prefix = self.hparams.lang_prefix
        max_length = self.hparams.max_length

        def preprocess(examples):
            inputs = [lang_prefix + ex["en"] for ex in examples["translation"]]
            targets = [ex["zh"] for ex in examples["translation"]]

            model_inputs = tokenizer(
                inputs, max_length=max_length, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                text_target=targets, max_length=max_length, truncation=True, padding="max_length"
            )

            label_ids = [
                [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
                for seq in labels["input_ids"]
            ]
            model_inputs["labels"] = label_ids
            return model_inputs

        self.train_dataset = split["train"].map(
            preprocess, batched=True, remove_columns=split["train"].column_names
        )
        self.val_dataset = split["test"].map(
            preprocess, batched=True, remove_columns=split["test"].column_names
        )

        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        print(f"数据集划分：训练 {len(self.train_dataset)} 条，验证 {len(self.val_dataset)} 条")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
