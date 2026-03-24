"""
HW4 - 英中翻译 LightningModule
封装 MarianMT 模型，训练使用 teacher forcing，验证时自回归生成并计算 BLEU。
"""

from typing import List, Optional

import evaluate
import numpy as np
import torch
from lightning import LightningModule
from transformers import MarianMTModel, MarianTokenizer


class TranslationLitModule(LightningModule):

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-zh",
        max_length: int = 128,
        num_beams: int = 4,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.bleu = evaluate.load("sacrebleu")

        self.val_preds = []
        self.val_refs = []

        self.history = {"train_loss": [], "val_loss": [], "val_bleu": []}

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        generated = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.hparams.max_length,
            num_beams=self.hparams.num_beams,
        )

        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.val_preds.extend(preds)
        self.val_refs.extend(refs)

    def on_validation_epoch_end(self):
        if self.val_preds:
            result = self.bleu.compute(
                predictions=self.val_preds,
                references=[[r] for r in self.val_refs],
                tokenize="zh",
            )
            bleu_score = result["score"]
            self.log("val/bleu", bleu_score, prog_bar=True, sync_dist=True)
            self.history["val_bleu"].append(bleu_score)

        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.history["val_loss"].append(val_loss.item())

        self.val_preds.clear()
        self.val_refs.clear()

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        if train_loss is None:
            train_loss = self.trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            self.history["train_loss"].append(train_loss.item())

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        return optimizer

    def translate(self, texts: List[str], lang_prefix: str = ">>cmn<< ") -> List[str]:
        """将英文文本列表翻译为中文。"""
        inputs = [lang_prefix + t for t in texts]
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.hparams.max_length,
        ).to(self.device)

        generated = self.model.generate(
            **encoded,
            max_length=self.hparams.max_length,
            num_beams=self.hparams.num_beams,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_with_attention(self, text: str, lang_prefix: str = ">>cmn<< "):
        """翻译并返回 cross-attention 权重，用于可视化。"""
        inputs = self.tokenizer(
            lang_prefix + text, return_tensors="pt", truncation=True,
            max_length=self.hparams.max_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=self.hparams.max_length,
            num_beams=1,
        )

        generated_ids = outputs.sequences[0]
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        src_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tgt_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[1:])

        cross_attentions = outputs.cross_attentions
        num_layers = len(cross_attentions[0])
        num_steps = len(cross_attentions)
        src_len = inputs["input_ids"].shape[1]

        attn_matrix = torch.zeros(num_steps, src_len)
        for step in range(num_steps):
            layer_attn = cross_attentions[step][-1]
            attn_matrix[step] = layer_attn[0].mean(dim=0).squeeze(0)[:src_len].cpu()

        return translation, attn_matrix.numpy(), src_tokens, tgt_tokens
