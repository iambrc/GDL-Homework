"""
HW4 - 翻译模型测试脚本
功能：
  1. 加载微调后模型，评估 BLEU
  2. 打印翻译样例
  3. 注意力可视化（Cross-Attention 热力图）
  4. 交互式翻译

运行方式：
    python HW4/src/test.py --model_path <path_to_best_model>
    python HW4/src/test.py --model_path <path_to_best_model> --interact
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


# ======================== 配置 ========================

LANG_PREFIX = ">>cmn<< "
MAX_LENGTH = 128
NUM_BEAMS = 4
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"


def get_chinese_font():
    """获取支持中文的 FontProperties，找不到则返回 None。"""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    local = FONT_DIR / "NotoSansSC.ttf"
    if local.exists():
        return fm.FontProperties(fname=str(local))

    for family in ["Noto Sans CJK SC", "Noto Sans SC", "SimHei",
                    "WenQuanYi Micro Hei", "Microsoft YaHei", "PingFang SC"]:
        hits = [f for f in fm.fontManager.ttflist if family.lower() in f.name.lower()]
        if hits:
            return fm.FontProperties(fname=hits[0].fname)

    try:
        urllib.request.urlretrieve(FONT_URL, str(local))
        return fm.FontProperties(fname=str(local))
    except Exception:
        return None


FONT_PROP = get_chinese_font()


def parse_args():
    parser = argparse.ArgumentParser(description="HW4: Translation Model Testing & Visualization")
    parser.add_argument("--model_path", type=str, required=True, help="微调后模型的路径")
    parser.add_argument("--interact", action="store_true", help="进入交互式翻译模式")
    parser.add_argument("--num_samples", type=int, default=5000, help="评估子采样数量")
    parser.add_argument("--output_dir", type=str, default="./test_outputs", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: torch.device):
    """加载模型和分词器。"""
    print(f"加载模型: {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def evaluate_bleu(model, tokenizer, device, num_samples=5000):
    """在 WMT19 zh-en 验证集上评估 BLEU。"""
    print("加载评估数据集...")
    raw = load_dataset("wmt19", "zh-en", split="train")

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(raw), generator=generator)[:num_samples].tolist()
    raw = raw.select(indices)

    split = raw.train_test_split(test_size=0.1, seed=42)
    val_data = split["test"]

    metric = evaluate.load("sacrebleu")
    all_preds, all_refs = [], []

    print(f"评估 {len(val_data)} 条翻译...")
    batch_size = 32
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i : i + batch_size]
        sources = [LANG_PREFIX + ex["en"] for ex in batch["translation"]]
        references = [ex["zh"] for ex in batch["translation"]]

        encoded = tokenizer(
            sources, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            generated = model.generate(**encoded, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(references)

        if (i // batch_size) % 5 == 0:
            print(f"  进度: {min(i + batch_size, len(val_data))}/{len(val_data)}")

    result = metric.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs],
        tokenize="zh",
    )
    return result["score"], all_preds, all_refs


def print_samples(preds, refs, sources=None, n=5):
    """打印翻译样例。"""
    print("\n" + "=" * 80)
    print("翻译样例对比")
    print("=" * 80)
    for i in range(min(n, len(preds))):
        print(f"\n[样例 {i+1}]")
        if sources:
            print(f"  源文(EN) : {sources[i]}")
        print(f"  参考(ZH) : {refs[i]}")
        print(f"  模型输出 : {preds[i]}")
    print("=" * 80 + "\n")


def translate_with_attention(model, tokenizer, text: str, device):
    """翻译并提取 cross-attention 权重。"""
    inputs = tokenizer(
        LANG_PREFIX + text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=MAX_LENGTH,
            num_beams=1,
        )

    generated_ids = outputs.sequences[0]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tgt_ids = generated_ids[1:]  # skip decoder start token
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids)

    cross_attentions = outputs.cross_attentions
    num_steps = len(cross_attentions)
    src_len = inputs["input_ids"].shape[1]

    # 取最后一层，对所有 head 取平均
    attn_matrix = np.zeros((num_steps, src_len))
    for step in range(num_steps):
        last_layer = cross_attentions[step][-1]  # [batch, heads, 1, src_len]
        attn_matrix[step] = last_layer[0].mean(dim=0).squeeze(0)[:src_len].cpu().numpy()

    # 去除 padding/special tokens
    src_end = len(src_tokens)
    for k, tok in enumerate(src_tokens):
        if tok in ["<pad>", "</s>"]:
            src_end = k
            break
    tgt_end = len(tgt_tokens)
    for k, tok in enumerate(tgt_tokens):
        if tok in ["<pad>", "</s>"]:
            tgt_end = k
            break

    src_tokens = src_tokens[:src_end]
    tgt_tokens = tgt_tokens[:tgt_end]
    attn_matrix = attn_matrix[:tgt_end, :src_end]

    return translation, attn_matrix, src_tokens, tgt_tokens


def plot_attention(attn_matrix, src_tokens, tgt_tokens, save_path, title="Cross-Attention Heatmap"):
    """绘制 cross-attention 热力图（支持中文）。"""
    fp = FONT_PROP
    fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.7), max(6, len(tgt_tokens) * 0.55)))

    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10, fontproperties=fp)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=10, fontproperties=fp)

    ax.set_xlabel("Source Tokens (English)", fontsize=12, fontproperties=fp)
    ax.set_ylabel("Target Tokens (Chinese)", fontsize=12, fontproperties=fp)
    ax.set_title(title, fontsize=14, fontweight="bold", fontproperties=fp)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", fontproperties=fp)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"注意力热力图已保存至：{save_path}")


def interactive_translate(model, tokenizer, device):
    """交互式翻译模式。"""
    print("\n" + "=" * 60)
    print("交互式英中翻译（输入英文句子，输入 'quit' 退出）")
    print("=" * 60)

    while True:
        try:
            text = input("\n[EN] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        encoded = tokenizer(
            LANG_PREFIX + text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            generated = model.generate(**encoded, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)

        translation = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"[ZH] >>> {translation}")


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    if args.interact:
        interactive_translate(model, tokenizer, device)
        return

    # 1. 评估 BLEU
    bleu, preds, refs = evaluate_bleu(model, tokenizer, device, args.num_samples)
    print(f"\nBLEU 分数: {bleu:.2f}")

    # 2. 打印翻译样例
    print_samples(preds, refs, n=5)

    # 3. 注意力可视化
    test_sentences = [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming the world.",
        "I love you.",
    ]

    for i, sent in enumerate(test_sentences):
        print(f"\n注意力可视化：'{sent}'")
        translation, attn_matrix, src_tokens, tgt_tokens = translate_with_attention(
            model, tokenizer, sent, device
        )
        print(f"  翻译结果: {translation}")

        save_path = output_dir / f"attention_heatmap_{i+1}.png"
        plot_attention(
            attn_matrix, src_tokens, tgt_tokens, str(save_path),
            title=f"Cross-Attention: \"{sent}\""
        )

    print(f"\n所有结果已保存至 {output_dir}")

    # 4. 进入交互模式
    print("\n是否进入交互翻译模式？(y/n)")
    try:
        ans = input().strip().lower()
        if ans in ("y", "yes"):
            interactive_translate(model, tokenizer, device)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
