"""
HW4 - Cross-Attention 注意力可视化（独立脚本）

运行方式：
    python HW4/src/visualize_attention.py --model_path <path_to_model>
    python HW4/src/visualize_attention.py --model_path <path_to_model> --sentences "Hello world." "I love you."
"""

import argparse
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

LANG_PREFIX = ">>cmn<< "
MAX_LENGTH = 128
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"
FONT_FILENAME = "NotoSansSC.ttf"


def ensure_chinese_font() -> str:
    """确保中文字体可用，返回字体路径。"""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    font_path = FONT_DIR / FONT_FILENAME

    if font_path.exists():
        return str(font_path)

    # 尝试查找系统已有的中文字体
    for family in ["Noto Sans CJK SC", "Noto Sans SC", "SimHei", "WenQuanYi Micro Hei",
                    "Microsoft YaHei", "PingFang SC", "STHeiti", "AR PL UMing CN"]:
        matches = [f for f in fm.fontManager.ttflist if family.lower() in f.name.lower()]
        if matches:
            return matches[0].fname

    # 下载 Noto Sans SC
    print(f"下载中文字体 Noto Sans SC -> {font_path}")
    try:
        urllib.request.urlretrieve(FONT_URL, str(font_path))
        print("字体下载完成")
    except Exception as e:
        print(f"字体下载失败: {e}")
        print("尝试使用备用方案...")
        # 备用：用 pip 安装的 matplotlib 自带的 DejaVu 也能显示部分 unicode
        return None

    return str(font_path)


def get_font_prop():
    """获取 matplotlib FontProperties 对象。"""
    font_path = ensure_chinese_font()
    if font_path:
        return fm.FontProperties(fname=font_path)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="HW4: Cross-Attention Visualization")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--sentences", type=str, nargs="+", default=None,
                        help="要可视化的英文句子列表")
    parser.add_argument("--output_dir", type=str, default="./test_outputs", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--layer", type=int, default=-1, help="使用哪一层的注意力 (-1 为最后一层)")
    return parser.parse_args()


def translate_with_attention(model, tokenizer, text: str, device, layer_idx=-1):
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
    tgt_ids = generated_ids[1:]
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids)

    cross_attentions = outputs.cross_attentions
    num_steps = len(cross_attentions)
    src_len = inputs["input_ids"].shape[1]

    attn_matrix = np.zeros((num_steps, src_len))
    for step in range(num_steps):
        layer_attn = cross_attentions[step][layer_idx]  # [batch, heads, 1, src_len]
        attn_matrix[step] = layer_attn[0].mean(dim=0).squeeze(0)[:src_len].cpu().numpy()

    # 去除 padding / special tokens
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


def plot_attention(attn_matrix, src_tokens, tgt_tokens, save_path,
                   title="Cross-Attention Heatmap", font_prop=None):
    """绘制 cross-attention 热力图，支持中文显示。"""
    n_src = len(src_tokens)
    n_tgt = len(tgt_tokens)
    fig, ax = plt.subplots(figsize=(max(8, n_src * 0.7), max(5, n_tgt * 0.55)))

    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n_src))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10,
                       fontproperties=font_prop)
    ax.set_yticks(range(n_tgt))
    ax.set_yticklabels(tgt_tokens, fontsize=10, fontproperties=font_prop)

    ax.set_xlabel("Source Tokens (English)", fontsize=12,
                  fontproperties=font_prop)
    ax.set_ylabel("Target Tokens (Chinese)", fontsize=12,
                  fontproperties=font_prop)
    ax.set_title(title, fontsize=14, fontweight="bold",
                 fontproperties=font_prop)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", fontproperties=font_prop)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_prop = get_font_prop()
    if font_prop:
        print(f"使用字体: {font_prop.get_file()}")
    else:
        print("警告: 未找到中文字体，中文可能无法正常显示")

    print(f"加载模型: {args.model_path}")
    tokenizer = MarianTokenizer.from_pretrained(args.model_path)
    model = MarianMTModel.from_pretrained(args.model_path).to(device)
    model.eval()

    sentences = args.sentences or [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming the world.",
        "I love you.",
        "The weather is nice today.",
        "He went to the store to buy some groceries.",
    ]

    print(f"\n共 {len(sentences)} 条句子待可视化\n")

    for i, sent in enumerate(sentences):
        print(f"[{i+1}/{len(sentences)}] \"{sent}\"")
        translation, attn_matrix, src_tokens, tgt_tokens = translate_with_attention(
            model, tokenizer, sent, device, layer_idx=args.layer
        )
        print(f"  翻译: {translation}")

        save_path = output_dir / f"attention_heatmap_{i+1}.png"
        plot_attention(
            attn_matrix, src_tokens, tgt_tokens, str(save_path),
            title=f"\"{sent}\" → \"{translation}\"",
            font_prop=font_prop,
        )

    print(f"\n所有热力图已保存至 {output_dir}")


if __name__ == "__main__":
    main()
