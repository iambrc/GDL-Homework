"""
HW2 Task 2 - MNIST 手写数字 GUI 识别程序

使用方法：
    python HW2/src/gui.py --model HW2/logs/train/runs/.../checkpoints/epoch_014.ckpt
    python HW2/src/gui.py --model HW2/logs/train/runs/.../mnist_mlp.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from torchvision import transforms

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.mlp_net import MlpNet

import tkinter as tk
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    sys.exit("请先安装 Pillow: pip install Pillow")

import typing
import functools
import omegaconf
import collections
torch.serialization.add_safe_globals([
    functools.partial, 
    torch.optim.Adam, 
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.nodes.AnyNode,
    omegaconf.nodes.StringNode,
    omegaconf.nodes.BooleanNode,
    omegaconf.nodes.IntegerNode,
    omegaconf.nodes.FloatNode,
    omegaconf.base.Container,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    list,
    dict,
    int,
    collections.defaultdict,
    omegaconf.base.Metadata
])


# --------------------------------------------------------------------------- #
#  常量
# --------------------------------------------------------------------------- #
CANVAS_SIZE = 280       # 显示画布边长（像素）
IMG_SIZE    = 28        # MNIST 输入尺寸
PEN_RADIUS  = 12        # 笔刷半径（画布坐标）
BLUR_RADIUS = 2         # 高斯模糊半径（模拟 MNIST 边缘平滑）
NORMALIZE   = transforms.Normalize((0.1307,), (0.3081,))

# Catppuccin Mocha 配色
BG         = "#1e1e2e"
SURFACE    = "#313244"
TEXT       = "#cdd6f4"
SUBTEXT    = "#a6adc8"
BLUE       = "#89b4fa"
GREEN      = "#a6e3a1"
RED        = "#f38ba8"
OVERLAY    = "#45475a"


# --------------------------------------------------------------------------- #
#  工具函数
# --------------------------------------------------------------------------- #
def load_model(model_path: str) -> torch.nn.Module:
    """加载 MlpNet，同时支持 Lightning .ckpt 和裸 state_dict .pth"""
    net = MlpNet(input_size=784, hidden_sizes=[256, 128], output_size=10, dropout=0.0)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    # Lightning checkpoint 包含 'state_dict' 键，且参数带 'net.' 前缀
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {
            k[len("net."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
        }
    else:
        state_dict = ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return net


# --------------------------------------------------------------------------- #
#  GUI 主类
# --------------------------------------------------------------------------- #
class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, model: torch.nn.Module, model_path: str):
        self.root  = root
        self.model = model
        self.root.title("MNIST Digit Recognizer")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # PIL 画布（与 tk.Canvas 同步，用于实际预测）
        self._new_pil_canvas()
        self._build_ui(model_path)
        self._reset_bar_chart()

    # ------------------------------------------------------------------ #
    #  UI 构建
    # ------------------------------------------------------------------ #
    def _build_ui(self, model_path: str):
        pad = dict(padx=12, pady=8)
        main = tk.Frame(self.root, bg=BG, **pad)
        main.pack()

        # 标题
        tk.Label(
            main, text="Handwritten Digit Recognition  MNIST-MLP",
            bg=BG, fg=TEXT,
            font=tkfont.Font(family="Helvetica", size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(0, 8))

        # 模型路径提示
        short_path = Path(model_path).name
        tk.Label(
            main, text=f"Model: {short_path}",
            bg=BG, fg=SUBTEXT, font=("Helvetica", 9),
        ).grid(row=1, column=0, columnspan=2, pady=(0, 6))

        # ---- 左侧：画布 ----
        left = tk.Frame(main, bg=BG)
        left.grid(row=2, column=0, padx=(0, 14), sticky="n")

        tk.Label(left, text="Draw a digit  (auto-predict on mouse release)",
                 bg=BG, fg=SUBTEXT, font=("Helvetica", 9)).pack()

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=2, highlightbackground=BLUE,
        )
        self.canvas.pack(pady=4)
        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<ButtonPress-1>",   self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: self._predict())

        # 按钮行
        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(pady=4)
        _btn = dict(font=("Helvetica", 11, "bold"), relief=tk.FLAT,
                    padx=22, pady=7, cursor="hand2", bd=0)
        tk.Button(btn_row, text="Clear", bg=RED,   fg=BG,
                  command=self._clear,   **_btn).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="Predict", bg=GREEN, fg=BG,
                  command=self._predict, **_btn).pack(side=tk.LEFT, padx=6)

        # ---- 右侧：结果 + 柱状图 ----
        right = tk.Frame(main, bg=BG)
        right.grid(row=2, column=1, sticky="n")

        tk.Label(right, text="Prediction", bg=BG, fg=SUBTEXT,
                 font=("Helvetica", 10)).pack()

        self.result_var = tk.StringVar(value="?")
        tk.Label(
            right, textvariable=self.result_var,
            bg=BG, fg=BLUE,
            font=tkfont.Font(family="Helvetica", size=72, weight="bold"),
            width=3,
        ).pack()

        self.conf_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.conf_var,
                 bg=BG, fg=GREEN, font=("Helvetica", 11)).pack(pady=(0, 6))

        # matplotlib 柱状图
        self.fig = Figure(figsize=(4.4, 3.3), facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)

        mpl_widget = FigureCanvasTkAgg(self.fig, master=right)
        mpl_widget.get_tk_widget().pack()
        self.mpl_canvas = mpl_widget

    # ------------------------------------------------------------------ #
    #  PIL 画布管理
    # ------------------------------------------------------------------ #
    def _new_pil_canvas(self):
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

    # ------------------------------------------------------------------ #
    #  事件处理
    # ------------------------------------------------------------------ #
    def _on_draw(self, event):
        r = PEN_RADIUS
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear(self):
        self.canvas.delete("all")
        self._new_pil_canvas()
        self.result_var.set("?")
        self.conf_var.set("")
        self._reset_bar_chart()

    # ------------------------------------------------------------------ #
    #  预处理 & 预测
    # ------------------------------------------------------------------ #
    def _preprocess(self) -> torch.Tensor:
        """画布图像 → 28×28 归一化 tensor（与训练预处理一致）"""
        img = self.pil_image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0           # [0, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        tensor = NORMALIZE(tensor)
        return tensor

    def _predict(self):
        if np.array(self.pil_image).max() == 0:
            return  # 画布为空，不预测

        with torch.no_grad():
            logits = self.model(self._preprocess())              # (1, 10)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()  # (10,)

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

        self.result_var.set(str(pred))
        self.conf_var.set(f"Confidence  {conf:.1f}%")
        self._update_bar_chart(probs, pred)

    # ------------------------------------------------------------------ #
    #  柱状图
    # ------------------------------------------------------------------ #
    def _reset_bar_chart(self):
        self.ax.clear()
        self._draw_bar_chart(np.zeros(10), highlight=-1)
        self.mpl_canvas.draw()

    def _update_bar_chart(self, probs: np.ndarray, pred: int):
        self.ax.clear()
        self._draw_bar_chart(probs, highlight=pred)
        self.mpl_canvas.draw()

    def _draw_bar_chart(self, probs: np.ndarray, highlight: int):
        colors = [BLUE] * 10
        if 0 <= highlight <= 9:
            colors[highlight] = GREEN

        self.ax.bar(range(10), probs * 100, color=colors, width=0.7, zorder=2)

        # 样式
        self.ax.set_facecolor(SURFACE)
        self.ax.set_xlim(-0.5, 9.5)
        self.ax.set_ylim(0, 108)
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels([str(i) for i in range(10)],
                                color=TEXT, fontsize=9)
        self.ax.tick_params(axis="y", colors=SUBTEXT, labelsize=8)
        self.ax.set_xlabel("Digit Class", color=SUBTEXT, fontsize=9)
        self.ax.set_ylabel("Confidence (%)", color=SUBTEXT, fontsize=9)
        self.ax.set_title("Prediction Confidence per Class", color=TEXT, fontsize=10)
        self.ax.grid(axis="y", color=OVERLAY, linestyle="--", alpha=0.6, zorder=0)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(OVERLAY)

        # 在预测柱顶标注数值
        if 0 <= highlight <= 9 and probs[highlight] > 0.01:
            self.ax.text(
                highlight, probs[highlight] * 100 + 2,
                f"{probs[highlight] * 100:.1f}%",
                ha="center", va="bottom",
                color=GREEN, fontsize=8, fontweight="bold",
            )


# --------------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="MNIST 手写数字 GUI 识别")
    parser.add_argument(
        "--model", type=str, required=True,
        help="模型权重路径，支持 Lightning .ckpt 或裸 state_dict .pth",
    )
    args = parser.parse_args()

    model_path = args.model

    if not Path(model_path).exists():
        print(f"[错误] 找不到指定的模型文件：{model_path}")
        sys.exit(1)

    print(f"加载模型：{model_path}")
    model = load_model(model_path)

    root = tk.Tk()
    DigitRecognizerApp(root, model, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
