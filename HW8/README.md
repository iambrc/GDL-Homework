# HW8 — 隐式表示：SDF 与 DeepSDF

## 目录结构

```
HW8/
├── deepsdf.py              # 解析 SDF 数据生成 + DeepSDF 训练 + Marching Cubes
├── requirements.txt
├── README.md
├── report.md               # 实验报告（含思考问题）
└── assets/                 # 输出图片 / 权重
    ├── sdf_slices.png              # 任务一：三种形状 z=0 的 SDF 热力图
    ├── deepsdf_loss.png            # 任务二：训练 loss 曲线
    ├── deepsdf_reconstructions.png # 任务三：Marching Cubes 重建结果
    ├── deepsdf_interpolation.png   # 任务四（可选）：隐空间插值
    └── deepsdf.pt                  # 训练好的网络权重 + 隐编码
```

## 环境

```bash
pip install -r requirements.txt
# 若 scikit-image 安装困难，可改用 conda:
# conda install -c conda-forge scikit-image
```

> 本次作业**无需下载任何数据集**——训练数据完全由解析 SDF 公式在线生成。

## 运行

```bash
# 默认 500 epochs，自动检测 CUDA
python deepsdf.py

# 调超参数
python deepsdf.py --epochs 500 --latent_dim 32 --hidden_dim 256 \
                  --batch_size 4096 --lr_model 1e-4 --lr_latent 1e-3 \
                  --resolution 64

# 跳过可选任务四（隐空间插值）
python deepsdf.py --no_interp
```

脚本运行流程（控制台输出）：

1. **任务一**：生成 3 类 × 15 个形状（球体 / 圆环 / 长方体）的解析 SDF 训练数据
   （混合采样：50% 均匀 + 50% 表面附近）；
2. 画 `z = 0` 平面的 SDF 切面热力图，保存为 `assets/sdf_slices.png`；
3. **任务二**：构建 DeepSDF 自解码器（4 层 256d 隐藏 + Tanh 输出 + skip + WeightNorm），
   联合优化网络参数与 `nn.Embedding` 隐编码 500 epochs；
4. 保存训练曲线 `assets/deepsdf_loss.png` 与权重 `assets/deepsdf.pt`；
5. **任务三**：用 Marching Cubes 在 `64³` 网格上提取三种代表形状的零等值面，
   绘制 3D 三角网格，保存为 `assets/deepsdf_reconstructions.png`；
6. **任务四（可选）**：在 sphere → torus 之间线性插值隐编码，
   保存为 `assets/deepsdf_interpolation.png`；
7. 终端打印形状数 / 参数量 / final loss 等摘要——便于和报告中对照填写。

## 超参数（默认）

| 参数 | 值 | 说明 |
|---|---|---|
| `num_per_class` | 15 | 每类形状实例数 |
| `num_pts` | 10000 | 每形状采样点数 |
| `latent_dim` | 32 | 隐编码维度 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `num_layers` | 4 | 隐藏层数（中间层做 skip） |
| `clamp_delta` | 0.1 | SDF 截断阈值 δ |
| `lambda_reg` | 1e-4 | 隐编码 L2 正则化 |
| `batch_size` | 4096 | 小批量大小 |
| `epochs` | 500 | 训练轮数 |
| `lr_model` | 1e-4 | 网络 Adam 学习率 |
| `lr_latent` | 1e-3 | 隐编码 Adam 学习率（10×）|
| `resolution` | 64 | Marching Cubes 网格分辨率 |
| `seed` | 42 | 随机种子 |

## 模型架构速览

```
Input: z (B, 32) ⊕ x (B, 3)                          # 拼接得到 (B, 35)
  → Linear(35, 256) + ReLU                            # 层 0
  → Linear(256, 256) + ReLU                           # 层 1
  → Linear(256+35, 256) + ReLU                        # 层 2 (skip: 再次拼接原始输入)
  → Linear(256, 256) + ReLU                           # 层 3
  → Linear(256, 1) + Tanh                             # → (B, 1)
```

四个隐藏层均用 `nn.utils.weight_norm` 包裹；输出层的 Tanh 把预测限制在
`(-1, 1)`，与 `clamp_delta=0.1` 配合避免梯度死锁。

## 产出文件

| 文件 | 来自 | 说明 |
|---|---|---|
| `assets/sdf_slices.png` | `visualize_sdf_slices()` | 三种形状的 SDF 热力图（红蓝渐变 + 黑色零等值线） |
| `assets/deepsdf_loss.png` | `plot_loss_curve()` | 500 epochs 的训练 loss（对数纵轴） |
| `assets/deepsdf_reconstructions.png` | `visualize_reconstructions()` | Marching Cubes 提取的三类形状三角网格 |
| `assets/deepsdf_interpolation.png` | `visualize_interpolation()` | sphere → torus 隐编码线性插值 |
| `assets/deepsdf.pt` | `main()` | `model_state` + `latent_codes` + `info` + `args` + `losses` |

## 检查表

- [x] 三种形状的解析 SDF 函数（球、环、盒）实现正确；
- [x] 混合采样（50% 均匀 + 50% 表面附近）使近表面区域数据密度充足
      （实测 `|SDF| ≤ 0.1` 的比例约 50%）；
- [x] DeepSDF 自解码器实现：拼接 `z ⊕ x` → MLP + skip + WeightNorm → Tanh；
- [x] 独立的 Adam 优化器、隐编码 LR = 模型 LR 的 10×；
- [x] **Clamped L1 只 clamp target**，不 clamp pred（避免梯度死锁）；
- [x] Marching Cubes 提取零等值面，并把网格坐标映射回 `[-1, 1]^3`；
- [x] 自动检测 CUDA / CPU。
