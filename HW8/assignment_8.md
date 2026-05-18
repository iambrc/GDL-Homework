# 作业八：隐式表示 — SDF 与 DeepSDF

**发布日期**：2026 年 4 月 19 日

**截止日期**：2026 年 4 月 26 日 23:59（CST）

---

## 一、作业背景

在前七次作业中，我们沿着几何深度学习的主线，探索了不同数据域和 3D 几何表示上的神经网络架构：

| 作业 | 数据域 / 表示 | 架构 | 核心思想 |
|------|-------------|------|---------|
| 1-2 | 无结构（向量） | MLP | 通用近似 |
| 3 | 规则网格（图像） | CNN | 平移等变 |
| 4 | 序列 | Transformer | 注意力 + 位置编码 |
| 5 | 图 | GCN / GAT | 消息传递，置换等变 |
| 6 | 点云（显式） | PointNet | 共享 MLP + 对称函数 |
| 7 | 体素（显式） | 3D CNN | 3D 卷积 |
| **8** | **SDF（隐式）** | **DeepSDF** | **神经隐式表示** |

作业 6（点云 / PointNet）和作业 7（体素 / 3D CNN）探索了两种**显式**三维表示——它们直接存储几何数据（点的坐标或体素的占有值）。本次作业转向一个全新的范式：**隐式表示**——用一个函数（而非数据结构）来定义形状。

### 什么是 SDF？

**Signed Distance Function（有符号距离函数，SDF）** 将 3D 空间中任意一点映射到其到最近表面的**有符号距离**：

$$f: \mathbb{R}^3 \to \mathbb{R}, \quad f(\mathbf{x}) = \begin{cases} > 0 & \text{点在形状外部} \\ = 0 & \text{点在表面上（零等值面）} \\ < 0 & \text{点在形状内部} \end{cases}$$

SDF 是一种**隐式表示**——形状的表面并不直接存储，而是作为函数的零等值面 $\{\mathbf{x} : f(\mathbf{x}) = 0\}$ 被隐式定义。

### 什么是 DeepSDF？

**DeepSDF**（Park et al., 2019）用神经网络参数化 SDF：

$$f_\theta(\mathbf{z}, \mathbf{x}) \to s$$

其中 $\mathbf{z}$ 是形状的**隐编码（latent code）**，$\mathbf{x}$ 是查询点。关键创新是**自解码器（auto-decoder）**架构：网络参数 $\theta$ 和隐编码 $\{\mathbf{z}_i\}$ 被**联合优化**，无需编码器即可学习形状的紧凑表示。

本次作业使用三种可解析计算 SDF 的基元形状（球体、圆环、长方体）作为训练数据，亲手实现 DeepSDF 的核心流程：

- **任务一**：理解 SDF 的数学定义，编写解析 SDF 函数，生成训练数据
- **任务二**：实现 DeepSDF 自解码器架构，联合优化网络与隐编码
- **任务三**：使用 Marching Cubes 从网络中提取 3D 表面网格
- **任务四（可选）**：在隐空间中插值，观察形状的平滑渐变

---

## 二、环境准备

### 安装依赖

```bash
pip install torch numpy matplotlib scikit-image
```

> **无需外部数据集**：本次作业的训练数据完全由代码生成（解析几何体的 SDF 值），无需下载任何数据集。
>
> **运行环境**：所有任务在 CPU 上约需 **5–10 分钟**，在 GPU 上约 **1–2 分钟**。
>
> **scikit-image** 用于 Marching Cubes 算法（`skimage.measure.marching_cubes`）。若安装困难，可尝试 `conda install scikit-image`。

---

## 三、作业内容

所有任务在同一个文件 `deepsdf.py` 中完成。

---

### 任务一：生成解析 SDF 训练数据

#### 背景知识

对于简单的基元形状，SDF 有解析公式（假设形状中心在原点）：

**球体**（半径 $r$）：

$$f_{\text{sphere}}(\mathbf{x}) = \|\mathbf{x}\| - r$$

**圆环**（主半径 $R$，管道半径 $r$，位于 $xz$ 平面）：

$$f_{\text{torus}}(\mathbf{x}) = \sqrt{\left(\sqrt{x^2 + z^2} - R\right)^2 + y^2} \;-\; r$$

**长方体**（半长 $\mathbf{b} = (b_x, b_y, b_z)$）：

$$f_{\text{box}}(\mathbf{x}) = \left\|\max\!\left(|\mathbf{x}| - \mathbf{b},\; 0\right)\right\| + \min\!\left(\max(|x|-b_x,\; |y|-b_y,\; |z|-b_z),\; 0\right)$$

> 长方体的 SDF 分两部分：第一项处理点在**外部**的情况（到最近面 / 边 / 角的欧式距离），第二项处理点在**内部**的情况（到最近面的负距离）。

#### 具体步骤

1. **实现三种形状的解析 SDF 函数**：支持中心偏移和参数变化（不同半径、不同长宽比等）。

2. **生成数据集**：
   - 三个类别（球体、圆环、长方体），每类 **15** 个形状实例（参数随机变化）
   - 每个实例采样 **10,000** 个查询点，计算对应的真实 SDF 值
   - **推荐**：混合采样策略——50% 在 $[-1, 1]^3$ 中均匀采样，50% 在形状表面附近采样（表面点 + 高斯噪声），使训练数据在近表面区域有足够密度

3. **可视化 SDF 切片**：对每种形状，绘制 $z = 0$ 平面的 SDF 热力图（用 `contourf`），在零等值线处画黑色轮廓线，保存为 `sdf_slices.png`。

#### 代码框架

```python
import numpy as np
import matplotlib.pyplot as plt

# ======================== 解析 SDF 函数 ========================

def sdf_sphere(points, center, radius):
    """球体 SDF: f(x) = ||x - c|| - r"""
    return np.linalg.norm(points - center, axis=-1) - radius

def sdf_torus(points, center, R, r):
    """圆环 SDF"""
    p = points - center
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    # TODO: 实现圆环 SDF 公式
    # Hint: q = sqrt(x^2 + z^2) - R, 然后 sqrt(q^2 + y^2) - r
    ...

def sdf_box(points, center, half_extents):
    """长方体 SDF"""
    q = np.abs(points - center) - half_extents
    # TODO: 实现长方体 SDF 公式
    # Hint: 外部距离 = ||max(q, 0)||, 内部距离 = min(max(q_x, q_y, q_z), 0)
    ...

# ======================== 数据集生成 ========================

def generate_sdf_dataset(num_per_class=15, num_pts=10000):
    """为每类形状生成带参数变化的 SDF 训练数据"""
    all_pts, all_sdf, all_ids, info = [], [], [], []
    shape_id = 0

    for _ in range(num_per_class):
        r = np.random.uniform(0.25, 0.55)
        c = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        pts = np.random.uniform(-1, 1, (num_pts, 3)).astype(np.float32)
        sdf = sdf_sphere(pts, c, r).astype(np.float32)
        all_pts.append(pts)
        all_sdf.append(sdf)
        all_ids.append(np.full(num_pts, shape_id, dtype=np.int64))
        info.append(dict(type='sphere', label=0))
        shape_id += 1

    # TODO: 类似地生成圆环 (R in [0.3,0.5], r in [0.08,0.18]) 和
    #       长方体 (half_extents in [0.15,0.5]) 数据
    ...

    return (np.concatenate(all_pts), np.concatenate(all_sdf),
            np.concatenate(all_ids), info)

# ======================== SDF 切面可视化 ========================

# TODO: 在 z=0 平面创建网格, 对三种形状分别计算 SDF 并用 contourf 绘图
# Hint: 用 np.meshgrid 创建 200x200 网格, 第三维设为 0
```

---

### 任务二：实现并训练 DeepSDF

#### 自解码器（Auto-Decoder）架构

与自编码器（auto-encoder）不同，DeepSDF **没有编码器**。每个形状实例 $i$ 有一个可学习的隐编码 $\mathbf{z}_i \in \mathbb{R}^{d}$（用 `nn.Embedding` 存储），网络 $f_\theta$ 接收隐编码与查询点的**拼接**作为输入：

$$f_\theta: \mathbb{R}^{d+3} \to \mathbb{R}, \quad (\mathbf{z}_i, \mathbf{x}) \mapsto s$$

训练目标——联合优化网络参数 $\theta$ 和所有隐编码 $\{\mathbf{z}_i\}$：

$$\mathcal{L} = \sum_{i} \sum_{(\mathbf{x}_j, s_j^*) \in \Omega_i} \Big| \text{clamp}\big(f_\theta(\mathbf{z}_i, \mathbf{x}_j),\, \delta\big) - \text{clamp}(s_j^*,\, \delta) \Big| + \lambda \|\mathbf{z}_i\|^2$$

其中 $\text{clamp}(x, \delta) = \min(\delta, \max(-\delta, x))$ 将 SDF 值截断到 $[-\delta, \delta]$，使网络聚焦于**近表面区域**。

#### 具体步骤

1. **搭建 DeepSDF 网络**：4 层隐藏层（256 维 + ReLU），输入 $\mathbf{z} \oplus \mathbf{x}$（拼接），输出标量 SDF 值。推荐添加：
   - **Tanh 输出激活**：将预测值限制在 $(-1, 1)$，与 SDF 有界的物理性质一致
   - **Skip Connection**：在中间某层将原始输入 $[\mathbf{z}, \mathbf{x}]$ 重新拼接，保证梯度流回 latent code
   - **Weight Normalization**（可选）：`nn.utils.weight_norm(nn.Linear(...))` 稳定训练

2. **定义隐编码**：`nn.Embedding(num_shapes, latent_dim)`，初始化为 $\mathcal{N}(0, 0.01^2)$。

3. **联合训练**：Adam 优化器更新网络参数和隐编码，500 epochs。推荐将网络参数和隐编码使用**独立的 Adam 优化器**，并为隐编码设置更高的学习率（2–5x）。

4. **绘制训练曲线**：记录每 epoch 的平均 loss，保存为 `deepsdf_loss.png`。

#### 参考超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `latent_dim` | 32 | 隐编码维度 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `num_layers` | 4 | 隐藏层数量 |
| `clamp_delta` | 0.1 | SDF 截断阈值 |
| `lambda_reg` | 1e-4 | 隐编码 L2 正则化系数 |
| `batch_size` | 4096 | 小批量大小 |
| `epochs` | 500 | 训练轮数 |
| `lr` | 5e-4 | 学习率（Adam） |

#### 代码框架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================== DeepSDF 模型 ========================

class DeepSDF(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, num_layers=4):
        super().__init__()
        # TODO: 构建 MLP 网络
        # 输入维度: latent_dim + 3 (隐编码拼接 3D 坐标)
        # 中间层: num_layers 个 [Linear -> ReLU]，推荐加 weight_norm
        # 输出层: Linear -> Tanh -> 1 维  (Tanh 将输出限制在 (-1,1))
        # 可选进阶: 在 num_layers//2 处加 skip connection（将原始输入重新拼接）
        ...

    def forward(self, z, x):
        """z: (B, latent_dim), x: (B, 3) -> (B,)"""
        # TODO: 拼接 z 和 x, 通过网络, 返回 SDF 预测值
        ...

# ======================== 训练配置 ========================

model = DeepSDF(latent_dim=32, hidden_dim=256, num_layers=4).to(device)

# 自解码器的核心: 隐编码是可学习的 Embedding
latent_codes = nn.Embedding(num_shapes, 32).to(device)
nn.init.normal_(latent_codes.weight, 0.0, 0.01)

# 联合优化: 推荐使用独立的 optimizer，并为隐编码设置更高 LR
# 原因: 混合 batch 中每个形状只贡献约 1/num_shapes 的梯度，
#       隐编码需要更高 LR 才能快速分化
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_latent = torch.optim.Adam(latent_codes.parameters(), lr=1e-3)

# ======================== 训练循环 ========================

for epoch in range(500):
    model.train()
    perm = torch.randperm(num_samples)

    for i in range(0, num_samples, 4096):
        idx = perm[i:i+4096]
        pts = pts_t[idx].to(device)       # (B, 3)   查询点
        gt_sdf = sdf_t[idx].to(device)    # (B,)     真实 SDF
        shape_ids = ids_t[idx].to(device)  # (B,)     形状编号

        z = latent_codes(shape_ids)        # (B, 32)  查表得到隐编码
        pred = model(z, pts)               # (B,)     预测 SDF

        # TODO: 计算 Clamped L1 Loss
        # 只对 target (gt_sdf) 做 clamp，不对 pred 做 clamp。
        # 若同时 clamp pred，当 |pred| > delta 时梯度为 0（clamp 饱和），
        # 导致近表面点的错误预测无法被纠正——训练会在第一个 epoch 后完全停滞。
        # Hint: F.l1_loss(pred, torch.clamp(gt_sdf, -delta, delta))
        loss_recon = ...

        # TODO: 隐编码 L2 正则化
        loss_reg = ...

        loss = loss_recon + loss_reg
        optimizer_model.zero_grad()
        optimizer_latent.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_latent.step()

    # TODO: 记录每 epoch 的平均 loss, 每 50 epoch 打印
    # 推荐: 加入 LR scheduler 应对训练平台期
    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30)
```

> ⚠️ **DeepSDF 训练的三个常见陷阱**
>
> 1. **Clamped Loss 梯度死锁**：若对 `pred` 也做 `clamp`，则当 `|pred| > δ` 时梯度完全为 0。这会在第一个 epoch 后造成 loss 完全"冻结"——loss 有值但没有任何梯度更新。**解决**：只 clamp `target`，或对输出层加 Tanh（将预测值限制在 `(-1,1)` 内，使 clamp 很少饱和）。
>
> 2. **均匀采样下近表面信号稀疏**：在 `[-1,1]³` 均匀采样中，对于半径 0.4 的形状，只有约 5% 的点落在 `|SDF| ≤ 0.1` 的近表面带内。Clamped Loss 只在这 5% 的点上有区分形状的梯度——latent code 几乎学不到任何东西。**解决**：混合采样，加入表面附近的采样点。
>
> 3. **模型与 latent code 学习率不匹配**：两者使用相同 LR 时，latent code 在混合 batch 中每轮只从约 2% 的样本获得梯度，实际收敛极慢。**解决**：latent code 的 LR 应为模型 LR 的 2–10 倍。

---

### 任务三：使用 Marching Cubes 提取 3D 表面

训练完成后，需要从网络预测的 SDF 场中**提取几何表面**。**Marching Cubes** 算法从离散的 3D 标量场中提取等值面（iso-surface）的三角网格：

1. 在 $[-1, 1]^3$ 中创建均匀的 $64^3$ 网格
2. 查询网络得到每个网格点的 SDF 值
3. 对每个 $2^3$ 体素单元，根据 8 个角点的正负号查表确定三角面片
4. 拼接所有三角面片得到完整的零等值面网格

#### 具体步骤

1. **构建 3D 网格**并分块查询网络（避免显存 / 内存溢出）。

2. **调用 `marching_cubes`** 提取零等值面。

3. **坐标变换**：Marching Cubes 返回的顶点在**网格坐标系**中（$[0, \text{resolution}-1]$），需映射回**世界坐标系**（$[-1, 1]$）。

4. **3D 可视化**：每类选一个代表形状，用 `Poly3DCollection` 绘制三角网格，保存为 `deepsdf_reconstructions.png`。

#### 代码框架

```python
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def extract_mesh(model, latent_code, resolution=64):
    """从 DeepSDF 中提取零等值面"""
    model.eval()
    grid = np.linspace(-1, 1, resolution)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing='ij')
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # TODO: 分块查询网络 (每块 8192 个点, 避免 OOM)
    # 对每个 chunk:
    #   pts = torch.tensor(chunk).to(device)
    #   z = latent_code.unsqueeze(0).expand(pts.shape[0], -1)
    #   sdf_pred = model(z, pts)
    sdf_volume = ...  # reshape 为 (resolution, resolution, resolution)

    # Marching Cubes
    verts, faces, _, _ = marching_cubes(sdf_volume, level=0.0)

    # TODO: 将顶点从网格坐标映射回世界坐标
    # Hint: verts = verts / (resolution - 1) * 2 - 1
    ...

    return verts, faces

# TODO: 对三类形状各取一个代表, 提取网格并用 Poly3DCollection 绘制 3D 图
```

---

### 任务四（可选）：隐空间插值

在两个形状的隐编码之间**线性插值**，观察 DeepSDF 学到的隐空间是否具有平滑的形状过渡：

$$\mathbf{z}_\alpha = (1 - \alpha)\, \mathbf{z}_A + \alpha\, \mathbf{z}_B, \quad \alpha \in [0, 1]$$

选取不同类别的两个形状（如球体 → 圆环），在 $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$ 处分别提取网格并绘制，保存为 `deepsdf_interpolation.png`。

如果隐空间是**光滑的**，你应该观察到形状在两个端点之间平滑渐变，而非突然跳变。

---

## 四、具体要求

| 项目 | 要求 |
|------|------|
| 编程语言 | Python 3.8+ |
| 框架 | PyTorch |
| 脚本文件 | `deepsdf.py`（所有任务在同一文件中） |
| SDF 切面图 | `sdf_slices.png`：三种形状的 SDF 热力图，零等值线清晰可辨 |
| 训练曲线 | `deepsdf_loss.png`：loss 呈下降趋势 |
| 形状重建 | `deepsdf_reconstructions.png`：三种形状的 Marching Cubes 重建结果 |
| 重建质量 | 重建形状应可识别为对应类别（球体光滑、圆环有孔洞、长方体有棱角） |
| GPU 支持 | 代码自动检测 CUDA，兼容纯 CPU 运行 |
| 代码规范 | 结构清晰，关键步骤有注释 |

---

## 五、运行方式

```bash
python deepsdf.py
```

> **运行时间**：数据生成 + 训练 500 epochs + Marching Cubes + 可视化，CPU 约 **5–10 分钟**，GPU 约 **1–2 分钟**。无需联网下载数据集。

---

## 六、检查方法

1. **SDF 切面图**：`sdf_slices.png` 中黑色轮廓线（零等值线）应呈现正确的形状截面——球体为圆形，圆环为两个小圆，长方体为矩形。红蓝颜色渐变应关于零等值线对称（外部正值为蓝色，内部负值为红色）。

2. **训练收敛**：`deepsdf_loss.png` 中 loss 应持续下降并趋于平稳（典型终值约 0.005–0.02）。若 loss 在第一个 epoch 急剧下降后立即变为完全水平的直线，说明遭遇了梯度死锁（参见任务二中的陷阱提示）。

3. **形状重建**：`deepsdf_reconstructions.png` 中三种形状应可明确识别——球体表面光滑且近似圆形，圆环有清晰的中央孔洞，长方体有可辨别的平面与棱角。

4. **隐空间插值**（可选）：`deepsdf_interpolation.png` 中形状应在两个端点之间平滑过渡。

---

## 七、思考问题（可选）

### Q1 | SDF 的性质与优势

(a) 解释 SDF 三个区域的几何含义：$f(\mathbf{x}) > 0$、$f(\mathbf{x}) = 0$、$f(\mathbf{x}) < 0$ 分别对应什么？

(b) 与**二元占有函数** $o: \mathbb{R}^3 \to \{0, 1\}$（Occupancy Function）相比，SDF 有什么优势？为什么 SDF 更适合用神经网络学习？

(c) SDF 满足 **Eikonal 方程** $\|\nabla f(\mathbf{x})\| = 1$（几乎处处成立）。这个约束的几何含义是什么？如果网络的预测不满足这个约束，会导致什么问题？

> **Hint**：$\|\nabla f\| = 1$ 意味着 SDF 的梯度始终是单位向量，指向离查询点最近的表面点的方向。这保证了 SDF 值确实反映"距离"而非任意标量场。若不满足，提取的零等值面可能有噪声、不光滑或不准确。

---

### Q2 | 自解码器 vs 自编码器

DeepSDF 使用**自解码器（Auto-Decoder）**而非自编码器（Auto-Encoder）来学习形状表示。

(a) 对比自编码器和自解码器的数据流。自编码器的编码器输入是什么？自解码器如何跳过编码器？

(b) 自解码器的训练中，隐编码 $\mathbf{z}_i$ 如何被优化？为什么需要正则化项 $\lambda\|\mathbf{z}_i\|^2$？如果不加正则化会怎样？

(c) 给定一个**训练时未见过的新形状**，自解码器如何获得它的隐编码 $\mathbf{z}^*$？与自编码器的推理过程有何不同？各有什么优劣？

> **Hint**：自编码器推理是 one-shot（一次前向传播得 $\mathbf{z}$），自解码器推理需要**优化**——随机初始化 $\mathbf{z}$，固定网络参数 $\theta$，用梯度下降最小化 $\|f_\theta(\mathbf{z}, \cdot) - s^*\|$。前者快但可能不精确，后者慢但能更精确地拟合新形状。

---

### Q3 | 损失函数设计

(a) 为什么 DeepSDF 使用 **L1 损失**而非 L2 损失？L2 损失在 SDF 学习中可能导致什么问题？

(b) 为什么对 SDF 值做 **clamping**（截断到 $[-\delta, \delta]$）？这对学习效果有什么影响？

(c)（进阶）一些方法（如 IGR, SIREN）在 loss 中加入 **Eikonal 正则化** $\mathcal{L}_{\text{eik}} = \mathbb{E}\left[(\|\nabla_{\mathbf{x}} f_\theta\| - 1)^2\right]$。这需要对网络输出关于输入 $\mathbf{x}$ 求梯度。在 PyTorch 中如何实现？

> **Hint**：对于 (a)，L2 对远离表面的大误差给予过高权重，导致网络"偏向"拟合远处的值而忽略近表面细节。L1 对所有距离同等重视。对于 (c)，使用 `torch.autograd.grad(outputs=sdf, inputs=points, grad_outputs=torch.ones_like(sdf), create_graph=True)` 可在保持计算图的情况下求空间梯度。

---

### Q4 | 隐式 vs 显式 3D 表示

填写以下对比表：

| 表示 | 存储方式 | 内存与分辨率的关系 | 拓扑灵活性 | 表面提取方式 | 典型应用 |
|------|---------|-------------------|-----------|------------|---------|
| 点云 | ？ | ？ | ？ | ？ | ？ |
| 体素 | ？ | ？ | ？ | ？ | ？ |
| 网格 (Mesh) | ？ | ？ | ？ | ？ | ？ |
| SDF (Neural) | ？ | ？ | ？ | ？ | ？ |

并回答：为什么说隐式表示在**分辨率**和**拓扑**方面具有天然优势？

> **Hint**：显式表示的精度受固定离散化限制（点数、体素分辨率、三角面数）。神经隐式表示可以在**任意精度**查询（网络是连续函数），且天然处理任意拓扑（亏格变化、不连通分量等），无需显式定义连接关系。

---

### Q5 | 从 DeepSDF 到 NeRF

**Neural Radiance Field (NeRF)**（Mildenhall et al., 2020）是神经隐式表示在**新视角合成**领域的标志性应用。

(a) NeRF 的隐式函数为 $F_\Theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$，其中 $\mathbf{x}$ 是 3D 位置，$\mathbf{d}$ 是观察方向，$\mathbf{c}$ 是颜色，$\sigma$ 是体密度。与 DeepSDF 的 $f_\theta(\mathbf{z}, \mathbf{x}) \to s$ 对比，两者有什么相同点和不同点？

(b) DeepSDF 通过 Marching Cubes 提取表面。NeRF 通过**体渲染（Volume Rendering）**生成图像。为什么 NeRF 不用 Marching Cubes？这两种"从隐式表示到可视化"的方式各有什么特点？

(c) DeepSDF 用隐编码 $\mathbf{z}$ 区分不同形状。NeRF 的原始版本为每个场景训练一个独立网络。后续工作如何解决"一个网络表示多个场景"的问题？

> **Hint**：(c) 参考 PixelNeRF（图像编码器提取条件特征，类似自编码器路线）和 CodeNeRF（可优化隐编码，类似 DeepSDF 的自解码器路线）。

---

### Q6 | 3D 表示全景对比

综合作业 6–8 所学，完成以下综合对比表（预留 BRep 列供下周参考）：

| | 点云 | 体素 | 网格 (Mesh) | SDF (Neural) | BRep |
|---|------|------|-------------|-------------|------|
| 数据结构 | ？ | ？ | ？ | ？ | ？ |
| 对称性 | ？ | ？ | ？ | ？ | ？ |
| 典型网络 | ？ | ？ | ？ | ？ | ？ |
| 分辨率限制 | ？ | ？ | ？ | ？ | ？ |
| 拓扑灵活性 | ？ | ？ | ？ | ？ | ？ |
| 工程应用 | ？ | ？ | ？ | ？ | ？ |

---

## 八、提交要求

将以下文件打包为 `学号_姓名_HW8.zip`：

- `deepsdf.py` — 完整代码（数据生成、模型、训练、可视化）
- `sdf_slices.png` — SDF 切面可视化
- `deepsdf_loss.png` — 训练 loss 曲线
- `deepsdf_reconstructions.png` — Marching Cubes 重建结果
- 终端输出截图（包含训练过程和最终输出）
- （可选）`deepsdf_interpolation.png` — 隐空间插值结果

通过课程 QQ 群的作业系统上传。

---

## 九、参考依赖

```
torch
numpy
matplotlib
scikit-image
```
