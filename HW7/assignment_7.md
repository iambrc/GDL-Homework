# 作业七：体素网格与 3D 卷积神经网络

**发布日期**：2026 年 4 月 12 日

**截止日期**：2026 年 4 月 19 日 23:59（CST）

---

## 一、作业背景

在作业六中，我们学习了**点云**——用无序点集表示 3D 形状，并用 PointNet 的"共享 MLP + 对称函数"实现了置换不变的分类器。点云虽然灵活，但它是**无结构**的——点与点之间没有空间邻域关系，PointNet 的 max pooling 也因此丢失了大量局部几何信息。

本次作业我们换一种思路：将 3D 形状放在**规则网格**上——这正是**体素（Voxel）** 表示。体素之于 3D 空间，正如像素之于 2D 图像：

| | 2D（作业三） | 3D（本次作业） |
|---|---|---|
| 表示 | 像素网格 $H \times W$ | 体素网格 $R \times R \times R$ |
| 值 | RGB 颜色 $\in [0,255]^3$ | 占据状态 $\in \{0, 1\}$ |
| 架构 | 2D CNN（`Conv2d`） | 3D CNN（`Conv3d`） |
| 对称性 | 2D 平移等变 | 3D 平移等变 |

有了规则网格结构，作业三中 2D CNN 的所有技术——局部卷积、权重共享、池化——都可以直接推广到 3D。这是最自然、最直觉的 3D 深度学习方法——但也付出了沉重的代价：**内存随分辨率呈立方增长**。$32^3 \approx 32\text{K}$ 个体素还能处理，$128^3 \approx 210$万个体素就已经吃力，$512^3 \approx 1.34$亿个体素则完全不可行。

本次作业延续作业六的 **ModelNet10** 数据集，完成两个任务：

- **任务一**：将点云转换为体素网格，理解体素化的过程和信息损失
- **任务二**：搭建 3D CNN 分类器，将作业三的 2D CNN 经验推广到 3D

> **从作业三到作业七的桥梁**：如果你理解了 2D CNN 中 `Conv2d`、`BatchNorm2d`、`MaxPool2d` 的作用，本次作业只需要在每个维度后面加一个"3d"——`Conv3d`、`BatchNorm3d`、`MaxPool3d`——架构思想完全一致。

---

## 二、环境准备

### 安装依赖

```bash
pip install torch torchvision
pip install torch_geometric
pip install matplotlib numpy
```

> **PyTorch Geometric 安装**：如果已在作业六中安装过 `torch_geometric`，可跳过此步。若 `pip install torch_geometric` 报错，请参考 [官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

### 数据集

本次作业继续使用 **ModelNet10** 数据集（与作业六相同），由 PyG 自动下载。如果作业六已下载过数据，指向相同的 `root` 路径即可复用。

> **首次运行提示**：数据集下载和点云预处理（从网格采样 1024 个点）需要几分钟时间，后续运行会直接加载缓存。如果下载受限，可设置镜像或手动下载后放入对应目录。

---

## 三、作业内容

### 任务一：点云到体素网格转换（`voxel_cnn.py`）

将 ModelNet10 的点云数据转换为 $32 \times 32 \times 32$ 的二值占据网格（binary occupancy grid），并可视化。

#### 具体步骤

1. **加载 ModelNet10 数据集**：使用 PyG 的 `ModelNet` 类加载数据，用 `SamplePoints(1024)` 将网格模型采样为 1024 个点的点云。打印数据集基本信息（训练/测试样本数、类别数）。

2. **实现体素化函数**：将点云归一化到 $[0, 1]^3$，然后离散化到 $R \times R \times R$ 的网格上，将有点落入的体素标记为 1（占据），其余为 0（空）。

3. **批量体素化**：将所有训练和测试样本转换为体素网格，封装为 PyTorch `Dataset`，用 `DataLoader` 加载。

4. **可视化**：选取 4 个不同类别的体素化样本，用 `matplotlib` 的 3D 散点图绘制占据体素，保存为 `voxel_vis.png`。

#### 体素化伪代码

```python
def point_cloud_to_voxel(pos, resolution=32):
    """
    Args:
        pos: (N, 3) 点云坐标
        resolution: 体素网格分辨率 R
    Returns:
        grid: (R, R, R) 二值占据网格
    """
    # Step 1: 归一化到 [0, 1]^3
    # Hint: center = (pos.max + pos.min) / 2, 然后平移 + 缩放
    
    # Step 2: 映射到网格索引 [0, R-1]
    # Hint: indices = (pos_normalized * (R - 1)).long().clamp(0, R - 1)
    
    # Step 3: 创建空网格，设置占据体素
    # grid = torch.zeros(R, R, R)
    # grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    ...
```

---

### 任务二：搭建并训练 3D CNN 分类器（`voxel_cnn.py`）

在体素化数据上搭建 3D 卷积神经网络，完成 ModelNet10 的 10 类分类任务。

#### 模型架构

使用 3 个卷积块 + 全连接分类头，每个卷积块包含 `Conv3d → BatchNorm3d → ReLU → MaxPool3d`：

```
Input: (B, 1, 32, 32, 32)          # 单通道二值体素
  → Conv3d(1, 32, 3, padding=1)    # → (B, 32, 32, 32, 32)
  → BN3d + ReLU + MaxPool3d(2)     # → (B, 32, 16, 16, 16)
  → Conv3d(32, 64, 3, padding=1)   # → (B, 64, 16, 16, 16)
  → BN3d + ReLU + MaxPool3d(2)     # → (B, 64, 8, 8, 8)
  → Conv3d(64, 128, 3, padding=1)  # → (B, 128, 8, 8, 8)
  → BN3d + ReLU + MaxPool3d(2)     # → (B, 128, 4, 4, 4)
  → Flatten                         # → (B, 8192)
  → Linear(8192, 256) + ReLU + Dropout(0.5)
  → Linear(256, 10)                 # → (B, 10)
```

#### 参考超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `resolution` | 32 | 体素网格分辨率 |
| `num_points` | 1024 | 每个形状采样的点数 |
| `batch_size` | 32 | 训练 batch 大小 |
| `learning_rate` | 1e-3 | Adam 学习率 |
| `num_epochs` | 30 | 训练轮数 |

#### 训练要求

- 记录每个 epoch 的训练 loss 和测试准确率
- 绘制 loss 和 accuracy 曲线，保存为 `training_curves.png`
- 打印最终测试集准确率（目标 ≥ **80%**）

#### 代码框架（供参考）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ======================== 配置 ========================

RESOLUTION = 32
NUM_POINTS = 1024
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
NUM_CLASSES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ======================== 加载 ModelNet10 ========================

pre_transform = T.SamplePoints(NUM_POINTS)
train_pyg = ModelNet(root='./data/ModelNet10', name='10', train=True, pre_transform=pre_transform)
test_pyg = ModelNet(root='./data/ModelNet10', name='10', train=False, pre_transform=pre_transform)

CLASS_NAMES = train_pyg.raw_file_names  # 类别名称列表
print(f"训练集: {len(train_pyg)} 样本, 测试集: {len(test_pyg)} 样本, 类别数: {NUM_CLASSES}")

# ======================== 体素化 ========================

def point_cloud_to_voxel(pos, resolution=32):
    # TODO: 实现体素化
    # Step 1: 归一化到 [0, 1]^3
    # Step 2: 映射到网格索引
    # Step 3: 填充体素网格
    ...

# ======================== 数据集封装 ========================

class VoxelDataset(Dataset):
    def __init__(self, pyg_dataset, resolution=32):
        # TODO: 遍历 pyg_dataset，对每个样本调用 point_cloud_to_voxel
        # 存储体素网格和标签
        ...

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回 (1, R, R, R) 的体素网格和标签
        return self.voxels[idx].unsqueeze(0), self.labels[idx]

print("体素化训练集...")
train_dataset = VoxelDataset(train_pyg, RESOLUTION)
print("体素化测试集...")
test_dataset = VoxelDataset(test_pyg, RESOLUTION)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================== 体素可视化 ========================

# TODO: 选取 4 个不同类别的样本，用 matplotlib 3D 散点图可视化
# Hint: ax = fig.add_subplot(1, 4, i+1, projection='3d')
#        occupied = np.argwhere(grid > 0.5)
#        ax.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2], ...)

# ======================== 3D CNN 模型 ========================

class VoxelCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: 定义 3 个 Conv3d+BN+ReLU+MaxPool 卷积块
        # TODO: 定义 FC 分类头
        ...

    def forward(self, x):
        # TODO: 前向传播
        ...

# ======================== 训练与评估 ========================

model = VoxelCNN(NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# TODO: 训练循环
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     for voxels, labels in train_loader:
#         voxels, labels = voxels.to(device), labels.to(device)
#         ...
#     model.eval()
#     # 计算测试准确率
#     ...

# TODO: 绘制 loss 和 accuracy 曲线，保存为 training_curves.png
# TODO: 打印最终测试准确率
```

---

### 任务三（可选）：PointNet vs 3D CNN 对比

将本次作业的 3D CNN 与作业六的 PointNet 在 ModelNet10 上进行对比分析。

**对比维度**：

| 维度 | PointNet（作业六） | 3D CNN（本次作业） |
|------|-------------------|-------------------|
| 输入表示 | 点云 (N, 3) | 体素网格 (1, R, R, R) |
| 测试准确率 | ？% | ？% |
| 模型参数量 | ？M | ？M |
| 单 epoch 训练时间 | ？s | ？s |
| 单样本推理时间 | ？ms | ？ms |
| 输入数据大小 | ？ | ？ |

分析两种方法各自的优劣，讨论在什么场景下应选择哪种方法。

---

## 四、具体要求

| 项目 | 要求 |
|------|------|
| 编程语言 | Python 3.8+ |
| 框架 | PyTorch + PyTorch Geometric（数据加载） |
| 数据集 | ModelNet10（10 类，~4,899 个形状） |
| 体素分辨率 | $32 \times 32 \times 32$ |
| 模型 | 3 层 Conv3d + FC 分类头 |
| 测试准确率 | ≥ **80%** |
| 可视化 | 体素可视化 `voxel_vis.png` + 训练曲线 `training_curves.png` |
| GPU 支持 | 代码自动检测 CUDA，兼容纯 CPU 运行 |
| 代码规范 | 结构清晰，关键步骤有注释 |

---

## 五、运行方式

```bash
python voxel_cnn.py
```

> **运行时间估计**：
> - **首次运行**：ModelNet10 下载和点云采样约需 3–5 分钟（缓存后后续运行跳过）
> - **体素化**：约 1–2 分钟（预处理所有样本）
> - **训练**：GPU 上 30 个 epoch 约需 5–10 分钟；纯 CPU 约需 30–60 分钟
> - 如需快速验证流程，可将 `NUM_EPOCHS` 调小至 5
>
> **复用作业六数据**：如果作业六已下载 ModelNet10 到 `./data/ModelNet10`，且使用了相同的 `SamplePoints(1024)` 作为 `pre_transform`，本次作业会直接加载缓存，无需重新下载。

---

## 六、检查方法

1. **体素可视化**：`voxel_vis.png` 中的 3D 散点图能清晰识别出不同类别的 3D 形状轮廓（如椅子、桌子、马桶等）。
2. **训练流程**：脚本正常运行至完成，控制台打印每 epoch 的 loss 和测试准确率。
3. **准确率达标**：最终测试集准确率 ≥ 80%。
4. **曲线图**：`training_curves.png` 显示 loss 持续下降、accuracy 持续上升的合理趋势。

---

## 七、思考问题（可选）

### Q1 | 从 2D 到 3D：卷积核的参数爆炸

作业三中，一个 $3 \times 3$ 的 2D 卷积核（输入通道 $C_{\text{in}}$，输出通道 $C_{\text{out}}$）有 $3 \times 3 \times C_{\text{in}} \times C_{\text{out}}$ 个参数。

(a) 本次作业中，一个 $3 \times 3 \times 3$ 的 3D 卷积核有多少参数？相比 2D 增加了多少倍？如果进一步推广到 4D（如时空视频卷积 $3 \times 3 \times 3 \times 3$），参数又是多少？

(b) 参数增加只是问题的一部分。更严重的是**特征图的大小**：一个 $256 \times 256$ 的 2D 特征图有 65,536 个元素；相同分辨率的 3D 特征图 $256 \times 256 \times 256$ 有多少个元素？需要多少 GB 显存（假设 FP32，64 通道）？

(c) 这就是为什么本次作业只能用 $32^3$ 分辨率。对比：作业三的图像分辨率为 $224 \times 224 \approx 5$万像素，而 $32^3 \approx 3.3$万体素。3D CNN 在远低于 2D CNN 的"等效分辨率"下工作，这对 3D 形状识别意味着什么？

> **Hint**：从 2D 到 3D，卷积核从 $k^2$ 增长到 $k^3$（仅增长 $k$ 倍），但特征图从 $R^2$ 增长到 $R^3$——特征图的增长才是真正的瓶颈。$32^3$ 分辨率下，一把椅子的细腿可能只占 1-2 个体素宽，大量细节被丢失。

---

### Q2 | 稀疏性：大多数体素是空的

3D 形状的表面是嵌入在 3D 空间中的**2D 流形**。这意味着占据的体素数量与分辨率的关系约为 $O(R^2)$，而总体素数量为 $O(R^3)$。

(a) 计算不同分辨率下的理论占据率 $\rho = O(R^2) / O(R^3) = O(1/R)$：

| 分辨率 $R$ | 总体素 $R^3$ | 表面体素 $\sim R^2$ | 占据率 $\rho$ |
|-----------|-------------|-------------------|-------------|
| 32 | ？ | ？ | ？ |
| 64 | ？ | ？ | ？ |
| 128 | ？ | ？ | ？ |
| 256 | ？ | ？ | ？ |

(b) 标准 `Conv3d` 对所有 $R^3$ 个体素都执行卷积运算——包括那些值为 0 的空体素。这意味着什么分辨率下有多少百分比的计算是"浪费"的？

(c) **稀疏卷积（Sparse Convolution）**（如 MinkowskiEngine、SpConv）的核心思想是什么？它如何利用体素的稀疏性？将计算复杂度从什么降低到了什么？

(d) 对比点云（PointNet）和体素（3D CNN）处理"稀疏性"的方式：PointNet 天然只处理表面上的点，没有浪费；3D CNN 必须处理整个网格。这是否意味着 3D CNN 在计算效率上永远劣于 PointNet？稀疏卷积如何改变这个结论？

> **Hint**：稀疏卷积只在非空体素及其邻域上执行卷积，维护一个"活跃体素集合"，复杂度从 $O(R^3)$ 降为 $O(N_{\text{occupied}})$，其中 $N_{\text{occupied}} \sim R^2$。这使得高分辨率（$128^3$、$256^3$）的 3D 卷积成为可能。MinkowskiEngine 是这一思路的代表性实现。

---

### Q3 | 平移等变 vs 旋转等变

(a) 作业三中，2D CNN 对 2D 图像的平移是等变的——将输入图像水平移动 $(\Delta x, \Delta y)$，输出特征图也相应移动。类比地，3D CNN 对 3D 体素网格的平移 $(\Delta x, \Delta y, \Delta z)$ 也是等变的。请简要解释为什么 3D 卷积天然保证了 3D 平移等变性（与 2D 完全类似的论证）。

(b) 如果将一个体素化的椅子绕 Y 轴旋转 45°，3D CNN 的输出会怎样？它还是同一把椅子的表示吗？为什么 3D CNN **不是**旋转等变的？

(c) 实践中如何缓解旋转不变性的缺失？以下三种策略各有什么优劣？

| 策略 | 做法 | 优势 | 劣势 |
|------|------|------|------|
| 数据增强 | 训练时随机旋转输入 | ？ | ？ |
| 规范朝向 | 将所有形状对齐到标准朝向 | ？ | ？ |
| 等变网络 | 设计旋转等变架构（如 SE(3)-Transformer） | ？ | ？ |

(d) 回顾课程主线，填写下表中的对称性：

| 架构 | 数据域 | 内置的对称性 | 不具备的对称性 |
|------|--------|------------|-------------|
| 2D CNN（作业三） | 2D 图像 | ？ | ？ |
| 3D CNN（本次作业） | 3D 体素 | ？ | ？ |
| PointNet（作业六） | 点云 | ？ | ？ |

> **Hint**：对于 (b)，旋转 45° 后，原本对齐在网格上的体素会落在两个网格之间——需要重新离散化，导致**混叠（aliasing）** 和信息损失。网格本身只有 90° 的旋转对称性（$C_4$ 群），而非连续旋转群 $SO(3)$。对于 (d)，PointNet 的 max pooling 对输入点的**排列**是不变的，但对旋转不是（除非加 T-Net）。

---

### Q4 | 体素化的信息损失

从点云到体素的转换是一个**有损过程**——多个不同的点可能落入同一个体素，而体素只记录"有/无"。

(a) 考虑一个 $32^3$ 的体素网格和一个包含 1024 个点的点云。平均每个体素最多只能容纳多少个点？如果有 $k$ 个点落入同一个体素，二值体素保留了多少信息？

(b) 将同一个 3D 形状分别用 $16^3$、$32^3$、$64^3$ 分辨率体素化。随着分辨率增加，体素化的"保真度"如何变化？存在一个理论上限吗？（提示：当分辨率高到每个体素最多包含一个点时，体素表示与原始点云等价。）

(c) 二值占据网格（0/1）之外，还有哪些方式可以编码体素的值？例如，如果记录每个体素内的**点密度**或**法向量统计**，对分类任务可能有什么帮助？

> **Hint**：对于 (a)，1024 个点分布在 $32^3 = 32768$ 个体素中，平均每个体素不到 0.1 个点。但点分布在表面上（约 $32^2 \approx 1024$ 个表面体素），所以表面体素平均约 1 个点。二值体素只记录"有点"而非"有几个点"，损失了密度信息。

---

### Q5 | 3D 表示大比拼

在学完点云（作业六）和体素（本次作业）后，让我们系统地比较不同的 3D 表示。填写下表：

| 表示 | 数据结构 | 内存复杂度 | 分辨率 | 拓扑信息 | 典型架构 | 优势 | 劣势 |
|------|---------|-----------|--------|---------|---------|------|------|
| 点云 | ？ | ？ | ？ | ？ | ？ | ？ | ？ |
| 体素 | ？ | ？ | ？ | ？ | ？ | ？ | ？ |
| 网格（Mesh） | ？ | ？ | ？ | ？ | ？ | ？ | ？ |
| SDF | ？ | ？ | ？ | ？ | ？ | ？ | ？ |

> **Hint**：
> - 点云：无序点集 $(N, 3)$，内存 $O(N)$，无拓扑信息
> - 体素：规则网格 $(R, R, R)$，内存 $O(R^3)$，隐含了空间邻域
> - 网格：顶点 + 面（$V + F$），内存 $O(V + F)$，完整的拓扑连接
> - SDF：隐式函数 $f: \mathbb{R}^3 \to \mathbb{R}$，内存 = 网络参数，分辨率无限
>
> 这张表在作业八（SDF）和作业九（BRep）中会继续补充。

---

## 八、提交要求

将以下文件打包为 `学号_姓名_HW7.zip`：

- `voxel_cnn.py` — 体素化 + 3D CNN 训练脚本
- `voxel_vis.png` — 体素可视化图
- `training_curves.png` — 训练曲线图
- 终端输出截图（包含最终测试准确率）
- （可选）PointNet vs 3D CNN 对比分析

通过课程 QQ 群的作业系统上传。

---

## 九、参考依赖

```
torch
torch_geometric
matplotlib
numpy
```
