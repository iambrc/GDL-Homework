# HW7 — 体素网格与 3D 卷积神经网络

## 目录结构

```
HW7/
├── voxel_cnn.py            # 点云体素化 + 3D CNN 训练 + 评估 + 可视化
├── requirements.txt
├── README.md
├── report.md               # 实验报告（含思考问题）
└── assets/                 # 输出图片 / 权重
    ├── voxel_vis.png       # 4 个不同类别的体素 3D 散点图
    ├── training_curves.png # loss / accuracy 曲线
    └── voxel_cnn.pt        # 训练好的 3D CNN 权重
```

## 环境

```bash
pip install -r requirements.txt
# torch_geometric 的安装请根据 torch / cuda 版本选择合适的 wheel
# 参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## 数据集

继续使用 **ModelNet10**（与作业六相同）。脚本默认通过 `--data_root`
指向作业六的缓存目录 `../HW6/data/ModelNet10`，避免重新下载和重新采样：

```python
# voxel_cnn.py 的默认值
parser.add_argument("--data_root", type=str,
                    default="../HW6/data/ModelNet10")
```

`pre_transform = Compose([SamplePoints(1024), NormalizeScale()])` 与作业六**完全一致**，
因此 PyG 会直接命中 `processed/training.pt` / `processed/test.pt` 缓存。

## 运行

```bash
# 默认 30 epoch，自动检测 CUDA
python voxel_cnn.py

# 调超参数
python voxel_cnn.py --epochs 30 --batch_size 32 --resolution 32 --lr 1e-3

# 指向自定义数据路径
python voxel_cnn.py --data_root /path/to/ModelNet10
```

脚本运行流程（控制台输出）：

1. 加载 ModelNet10 →（首次）从 mesh 采样 1024 个点 → `NormalizeScale`；
2. **任务一**：把每个点云体素化为 `32 × 32 × 32` 二值占据网格，封装成 `VoxelDataset`；
3. 选 4 个不同类别画 3D 散点图，保存为 `assets/voxel_vis.png`；
4. **任务二**：构建 `VoxelCNN`（3 层 `Conv3d` + FC 分类头），训练 30 epoch；
5. 保存训练曲线 `assets/training_curves.png` 与权重 `assets/voxel_cnn.pt`；
6. 报告参数量、平均 epoch 时间、单样本推理时间——便于和作业六的 PointNet 对比。

## 超参数（默认）

| 参数 | 值 |
|---|---|
| `resolution` | 32 |
| `num_points` | 1024 |
| `batch_size` | 32 |
| `optimizer` | Adam |
| `lr` | 1e-3 |
| `scheduler` | StepLR(step_size=20, gamma=0.5) |
| `epochs` | 30 |
| `dropout` | 0.5 |
| `seed` | 42 |

## 模型架构速览

```
Input: (B, 1, 32, 32, 32)        # 单通道二值体素
  → Conv3d(1, 32, 3, p=1) → BN3d → ReLU → MaxPool3d(2)   # → (B, 32, 16, 16, 16)
  → Conv3d(32, 64, 3, p=1) → BN3d → ReLU → MaxPool3d(2)  # → (B, 64,  8,  8,  8)
  → Conv3d(64,128, 3, p=1) → BN3d → ReLU → MaxPool3d(2)  # → (B, 128, 4,  4,  4)
  → Flatten                                              # → (B, 8192)
  → Linear(8192, 256) + ReLU + Dropout(0.5)
  → Linear(256, 10)                                      # → (B, 10)
```

参数量 ≈ **2.38 M**（其中 `Linear(8192, 256)` 一项就占 ~2.10 M）。

## 产出文件

| 文件 | 来自 | 说明 |
|---|---|---|
| `assets/voxel_vis.png` | `visualize_voxels()` | 4 个不同类别的体素 3D 散点图 |
| `assets/training_curves.png` | `plot_curves()` | 30 epoch 的 train/test loss + accuracy |
| `assets/voxel_cnn.pt` | `fit()` | 训练好的 `VoxelCNN` 权重 |

## 检查表

- [x] 体素化函数正确（先归一化到 `[0, 1]^3`，再离散化、再标记占据）；
- [x] `Dataset` / `DataLoader` 标准接口，可被 `nn.Module` 直接消费；
- [x] 4 个不同类别的可视化图；
- [x] 3 层 `Conv3d` + `BN3d` + `ReLU` + `MaxPool3d` 卷积块，与作业三的 2D CNN 同构；
- [x] 训练曲线（loss + acc）；
- [x] 自动检测 CUDA / CPU；
- [x] 测试准确率 ≥ 80%（参见 `report.md`）。
