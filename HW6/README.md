# HW6 — 点云与 PointNet / PointNet++

## 目录结构

```
HW6/
├── pointnet_cls.py         # PointNet 和 PointNet++ 分类网络（从头实现 + 训练 + 评估）
├── critical_points.py      # （可选）关键点可视化
├── requirements.txt
├── report.md               # 实验报告
├── assets/                 # 输出图片（训练曲线、对比图、关键点）
└── data/                   # ModelNet10（首次运行自动下载）
```

## 环境

```bash
pip install -r requirements.txt
# torch_geometric 的安装请根据 torch / cuda 版本选择合适的 wheel
# 参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## 运行

ModelNet10 数据集会由 `torch_geometric.datasets.ModelNet` 自动下载到 `./data/ModelNet10/`（约 50 MB，首次较慢）。

```bash
# 训练 PointNet 和 PointNet++（默认 30 epoch）
python pointnet_cls.py

# 只跑其中一个模型
python pointnet_cls.py --models pointnet
python pointnet_cls.py --models pointnetpp

# 调超参数 / 随机种子
python pointnet_cls.py --epochs 30 --batch_size 32 --num_points 1024 --seed 42

# 关键点可视化（训练完以后、checkpoint 存在时使用）
python critical_points.py --ckpt assets/pointnet.pt --idx 0
```

脚本会自动检测 CUDA，纯 CPU 下 30 epoch 两个模型大约 40–60 分钟；GPU 上约 15–25 分钟。

## 产出文件

| 文件 | 来自 | 说明 |
|---|---|---|
| `assets/pointnet_curves.png` | `pointnet_cls.py` | PointNet 训练 loss + accuracy 曲线 |
| `assets/pointnetpp_curves.png` | `pointnet_cls.py` | PointNet++ 训练 loss + accuracy 曲线 |
| `assets/comparison.png` | `pointnet_cls.py` | 两个模型的 test accuracy 对比 |
| `assets/pointnet.pt` / `assets/pointnetpp.pt` | `pointnet_cls.py` | 训练好的模型权重（便于关键点可视化） |
| `assets/critical_points.png` | `critical_points.py` | 关键点集可视化（可选任务） |

## 超参数（ModelNet10 上的标准配置）

| 参数 | 值 |
|---|---|
| `num_points` | 1024 |
| `batch_size` | 32 |
| `lr` | 0.001 |
| `optimizer` | Adam |
| `scheduler` | StepLR(step_size=20, gamma=0.5) |
| `epochs` | 30 |
| `reg_weight`（PointNet T-Net 正则） | 1e-3 |

## 架构速览

### PointNet

```
(B, N, 3) → InputTNet(3) → MLP(3→64) → FeatureTNet(64) → MLP(64→128→1024)
         → MaxPool(dim=N) → FC(1024→512→256→10)
```

### PointNet++ (SSG)

```
(B, N=1024, 3) → SA1: (npoint=512, r=0.2, K=32, mlp=[3, 64, 64, 128])
              → SA2: (npoint=128, r=0.4, K=64, mlp=[128+3, 128, 128, 256])
              → SA3: (global, mlp=[256+3, 256, 512, 1024])
              → FC(1024→512→256→10)
```
