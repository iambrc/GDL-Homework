# HW5 — 图神经网络（GCN）与 Cora 节点分类

## 目录结构

```
HW5/
├── gcn_cora.py            # 任务一：使用 PyG 的 GCNConv
├── gcn_from_scratch.py    # 任务二：从头实现 GCN（A_hat = D^{-1/2} (A+I) D^{-1/2}）
├── gat_cora.py            # 任务三（可选）：GAT 对比
├── utils.py               # 公共工具：训练循环 / 绘图 / t-SNE / wandb 开关
├── requirements.txt
├── report.md              # 实验报告
└── assets/                # 输出图片
```

## 环境

```bash
pip install -r requirements.txt
# torch_geometric 请根据 torch / cuda 版本选择合适的轮子
# 参考：https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## 运行

Cora 数据集由 `torch_geometric.datasets.Planetoid` 自动下载到 `./data/Cora/`。

```bash
# 任务一：PyG 的 GCN
python gcn_cora.py

# 任务二：从头实现 GCN
python gcn_from_scratch.py

# 任务三（可选）：GAT
python gat_cora.py

# 如需 wandb 记录（参考 HW4 的框架风格）
python gcn_cora.py --use_wandb --wandb_name hw5_gcn_cora
```

每个脚本都支持 `-h` 查看所有可调参数。训练默认 200 个 epoch，纯 CPU 下几秒钟完成。

## 产出文件

| 文件 | 来自 | 说明 |
|---|---|---|
| `assets/gcn_curves.png` | `gcn_cora.py` | 任务一 loss + accuracy 曲线 |
| `assets/gcn_tsne.png` | `gcn_cora.py` | 任务一 64 维隐藏表示的 t-SNE 图 |
| `assets/gcn_scratch_curves.png` | `gcn_from_scratch.py` | 任务二 loss + accuracy 曲线 |
| `assets/gcn_scratch_tsne.png` | `gcn_from_scratch.py` | 任务二 t-SNE 图 |
| `assets/gat_curves.png` | `gat_cora.py` | 任务三 loss + accuracy 曲线 |
| `assets/gat_tsne.png` | `gat_cora.py` | 任务三 t-SNE 图 |

## 超参数（Cora 上 GCN 的标准配置）

| 参数 | 值 |
|---|---|
| `hidden_dim` | 64 |
| `lr` | 0.01 |
| `weight_decay` | 5e-4 |
| `dropout` | 0.5 |
| `epochs` | 200 |
| `optimizer` | Adam |
| `seed` | 42 |

GAT（参数量略大，需要更强正则）：`hidden_dim=8, heads=8, dropout=0.6, lr=0.005`。
