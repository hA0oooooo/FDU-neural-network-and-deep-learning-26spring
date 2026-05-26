# PJ2：CIFAR-10 图像分类与 Batch Normalization 优化分析

本项目在 `PJ2/` 目录下完成两项任务。任务一实现可配置的四阶段 residual CNN，在 CIFAR-10 上比较模型容量、网络结构、优化器、正则化、MixUp/CutMix 数据增强与 Sharpness-Aware Minimization (SAM) 损失目标。最终 `cls_best_sam` 模型包含 `40.49M` 参数，测试准确率为 **96.06%**。任务二实现 VGG-A 与 VGG-A + BatchNorm，在五种 learning rate 下比较分类结果，并围绕优化稳定性输出四张分析图：不同学习率的 loss value 变化范围、局部梯度变化、梯度 Lipschitzness 的方向性经验代理，以及 SAM-style 局部损失上升。

### 1. 环境依赖

实验运行环境为 Python `3.11.15`、PyTorch `2.12.0+cu130`、CUDA `13.0` 和 NVIDIA RTX A6000。若已配置适合本机 CUDA 的 PyTorch 环境，安装其余依赖即可：

```bash
pip install -r requirements.txt
```

### 2. 数据准备

代码使用 `torchvision.datasets.CIFAR10`，首次运行时会自动下载数据至：

```text
data/
└── cifar-10-batches-py/
```

任务一在 CIFAR-10 official train split 中固定取 `45000` 张作为 training set，`5000` 张作为 validation set，任务二不划分 validation set，使用完整 `50000` 张 training images 训练 VGG-A / VGG-A+BN，并在 official test split 上评估。


### 3. 项目结构

```text
PJ2/
├── configs/
│   ├── classification/       # residual CNN 实验配置
│   └── bn/                   # VGG-A / VGG-A+BN 学习率配置
├── data/                     # CIFAR-10 数据集
├── outputs/
│   ├── classification/       # 分类训练记录、模型权重和图像
│   └── bn/                   # BN 训练记录和四张优化分析图
├── scripts/
│   ├── classification.py     # 任务一入口
│   ├── bn.py                 # 任务二入口
│   └── plot.py               # 绘图入口
├── src/
│   ├── models/
│   │   ├── blocks.py         # residual / non-residual blocks
│   │   ├── custom_cnn.py     # 自定义 residual CNN
│   │   └── vgg.py            # VGG-A / VGG-A+BN
│   └── utils/
│       ├── config.py         
│       ├── data.py           
│       ├── losses.py         
│       ├── metrics.py        
│       ├── plot.py           
│       ├── seed.py           
│       └── train.py          
└── requirements.txt
```

### 4. 任务一：CIFAR-10 Classification

##### 4.1 模型与实验

自定义 CNN 使用 stem convolution、四个 residual stages、global average pooling 和两层 classifier，配置文件控制 stage channels、block 数量、激活函数、BatchNorm、Dropout、residual connection、optimizer、label smoothing、MixUp、CutMix 与 SAM。当前 classification 配置包括：

| 配置 | 实验内容 | Params | Test Acc. |
| --- | --- | ---: | ---: |
| `cls_baseline` | Residual CNN baseline | 11.44M | 91.66% |
| `cls_large` | 增加模型宽度 | 20.26M | 92.31% |
| `cls_ultra` | 增加宽度与 blocks | 48.57M | 92.76% |
| `cls_activation_tanh` | ReLU 替换为 Tanh | 11.44M | 86.82% |
| `cls_activation_silu` | ReLU 替换为 SiLU | 11.44M | 92.08% |
| `cls_optimizer_sgd` | AdamW 替换为 SGD + momentum | 11.44M | 92.42% |
| `cls_no_dropout` | 移除 Dropout | 11.44M | 91.57% |
| `cls_no_batchnorm` | 移除 BatchNorm | 11.43M | 88.74% |
| `cls_no_residual` | 移除 residual shortcut | 11.26M | 91.60% |
| `cls_no_normalize` | 移除 input normalization | 11.44M | 92.03% |
| `cls_loss_label_smoothing` | Label smoothing | 11.44M | 91.91% |
| `cls_mixup` | MixUp augmentation | 11.44M | 92.11% |
| `cls_cutmix` | CutMix augmentation | 11.44M | 92.39% |
| `cls_best` | Widened CNN + SiLU + SGD + CutMix | 40.49M | 95.87% |
| `cls_best_sam` | Best + SAM loss objective | 40.49M | **96.06%** |

`cls_best_sam` 使用 SAM 式损失目标

```math
\min_w \max_{\|\epsilon\|_2 \leq \rho} L(w+\epsilon),
```

##### 4.2 训练与评估

训练一个或多个指定配置，或按脚本定义的顺序运行全部 classification 实验，注意 `cls_no_normalize` 将 normalization 设为 `null`，为确保该消融生效，应单独运行。

```bash
python scripts/classification.py --config cls_best.yaml
python scripts/classification.py --config cls_best.yaml cls_best_sam.yaml
python scripts/classification.py
```

#### 4.3 可视化

比较多个实验的训练曲线或单个实验的混淆矩阵：

```bash
python scripts/plot.py curves cls_baseline cls_large cls_ultra \
  --output classification/capacity_comparison.png

python scripts/plot.py curves cls_baseline cls_best cls_best_sam \
  --output classification/final_comparison.png

python scripts/plot.py confusion cls_best
python scripts/plot.py confusion cls_best_sam
```


### 5. 任务二：VGG-A Batch Normalization

##### 5.1 模型与实验

任务二比较以下两种模型分别在五个 learning rate 上训练 `100` epochs：

- `VGG-A`：`Conv -> ReLU`。
- `VGG-A+BN`：`Conv -> BatchNorm -> ReLU`。

| Learning Rate | VGG-A Test Acc. | VGG-A+BN Test Acc. |
| ---: | ---: | ---: |
| `1e-4` | 83.07% | 84.72% |
| `5e-4` | 86.39% | 87.72% |
| `1e-3` | 85.42% | **89.38%** |
| `2e-3` | 84.36% | 88.61% |
| `5e-3` | 74.37% | 88.36% |

#### 5.2 训练

运行全部十组实验并在训练完成后生成分析图，或训练一个指定配置：

```bash
python scripts/bn.py
python scripts/bn.py --config vgga_bn_lr_2e_3.yaml
```

四张分析图分别对应：

- `loss_landscape.png`：不同 learning rate 下 training loss 的变化范围。
- `sam_sharpness.png`：固定 learning rate 轨迹上的近似局部损失上升。
- `gradient_predictiveness.png`：局部参数扰动前后的梯度变化。
- `effective_beta_smoothness.png`：梯度 Lipschitzness 的方向性经验代理，即沿当前梯度方向的有限差分比值。