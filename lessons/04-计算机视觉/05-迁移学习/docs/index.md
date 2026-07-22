# 迁移学习

> 别人花了百万 GPU 小时教网络认识边缘和纹理。你应该先借用这些特征，再训练自己的任务。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 04 · 03（卷积神经网络）、阶段 04 · 04（图像分类）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 04 · 07（目标检测）— 检测模型的骨干网络来自迁移学习；阶段 12 · 03（视觉语言模型）— CLIP 的视觉编码器是迁移学习的典型应用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分特征提取和微调两种迁移策略，根据数据集大小、领域距离和算力预算选择正确方案
- [ ] 加载预训练骨干网络，替换分类头，用不超过 20 行代码训练出可用的基线模型
- [ ] 使用区分学习率逐步解冻层——让早期通用特征获得更小的更新幅度
- [ ] 诊断迁移学习中三个最常见的故障：学习率过高导致的特征漂移、BatchNorm 统计量在小数据集上的崩溃、以及灾难性遗忘

---

## 1. 问题

训练一个 ResNet-50 在 ImageNet 上达到 76% Top-1 准确率，大约需要 2,000 GPU 小时。几乎没有团队能为每个新任务都付出这个代价。

但几乎所有团队真正在交付的，都是一个预训练骨干网络加上一个在几百到几千张任务特定图像上训练的新分类头。这不是偷懒，这是工程现实。

关键在于：任何在 ImageNet 上训练过的 CNN，第一个卷积块学到的是边缘和 Gabor 滤波器。接下来的几个块学到纹理和简单图案。中间块学到物体的部件。最后的块学到接近 1,000 个 ImageNet 类别的组合特征。这个层次结构的前 90% 几乎可以无修改地迁移到医学影像、工业检测、卫星数据和几乎所有其他视觉任务——因为自然界中边缘和纹理的词汇量是有限的。最后 10% 才是你真正需要训练的部分。

但迁移学习有三个隐蔽的陷阱等着你：用过高的学习率摧毁预训练特征、冻结太多层导致模型信息不足、BatchNorm 的运行统计量漂移到一个网络从未见过的小数据分布上。本课将逐一演示这些陷阱以及如何规避。

---

## 2. 概念

### 2.1 直观理解

迁移学习的核心思想可以用一句话概括：**在大数据上学到的通用特征，在小数据上依然有用。**

想象一个学过 1,000 种物体的画家。当你让他画一种他没见过的新物体时，他不需要从零学起——他仍然知道什么是边缘、什么是纹理、什么是光影。他只需要学的是"新物体长什么样"，而不是"什么是边缘"。

这就是迁移学习：预训练网络的浅层学到的是通用视觉特征（边缘、纹理、对比度），这些特征在几乎所有视觉任务中都适用。深层学到的是任务特定的特征（ImageNet 的 1,000 个类别），需要根据新任务调整。

```
预训练 ResNet 的层次结构：

  浅层（conv1, layer1）          深层（layer3, layer4, fc）
  ┌─────────────────────┐       ┌─────────────────────┐
  │ 边缘、方向、对比度    │       │ 物体部件、类别组合    │
  │ 纹理、简单图案        │ ──→  │ ImageNet 1000 类    │
  │                     │       │                     │
  │ 通用，可直接迁移      │       │ 任务特定，需要调整    │
  └─────────────────────┘       └─────────────────────┘
         ↑                              ↑
      冻结或微调                    替换或重新训练
```

### 2.2 特征提取 vs 微调

两种策略，取决于你对预训练特征的信任程度和你有多少数据。

```
特征提取（骨干冻结）：                微调（端到端）：

  预训练骨干                          预训练骨干
  ┌──────────────────┐               ┌──────────────────┐
  │  所有层冻结       │               │  所有层可训练     │
  │  无梯度流动       │               │  但学习率不同     │
  └────────┬─────────┘               └────────┬─────────┘
           │                                  │
     ┌─────▼─────┐                      ┌─────▼─────┐
     │  新分类头  │                      │  新分类头  │
     │  训练     │                      │  正常 LR   │
     └───────────┘                      └───────────┘
```

选择指南：

| 数据集大小 | 领域距离 | 推荐方案 |
|---|---|---|
| < 1,000 张 | 接近 ImageNet | 冻结骨干，仅训练分类头 |
| 1,000 - 10,000 张 | 接近 | 冻结前 2-3 个阶段，微调其余部分 |
| 10,000 - 100,000 张 | 任意 | 端到端微调，使用区分学习率 |
| > 100,000 张 | 远距离 | 全部微调；如果领域差异足够大，考虑从头训练 |

"接近 ImageNet"大致指自然场景的 RGB 照片，包含类似物体的内容。医学 CT、卫星遥感、显微镜图像属于远距离领域——特征仍有帮助，但需要让更多的层适应。

### 2.3 为什么冻结有效

ImageNet 上训练的 CNN 学到的特征并不专门针对 1,000 个类别。它们针对的是自然图像的统计特性：特定方向的边缘、纹理、对比度模式、形状原语。这些统计特性在人类能命名的几乎所有视觉领域中都是稳定的。这就是为什么一个在 ImageNet 上训练的模型，仅用一个新的线性头（不微调骨干）在 CIFAR-10 上零样本评估就能达到 80%+ 的准确率。分类头学习的是"在已学到的特征中，哪些对这个任务更重要"。

### 2.4 区分学习率

微调时，早期层应该比晚期层训练得更慢。早期层编码通用特征——你希望保留它们；晚期层编码任务特定结构——你需要让它们大幅调整。

```
典型配置：

  阶段 0（stem + 第一组）:  lr = base_lr / 100   （基本固定）
  阶段 1:                  lr = base_lr / 10
  阶段 2:                  lr = base_lr / 3
  阶段 3（最后一组）:       lr = base_lr
  分类头:                   lr = base_lr（或略高）
```

在 PyTorch 中，这只是传给优化器的一个参数组列表。一个模型，五组学习率，零额外代码。

### 2.5 BatchNorm 问题

BN 层保存了在 ImageNet 上计算的 `running_mean` 和 `running_var` 缓冲区。如果你的任务有不同的像素分布——不同的光照、不同的传感器、不同的色彩空间——这些缓冲区就是错的。三个解决方案，按优先级排列：

1. **在训练模式下微调 BN。** 让 BN 随着其他参数一起更新运行统计。当任务数据集中等大小（≥ 5,000 样本）时，这是默认选择。
2. **冻结 BN 为评估模式。** 保留 ImageNet 统计量，仅训练权重。当数据集足够小，BN 的移动平均会产生噪声时使用。
3. **用 GroupNorm 替换 BN。** 彻底消除移动平均问题。在每 GPU 批次大小很小的检测和分割骨干中使用。

搞错这一点会无声无息地让你的准确率下降 5-15%。

---

## 3. 从零实现

### 第 1 步：加载预训练骨干并检查结构

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 加载预训练的 ResNet18
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 查看结构——ResNet18 有 4 个阶段（layer1-layer4）加一个 stem 和 fc 头
print(backbone)
print(f"\n分类头: {backbone.fc}")
print(f"特征维度: {backbone.fc.in_features}")
```

```text
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(...)
  (layer2): Sequential(...)
  (layer3): Sequential(...)
  (layer4): Sequential(...)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)

分类头: Linear(in_features=512, out_features=1000, bias=True)
特征维度: 512
```

`ResNet18` 的 `fc` 层输出 1000 维（对应 ImageNet 的 1000 个类别）。迁移学习的第一步就是把它替换成你任务需要的类别数。

### 第 2 步：特征提取——冻结所有层，替换分类头

```python
def make_feature_extractor(num_classes=10):
    """构建特征提取模型：冻结骨干，仅训练新的分类头。"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 冻结所有参数——骨干网络不再接收梯度
    for p in model.parameters():
        p.requires_grad = False

    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = make_feature_extractor(num_classes=10)

# 统计可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"可训练参数: {trainable:>10,}")
print(f"冻结参数:   {frozen:>10,}")
```

```text
可训练参数:      5,130
冻结参数:   11,173,962
```

只有 `model.fc` 的 5,130 个参数可训练。骨干网络的 1,100 万参数全部冻结，作为固定的特征提取器。

### 第 3 步：区分学习率微调

```python
def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    """为 ResNet 各阶段构建差异化学习率的参数组。

    decay=0.3 意味着每个阶段的学习率是下一个阶段的 30%。
    fc 获得 base_lr，layer4 获得 0.3 * base_lr，
    conv1/bn1 获得 0.3^5 * base_lr ≈ 0.00243 * base_lr。
    听起来极端，但经验上效果很好。
    """
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

# 构建微调模型
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = True

# 打印各阶段的学习率
groups = discriminative_param_groups(model)
for g in groups:
    count = sum(p.numel() for p in g["params"])
    print(f"  {g['name']:>10s}  lr={g['lr']:.2e}  params={count:>8,}")
```

```text
   conv1_bn1  lr=2.43e-06  params=      9,408
     layer1  lr=8.10e-06  params=    215,040
     layer2  lr=2.70e-05  params=    525,568
     layer3  lr=9.00e-05  params=  1,050,624
     layer4  lr=3.00e-04  params=  2,099,200
        fc  lr=1.00e-03  params=      5,130
```

### 第 4 步：BatchNorm 处理

```python
def freeze_bn_stats(model):
    """冻结 BatchNorm 的运行统计量。

    在 model.train() 之后调用，将 BN 层切回 eval 模式，
    使用固定的 ImageNet 统计量，避免小数据集上的噪声。
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model
```

关键：在每个训练轮次的 `model.train()` 之后调用此函数。`model.train()` 会将所有层设为训练模式，此函数仅将 BN 层恢复为评估模式。

### 第 5 步：完整的微调训练循环

```python
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def fine_tune(model, train_loader, val_loader, device,
              epochs=5, base_lr=1e-3, freeze_bn=False):
    """端到端微调，使用区分学习率和余弦退火调度。"""
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  "
              f"train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model
```

### 第 6 步：逐步解冻

逐步解冻是一种从后向前、每个轮次解冻一个阶段的策略。它以额外的轮次为代价，缓解特征漂移。

```python
def progressive_unfreeze_schedule(model):
    """返回 start 和 unfreeze 两个函数。

    start() 在第一个轮次前调用，冻结所有层，仅保留分类头可训练。
    unfreeze(epoch) 在每个轮次开始时调用，解冻一个阶段。
    """
    stages = ["layer4", "layer3", "layer2", "layer1"]

    def start():
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    def unfreeze(epoch):
        if epoch < len(stages):
            name = stages[epoch]
            for n, p in model.named_parameters():
                if n.startswith(name):
                    p.requires_grad = True
            return name
        return None

    return start, unfreeze
```

调用方式：第一个轮次前调用 `start()`，每个轮次开始时调用 `unfreeze(epoch)`。每当可训练参数集合变化时，需要重建优化器——否则冻结参数的缓存动量会干扰训练。

---

## 4. 工业工具

### 4.1 torchvision 内置实现

对于大多数实际任务，三行代码就够了。

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

### 4.2 timm：800+ 预训练骨干

`timm`（PyTorch Image Models）提供了约 800 种预训练视觉骨干，API 统一。对于 torchvision 之外的任何微调任务，这是标准选择。

```python
import timm

# 一行代码加载预训练模型，替换分类头
model = timm.create_model("resnet50", pretrained=True, num_classes=10)

# 列出所有可用的 ResNet 变体
models = timm.list_models("resnet*")
print(f"可用的 ResNet 变体: {len(models)} 种")
```

### 4.3 HuggingFace Transformers

对于 Vision Transformer（ViT）系列，`transformers` 提供了与文本模型一致的加载方式。

```python
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,  # 替换分类头
    ignore_mismatched_sizes=True,
)
```

### 4.4 性能对比

| 方案 | 适用场景 | 特点 |
|---|---|---|
| torchvision | 快速实验、教学 | 内置，API 简单，模型有限 |
| timm | 生产级微调 | 800+ 模型，训练配置完整，社区活跃 |
| HuggingFace Transformers | ViT/BEiT/DeiT 微调 | 与 NLP 生态统一，预训练权重丰富 |

---

## 5. 知识连线

本课学习的迁移学习技术，是后续多个阶段的基础：

- **阶段 04 · 07（目标检测）**：YOLO 和 Faster R-CNN 的骨干网络（ResNet、CSPDarknet）都使用 ImageNet 预训练权重初始化，微调策略直接沿用本课的区分学习率方案
- **阶段 12 · 03（视觉语言模型）**：CLIP 的视觉编码器是迁移学习的典范——它在 4 亿图文对上预训练，然后通过线性探测或微调适配下游任务
- **阶段 10 · 06（数据流水线）**：本课使用 ImageNet 均值/标准差进行归一化，这个看似简单的步骤在第 10 阶段会被深入讨论——预处理流水线中的每一步都与预训练模型的假设紧密耦合

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 快速原型 | torchvision 三行代码 | 冻结骨干 + 替换 fc |
| 小数据集（< 1k） | 特征提取 + 线性探测 | 验证预训练特征是否有效 |
| 中等数据集（1k-100k） | 区分学习率微调 | 衰减因子 0.3-0.5 |
| 大数据集（> 100k） | 端到端微调 | 可以用更大学习率 |
| 边缘部署 | 微调后蒸馏到 MobileNet | 不要直接部署 ResNet50 |

### 6.2 中文场景特别建议

- 医学影像领域（CT、MRI、X 光）是中国 AI 应用的重点方向。领域距离"远"，但预训练特征仍然有效——关键是使用区分学习率并冻结 BN 统计量
- 工业质检是中国制造业的热门应用。工业图像（PCB 板、纺织品、食品）与 ImageNet 差异大，建议从 timm 加载 backbone 并使用 GroupNorm 替换 BN
- 遥感图像（高分卫星、无人机航拍）是另一个典型远距离领域。建议使用在遥感数据集上预训练的专用模型（如 SatMAE），而非通用 ImageNet 模型

### 6.3 踩坑经验

- **微调准确率低于线性探测**：这几乎总是训练 bug——学习率过高、BN 处理不当、或优化器配置错误。微调永远不应该比线性探测差，因为线性探测是微调的特例（骨干学习率为 0）
- **BatchNorm 在小数据集上静默降精度**：如果数据集 < 5,000 张，必须冻结 BN 统计量或替换为 GroupNorm。否则 BN 的运行统计量会漂移到噪声状态，准确率无声下降 5-15%
- **冻结后忘记重建优化器**：当你改变 `requires_grad` 状态后，必须重新构建优化器。旧优化器中缓存了冻结参数的动量，会导致训练不稳定
- **图像归一化不匹配**：预训练模型期望 ImageNet 的均值/标准差（`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`）。如果你的输入没有做这个归一化，准确率会直接崩塌

---

## 7. 常见错误

### 错误 1：微调学习率过高

**现象：** 训练 loss 前几个轮次剧烈震荡，最终准确率甚至低于随机猜测（10 分类中约 10%）。

**原因：** 所有层使用相同的学习率（如 `1e-3`），浅层的预训练特征被过大的梯度更新破坏。模型失去了"什么是边缘"的先验知识，后续层无法有效学习。

**修复：**

```python
# 错误写法：所有层同一学习率
optimizer = SGD(model.parameters(), lr=1e-3)

# 正确写法：使用区分学习率
groups = discriminative_param_groups(model, base_lr=1e-3)
optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4)
```

### 错误 2：小数据集上未冻结 BatchNorm

**现象：** 训练过程看似正常，loss 下降，但验证准确率比预期低 5-15%。

**原因：** BN 层的 `running_mean` 和 `running_var` 来自 ImageNet，与目标数据分布不匹配。小数据集上的移动平均计算会产生噪声统计量，导致 BN 归一化激活值时使用了错误的均值和方差。

**修复：**

```python
# 错误写法：忽略 BN 状态
model.train()

# 正确写法：训练后冻结 BN
model.train()
freeze_bn_stats(model)  # 仅将 BN 层切回 eval 模式
```

### 错误 3：改变冻结状态后未重建优化器

**现象：** 解冻层后训练不稳定，loss 出现突然跳变。

**原因：** 优化器为每个参数维护了动量缓冲区。当你冻结/解冻参数后，优化器中仍然保存着旧的动量状态，与新的可训练参数集合不匹配。

**修复：**

```python
# 解冻后必须重建优化器
for n, p in model.named_parameters():
    if n.startswith("layer3"):
        p.requires_grad = True

# 重建优化器——丢弃旧的动量缓冲区
optimizer = SGD(discriminative_param_groups(model), momentum=0.9)
```

### 错误 4：输入未做 ImageNet 归一化

**现象：** 预训练模型准确率从 ~80% 暴跌到 ~10%（随机水平）。

**原因：** 预训练模型的权重是基于归一化后的输入训练的。如果你直接输入 [0, 1] 范围的像素值，模型看到的数据分布与训练时完全不同，所有特征提取完全失效。

**修复：**

```python
# 错误写法：未归一化
return torch.from_numpy(img).permute(2, 0, 1).float()

# 正确写法：应用 ImageNet 归一化
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
img = (img - MEAN) / STD
return torch.from_numpy(img).permute(2, 0, 1).float()
```

---

## 8. 面试考点

### Q1：什么是迁移学习？为什么在计算机视觉中特别有效？（难度：⭐）

**参考答案：**

迁移学习是将在大规模数据集（如 ImageNet）上预训练的模型参数，迁移到新任务上的技术。在计算机视觉中特别有效，因为 CNN 的浅层学到的是通用视觉特征——边缘、纹理、对比度——这些特征在几乎所有视觉领域都适用，无论任务是医学影像还是卫星遥感。深层特征虽然更任务特定，但仍包含有价值的先验知识，比随机初始化收敛更快、性能更好。

### Q2：特征提取和微调应该怎么选？给出决策框架。（难度：⭐⭐）

**参考答案：**

决策框架基于三个因素：

1. **数据量**：小于 1,000 张选特征提取；1,000-10,000 张选部分微调（冻结浅层）；大于 10,000 张选端到端微调
2. **领域距离**：与 ImageNet 越远（医学、卫星），需要解冻的层越多
3. **算力预算**：特征提取只需训练分类头，计算量极小；端到端微调需要整个网络的反向传播

关键原则：特征提取是微调的特例（骨干学习率为 0），所以特征提取的准确率是微调的下界。如果微调结果低于特征提取，说明微调配置有问题。

### Q3：BatchNorm 在迁移学习中为什么是隐患？如何解决？（难度：⭐⭐）

**参考答案：**

BN 层保存了 `running_mean` 和 `running_var` 统计量，这些是在 ImageNet 上累积的。当目标域的像素分布与 ImageNet 不同时（如灰度医学影像），这些统计量就是错的。在小数据集上问题更严重——移动平均无法在短训练中充分适应新分布。

解决方案按优先级：（1）数据集 ≥ 5,000 张时，让 BN 在训练模式下随其他参数一起更新；（2）数据集更小时，冻结 BN 为 eval 模式，使用固定的 ImageNet 统计量；（3）用 GroupNorm 替换 BN，彻底消除移动平均问题。

### Q4：解释区分学习率的原理。为什么不能用统一学习率微调整个网络？（难度：⭐⭐）

**参考答案：**

CNN 的浅层学到的是通用特征（边缘、纹理），这些特征在目标任务上几乎不需要改变。深层学到的是任务特定特征，需要大幅调整。如果使用统一学习率，要么太大（破坏浅层特征），要么太小（深层适应不足）。区分学习率让每层以适合自己的速度更新——浅层用 base_lr/100 保持稳定，深层用 base_lr 快速适应新任务。这在实践中的效果显著优于统一学习率。

### Q5：你在微调后发现验证准确率只有随机水平（如 10 分类中 10%）。列出可能的原因和排查步骤。（难度：⭐⭐⭐）

**参考答案：**

可能原因和排查：

1. **输入未归一化**：检查是否应用了 ImageNet 的均值/标准差。这是最常见也最容易忽略的错误
2. **学习率过高**：检查学习率是否超过了 `1e-3`。如果所有层用 `1e-3`，浅层特征会被摧毁。尝试将基础学习率降低到 `1e-4` 并使用区分学习率
3. **BatchNorm 统计量漂移**：冻结 BN 统计量后重新训练，观察准确率是否恢复
4. **标签错误**：检查数据集的标签是否与模型预期的类别顺序一致
5. **优化器状态残留**：如果之前有过冻结/解冻操作，确保重建了优化器

关键诊断：始终同时报告线性探测准确率（冻结骨干 + 线性头）和微调准确率。如果微调 < 线性探测，问题在微调配置，不在数据或模型。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 特征提取 | "冻结骨干，训练头" | 骨干网络参数冻结，只有新的分类头接收梯度 |
| 微调 | "端到端重训练" | 所有参数可训练，但学习率远小于从头训练 |
| 区分学习率 | "浅层学习率小一点" | 优化器参数组中，早期阶段的学习率是晚期阶段的一个分数 |
| 逐层学习率衰减 | "平滑的学习率梯度" | 每层学习率乘以 decay^(L-k)；在 Transformer 微调中更常见 |
| 灾难性遗忘 | "模型忘了 ImageNet" | 过高的学习率在新任务信号被学到之前就覆盖了预训练特征 |
| BatchNorm 统计量漂移 | "运行均值不准了" | BN 的 running_mean/var 在与当前任务不同的分布上计算，无声降低准确率 |
| 线性探测 | "冻结骨干 + 线性头" | 评估预训练特征质量的方法——在冻结表示上训练最佳线性分类器的准确率 |

---

## 📚 小结

迁移学习让小数据集也能训出高性能模型——前提是正确使用预训练特征。你掌握了特征提取和微调两种策略的选择框架，学会了区分学习率的配置方法，理解了 BatchNorm 在迁移场景下的陷阱。

完整代码对比了特征提取和微调在合成数据集上的表现，展示了从加载预训练模型到训练循环的完整流程。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么在 ImageNet 上训练的 CNN 的浅层特征可以迁移到医学影像。如果医学影像全是灰度图（单通道），预训练的 RGB 权重还有用吗？写 200 字以内的说明。

2. 【实现】修改 `discriminative_param_groups` 函数，使其支持 EfficientNet 而不是 ResNet。提示：EfficientNet 的阶段名称是 `features[0]`、`features[1]`、...、`classifier`。

3. 【实验】在 `main.py` 中故意引入一个 bug：将骨干网络的学习率设为 `1e-1`（而不是分类头的 `3e-2`）。观察 loss 爆炸的现象，然后用区分学习率修复它。记录每个阶段开始发散的学习率阈值。

4. 【思考】假设你要在一个工业质检项目上做迁移学习：500 张 PCB 板缺陷图像，分为 5 类，图像来自工业相机（与 ImageNet 的自然照片差异大）。设计一个完整的迁移学习方案，包括：选择哪种策略、如何处理 BN、使用什么数据增强、训练多少轮次。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 迁移学习完整流程 | `code/main.py` | 特征提取和微调的完整对比，可直接运行 |
| 迁移学习方案规划提示词 | `outputs/prompt-transfer-learning-guide.md` | 根据数据集大小、领域距离、算力预算自动选择迁移策略 |

---

## 📖 参考资料

1. [论文] Yosinski et al. "How transferable are features in deep neural networks?". NeurIPS, 2014. https://arxiv.org/abs/1411.1792
2. [论文] Howard & Ruder. "Universal Language Model Fine-tuning for Text Classification". ACL, 2018. https://arxiv.org/abs/1801.06146
3. [论文] Kornblith et al. "Similarity of Neural Network Representations Revisited". ICML, 2019. https://arxiv.org/abs/1905.00442
4. [官方文档] PyTorch. "torchvision.models". https://pytorch.org/vision/stable/models.html
5. [官方文档] Hugging Face. "timm". https://huggingface.co/docs/timm
6. [GitHub] Ross Wightman. "pytorch-image-models". https://github.com/huggingface/pytorch-image-models

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、工程最佳实践、常见错误、面试考点等均为原创内容。
