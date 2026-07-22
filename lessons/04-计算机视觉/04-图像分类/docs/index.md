# 图像分类：从像素到概率分布

> 分类器就是一个从像素到概率分布的函数。剩下的都是流水线工程。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 02 · 09（模型评估）、阶段 03 · 04（卷积神经网络）、阶段 03 · 05（损失函数）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 04 · 05（目标检测）—— 目标检测的每个区域本质上都在做图像分类

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建端到端的图像分类流水线：从数据加载到模型评估，每一步都可以独立调试
- [ ] 解释训练循环中五个关键不变量——忘记任何一个都会产生静默错误，且损失曲线看起来"合理"
- [ ] 实现 Mixup、Cutout 和标签平滑，说明各自在什么场景下值得使用
- [ ] 读取混淆矩阵，定位模型的具体失败模式——哪些类别被混淆、为什么会混淆、下一步该做什么
- [ ] 对比不同数据增强策略的效果，用实验数据支撑选择

---

## 1. 问题

每一个"高大上"的视觉任务，底层都在做图像分类。

目标检测？在候选区域上分类"这个框里是什么"。图像分割？在每个像素上分类"这个像素属于哪个物体"。图像检索？计算图片与各类别中心的距离然后排序。搞不定分类，后面的一切都建立在流沙之上。

**但分类的 bug 大多不在模型里。** 它们藏在流水线中：

- 归一化用错了统计量——ImageNet 的均值/标准差套到 CIFAR-10 上，准确率偷偷丢 1%，没人知道
- 训练数据没打乱——模型按类别顺序学习，梯度严重偏置
- 数据增强改了标签——旋转 180 度让 "6" 变成 "9"，但标签还是 "6"
- 验证集被训练数据污染——数据泄露导致验证准确率虚高 15%
- 忘记调用 `model.train()` / `model.eval()`——BatchNorm 在训练和评估时行为不同，不切换就会得到错误结果

一个正确的 CNN 在 CIFAR-10 上能达到 93% 准确率，但流水线中有上述任何一个 bug，同样的模型只会拿到 70-75%，而且损失曲线在整个训练过程中看起来都很"正常"。

**这节课的目的，就是把整个流水线从头搭起来，让每一部分都可以被检查和调试。**

---

## 2. 概念

### 2.1 分类流水线全景

```
┌─────────────────────────────────────────────────────────────┐
│                    图像分类训练流水线                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始图像                                                    │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────┐    训练时增强，验证时不增强                  │
│  │  数据增强     │    随机裁剪、翻转、Cutout、颜色抖动         │
│  └──────────────┘                                           │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────┐    用训练集统计量标准化                     │
│  │  归一化       │    (img - mean) / std                     │
│  └──────────────┘                                           │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────┐    分批加载 + 打乱 + 多进程预取             │
│  │  DataLoader   │                                          │
│  └──────────────┘                                           │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────┐    CNN / ResNet / ViT                     │
│  │  模型         │                                          │
│  └──────────────┘                                           │
│     │                                                       │
│     ├─── logits (原始输出) ──→ argmax ──→ 预测类别（推理时）  │
│     │                                                       │
│     └─── logits ──→ 交叉熵损失 ──→ 反向传播 ──→ 参数更新    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

这条链路中的每一个环节都是一个潜在的 bug 来源。接下来逐一理解每个环节的作用。

### 2.2 交叉熵、Logits 与 Softmax

分类器对每张图像输出 C 个数字，称为 **logits**（原始分数）。Softmax 将其转化为概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

交叉熵衡量正确类别的负对数概率：

$$
\text{CE}(z, y) = -\log(\text{softmax}(z)_y) = -z_y + \log\left(\sum_{j=1}^{C} e^{z_j}\right)
$$

右边的形式是数值稳定的实现方式（log-sum-exp 技巧）。PyTorch 的 `nn.CrossEntropyLoss` 融合了 softmax + NLL 两个操作，直接接受原始 logits。

**这是一个常见的陷阱：** 如果你先对 logits 调用 softmax，再传入 CrossEntropyLoss，实际上计算的是 `log(softmax(softmax(z)))`——这个值在数学上没有意义，梯度会在几轮训练后趋近于零。

### 2.3 数据增强为什么有效

CNN 通过权重共享天然具有一定的平移不变性，但**没有**以下不变性：

```
原始图像: "朝左的狗"
水平翻转: "朝右的狗"          ← 标签相同，像素完全不同
随机裁剪: "狗，局部可见"      ← 标签相同，像素完全不同
颜色抖动: "暖光下的狗"        ← 标签相同，像素完全不同
随机擦除: "狗，部分遮挡"      ← 标签相同，像素完全不同
```

每一次随机变换都是在告诉模型："这两种图像有相同的标签，学习那些与变换无关的特征。"这本质上是通过数据的方式为模型注入归纳偏置——告诉你"什么不变、什么变化"。

**关键原则：** 增强必须保持标签不变。对数字 6 旋转 180 度会让它变成 9——所以数字分类器通常只用小角度旋转。

### 2.4 Mixup 与标签平滑

**Mixup** 混合两个样本的像素和标签：

$$
\begin{aligned}
\lambda &\sim \text{Beta}(\alpha, \alpha) \\
x &= \lambda \cdot x_i + (1 - \lambda) \cdot x_j \\
y &= \lambda \cdot y_i + (1 - \lambda) \cdot y_j
\end{aligned}
$$

为什么有效？标准训练用独热标签 `[0, 0, 1, 0, 0]`，模型被迫在每个训练点输出任意尖锐的结果——这是过拟合和校准差的根源。Mixup 迫使模型在类别之间学习平滑的插值，训练损失会上升，但测试准确率和校准都会改善。

**标签平滑**是 Mixup 的"近亲"：

$$
y_{\text{smooth}} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{C} & \text{正确类别} \\ \frac{\epsilon}{C} & \text{其他类别} \end{cases}
$$

当 $\epsilon = 0.1$、$C = 10$ 时，目标从 `[0, 0, 1, 0, ...]` 变成 `[0.01, 0.01, 0.91, 0.01, ...]`。PyTorch 1.10+ 内置支持：`nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 2.5 超越准确率的评估

聚合准确率在类别不平衡时具有欺骗性。一个在 90:10 不平衡数据上永远预测多数类的无用模型也能得到 90% 准确率。

真正有用的评估工具：

| 评估维度 | 含义 | 用途 |
|---|---|---|
| **逐类准确率** | 每个类别单独的准确率 | 立刻发现表现差的类别 |
| **混淆矩阵** | C x C 表格，(i, j) = 真实类别 i 被预测为 j 的数量 | 精确定位哪两类被混淆 |
| **Top-1 准确率** | 最高概率的预测是否正确 | 最严格的标准 |
| **Top-5 准确率** | 前 5 个预测中是否包含正确类别 | 适用于细粒度分类（如 ImageNet） |
| **F1 分数** | 精确率和召回率的调和平均 | 类别不平衡时的核心指标 |

混淆矩阵的读法：行是真实类别，列是预测类别。对角线是正确分类，非对角线是错误。如果某个非对角线位置（i, j）的值很大，说明类别 i 经常被误分为类别 j——这就是你需要针对性改进的地方。

---

## 3. 从零实现

完整代码见 `code/main.py`，以下分步讲解核心逻辑。

### 第 1 步：合成数据集

使用合成数据而非真实数据集，好处是可立即运行（无需下载）、可复现（固定种子）、速度快。合成数据具有与真实 CIFAR-10 相同的格式（32x32 RGB），换真实数据时流水线无需修改。

```python
import numpy as np
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=300, num_classes=10, seed=0):
    """每个类别有独特的颜色频率模式，加入高斯噪声。"""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(
                np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
            )
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)  # 噪声防止记忆
            img = np.clip(img, 0, 1).astype(np.float32)
            images.append(img)
            labels.append(c)
    return np.stack(images), np.array(labels)
```

每个类别使用不同的 `freq`（空间频率）和 `centre`（颜色中心），模型必须学到这些模式才能分类。噪声迫使模型学习可泛化的特征，而不是记住单个像素值。

### 第 2 步：数据增强

所有变换在 numpy 上操作，不依赖 torchvision——这样可以清楚地看到每个变换在做什么。

```python
def random_crop(pad=4):
    """反射填充 → 随机裁剪。反射填充比零填充更自然。"""
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad + 1)
        x = np.random.randint(0, 2 * pad + 1)
        return padded[y:y + h, x:x + w, :]
    return _fn


def cutout(size=8):
    """随机遮挡一个 size x size 的正方形区域。"""
    def _fn(img):
        h, w = img.shape[:2]
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, y - size // 2), min(h, y + size // 2)
        x1, x2 = max(0, x - size // 2), min(w, x + size // 2)
        img = img.copy()
        img[y1:y2, x1:x2, :] = 0.0
        return img
    return _fn
```

**为什么用反射填充而非零填充：** 零填充在裁剪后留下黑色边框，模型可以学到"角落像素=黑色"这个无用的模式来判断增强参数——这不是我们想要的。反射填充使用边缘像素的镜像，裁剪后的图像看起来仍然自然。

### 第 3 步：Mixup

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    """对一个批次进行 Mixup：线性插值图像和软标签。"""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y_onehot = F.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    """对软标签计算交叉熵。"""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

Mixup 在批次内部混合两个样本。`soft_cross_entropy` 是标准交叉熵的推广——当目标是独热向量时退化为标准形式。

### 第 4 步：训练循环的五条铁律

这五个不变量覆盖了训练循环中最常见的静默 bug：

```python
def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()  # 铁律 1: 切换到训练模式（BatchNorm/Dropout）
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()          # 铁律 2: 每个批次前清零梯度
        logits = model(x)              # 铁律 3: 不调 softmax
        loss = F.cross_entropy(logits, y)  # 铁律 3: 接收原始 logits
        loss.backward()
        optimizer.step()
```

```python
@torch.no_grad()                       # 铁律 4: 关闭自动求导
def evaluate(model, loader, device, num_classes):
    model.eval()                       # 铁律 5: 切换到评估模式
    for x, y in loader:
        logits = model(x)              # argmax 即可，不需要 softmax
        pred = logits.argmax(dim=-1)
```

每条铁律对应的后果：

| 铁律 | 违反后果 |
|---|---|
| `model.train()` | BatchNorm 用全局均值而非批次统计量，Dropout 不工作 |
| `zero_grad()` | 梯度累积导致更新方向错误，训练不收敛 |
| 传 logits 而非 softmax | `log(softmax(softmax(z)))` 梯度趋零，loss 曲线平缓但无学习 |
| `@torch.no_grad()` | 计算图驻留显存，OOM 或隐式参数更新 |
| `model.eval()` | BatchNorm 和 Dropout 在推理时行为异常 |

### 第 5 步：混淆矩阵分析

```python
def print_confusion_matrix(confusion, class_names=None):
    """打印混淆矩阵和逐类精确率/召回率/F1。"""
    c = confusion.shape[0]
    # 打印矩阵
    header = f"{'真实\\预测':>8}" + "".join(f"{n:>6}" for n in class_names)
    print(header)
    for i in range(c):
        print(f"{class_names[i]:>8}" + "".join(f"{v:>6}" for v in confusion[i].tolist()))
    # 计算逐类指标
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    for i in range(c):
        print(f"{class_names[i]:>8}  P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}")
```

行是真实类别，列是预测类别。对角线是正确的，非对角线是错误的。如果某个位置 (i, j) 的值很大，说明类别 i 经常被误分为类别 j。

---

## 4. 工业工具

### 4.1 torchvision：标准数据集与变换

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

# CIFAR-10 的官方均值和标准差（从训练集计算）
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),  # 反射填充
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True, download=True, transform=train_tf)
val_ds = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

两个容易出错的地方：
- **均值/标准差是数据集特定的**——CIFAR-10 的统计量与 ImageNet 不同。复制粘贴 ImageNet 的统计量是约 1% 的准确率泄漏。
- **训练集和验证集使用相同的标准化参数**——从训练集计算，应用到验证集。

### 4.2 预训练模型迁移学习

当数据量不大（< 10K）时，从预训练模型微调通常比从零训练效果好很多。

```python
from torchvision import models

# 使用预训练的 ResNet-18，替换最后的全连接层
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 只微调最后几层，冻结前面的特征提取器
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
```

### 4.3 PyTorch Lightning：简化训练循环

PyTorch Lightning 将训练循环中的重复代码封装起来，让你专注于模型逻辑：

```python
import pytorch_lightning as pl

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = MiniClassifier(num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.1, momentum=0.9)
```

### 4.4 性能对比

| 工具 | 上手难度 | 灵活性 | 适用场景 |
|---|---|---|---|
| 纯 PyTorch | 低 | 极高 | 学习理解、研究 |
| torchvision 工具 | 低 | 中 | 快速原型、标准数据集 |
| PyTorch Lightning | 中 | 中高 | 团队项目、需要快速迭代 |
| HuggingFace Trainer | 中 | 中 | NLP/多模态任务 |
| MXNet/GluonCV | 中 | 中 | 已有 MXNet 基础的项目 |

---

## 5. 知识连线

本课学习的图像分类技能，是后续多个阶段的基础：

- **阶段 04 · 05（目标检测）**：目标检测的区域分类头就是一个图像分类器。理解了数据增强、损失函数和评估指标的完整逻辑，你就能理解为什么 Faster R-CNN 的训练需要分阶段——先训练 RPN，再训练分类头
- **阶段 04 · 06（语义分割）**：分割网络的每个像素预测本质上是一个分类问题。混淆矩阵的分析方法同样适用于评估分割模型对不同物体的分类效果
- **阶段 07（Transformer 深入）**：Vision Transformer (ViT) 将图像切成 patch 再做分类——本课的训练流水线（数据加载、增强、损失、评估）同样适用于 ViT，只是模型架构不同
- **阶段 10（大语言模型从零）**：多模态模型（如 LLaVA）的视觉编码器通常就是预训练的图像分类模型。理解分类器的特征提取能力，就能理解多模态模型"看到"了什么

---

## 6. 工程最佳实践

### 6.1 数据集特定配置

不同数据集需要不同的预处理和增强策略：

| 数据集 | 图像尺寸 | 均值 | 标准差 | 推荐增强 |
|---|---|---|---|---|
| MNIST | 28x28 灰度 | 0.1307 | 0.3081 | 轻微裁剪（pad=2） |
| Fashion-MNIST | 28x28 灰度 | 0.2860 | 0.3530 | 裁剪 + 轻微旋转 |
| CIFAR-10 | 32x32 RGB | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) | 翻转 + 裁剪 + Cutout |
| CIFAR-100 | 32x32 RGB | (0.5071, 0.4867, 0.4408) | (0.2675, 0.2565, 0.2761) | 翻转 + 裁剪 + Cutout + Mixup |
| ImageNet | 224x224 RGB | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | RandAugment + Mixup + CutMix |

### 6.2 中文场景建议

- 中文 OCR 数据集（如 CASIA-HWDB）通常是灰度图像，预处理策略更接近 MNIST 而非 CIFAR-10
- 中文验证码识别可能需要对字符做弹性形变增强
- 工业缺陷检测中，正样本（正常产品）远多于负样本（缺陷产品），需要过采样或加权损失
- 遥感场景的图像分类需要考虑大尺寸图像的滑窗切片策略

### 6.3 踩坑经验

- **冻结学习率过低**：SGD 的 lr=0.01 对于从零训练太保守，CIFAR-10 建议从 0.1 开始
- **验证集太小**：验证集少于 1000 张时，准确率波动可能掩盖真实的性能差异
- **eval 时还在用 train 的增强**：验证集只做标准化，不做任何随机增强
- **混淆矩阵不按行归一化**：未归一化的混淆矩阵在类别数量不同时难以比较，务必同时查看原始计数和归一化比例

---

## 7. 常见错误

### 错误 1：交叉熵前多做了一次 Softmax

**现象：** 训练 loss 前几轮有微小下降，然后停滞。准确率一直在 10% 左右（10 类数据集的随机基线）。

**原因：** `nn.CrossEntropyLoss` 内部融合了 `log_softmax + NLL`。在外部先做 softmax 相当于计算 `log(softmax(softmax(z)))`，梯度极小。

**修复：**

```python
# ❌ 错误
logits = model(x)
loss = F.cross_entropy(F.softmax(logits, dim=-1), y)

# ✓ 正确：直接传入原始 logits
logits = model(x)
loss = F.cross_entropy(logits, y)
```

### 错误 2：忘记切换 train/eval 模式

**现象：** 验证准确率和训练准确率交替波动，看起来像过拟合但实际不是。BatchNorm 在 eval 模式下应该用移动均值/方差，如果留在 train 模式则用批次统计量，结果不可复现。

**修复：**

```python
# ❌ 遗漏
def evaluate(model, loader):
    for x, y in loader:
        logits = model(x)  # BatchNorm 用批次统计量 → 结果随机

# ✓ 正确
@torch.no_grad()
def evaluate(model, loader):
    model.eval()  # 切换到评估模式
    for x, y in loader:
        logits = model(x)  # BatchNorm 用移动均值 → 确定性结果
```

### 错误 3：验证集使用 ImageNet 统计量

**现象：** 在 CIFAR-10 上准确率始终比论文低 1-2%，无法复现。

**原因：** ImageNet 的均值 `(0.485, 0.456, 0.406)` 和标准差 `(0.229, 0.224, 0.225)` 与 CIFAR-10 不同。标准化的输入分布偏移导致模型表现不佳。

**修复：** 始终使用目标数据集自身的统计量。用 `torchvision` 加载时自动计算，手动构建时从训练集逐通道统计：

```python
# 正确做法：从训练集计算
mean = train_images.mean(axis=(0, 1, 2))  # 逐通道均值
std = train_images.std(axis=(0, 1, 2))    # 逐通道标准差
```

### 错误 4：零填充替代反射填充

**现象：** 模型在验证集上准确率正常，但对边缘区域的预测不稳定。

**原因：** 零填充（`padding_mode='zeros'`）在裁剪后留下黑色边框，模型学到"角落是黑色的"这个增强副作用，而非真正的图像特征。

**修复：**

```python
# ❌ 零填充
RandomCrop(32, padding=4, padding_mode='zeros')

# ✓ 反射填充
RandomCrop(32, padding=4, padding_mode='reflect')
```

### 错误 5：Mixup 后训练准确率虚高

**现象：** 启用 Mixup 后训练准确率似乎更高，但验证准确率没有相应提升。

**原因：** Mixup 后训练集上的标签是软标签，用原始标签 y 计算训练准确率时，模型可能在混合样本上表现好但对原始样本的 argmax 不准。训练准确率只是近似信号，以验证集为准。

**修复：** 这不是 bug，而是指标的含义变了。启用 Mixup 后，不要把训练准确率作为判断依据，只看验证准确率。

---

## 8. 面试考点

### Q1：解释训练循环中为什么不能先 softmax 再传入 CrossEntropyLoss。（难度：⭐）

**参考答案：**

PyTorch 的 `nn.CrossEntropyLoss` 内部融合了 `log_softmax + NLL` 两个操作，直接接受原始 logits 输入，这是数值上更稳定的做法。如果先做 softmax 再传入，实际计算的是 `log(softmax(softmax(z)))`——双重 softmax 会将所有输出值压缩到接近均匀的范围，梯度趋近于零，模型无法学习。

正确的做法是直接将模型输出的 logits 传入交叉熵损失，评估时用 argmax 取预测类别（argmax 在 logits 上和 softmax 后的结果完全相同，无需额外计算）。

### Q2：为什么数据增强能提升测试准确率但不是过拟合的反面？（难度：⭐⭐）

**参考答案：**

数据增强的原理是通过扩充训练集来教会模型不变性——同一个对象的翻转、裁剪、颜色变化版本应该被预测为同一类别。这本质上是在注入归纳偏置，告诉模型"哪些变换不影响语义"。

它不是"过拟合的反面"，而是让模型学到更鲁棒的特征。正则化（如 Dropout、权重衰减）是限制模型复杂度，数据增强是扩展训练数据的多样性——两者目标不同，可以同时使用。实际上，Mixup 和 Cutout 更接近正则化，因为它们的标签变成了软标签，降低了模型对训练点的"记忆"。

### Q3：如何从混淆矩阵中诊断数据标注问题？（难度：⭐⭐）

**参考答案：**

当混淆矩阵中某两个类别 i 和 j 出现严重的双向混淆（i 被大量误分为 j，j 也被大量误分为 i）时，有三种可能：

1. **两个类别视觉上确实相似**（如猫和狗的幼崽）——需要收集更多有区分性的样本或设计针对性的增强
2. **标签噪声**——部分样本被错误标注，需要人工审核被混淆的样本
3. **类别定义模糊**——两个类别之间的边界本身就不清楚，需要重新定义类别

诊断步骤：首先检查精确率和召回率的差值——如果某个类别的精确率和召回率差异很大（> 0.2），很可能是标签噪声。然后随机抽样被混淆的图像进行人工检查。

### Q4：对比 MNIST、CIFAR-10 和 Fashion-MNIST 的分类难度差异。（难度：⭐⭐）

**参考答案：**

| 数据集 | 难度 | 核心挑战 |
|---|---|---|
| MNIST | 最简单 | 灰度、背景干净、数字变形有限，简单 CNN 即可达 99%+ |
| Fashion-MNIST | 中等 | 灰度但类别间视觉差异更小（T-shirt vs Shirt），需要更细粒度的特征 |
| CIFAR-10 | 最难 | 彩色自然图像、背景复杂、物体姿态和光照变化大 |

MNIST 的简单性让很多模型都能达到 99%+，这导致它作为基准测试的价值有限。CIFAR-10 的挑战来自三个方面：像素数量更多（32x32 vs 28x28）、信息通道更多（RGB vs 灰度）、物体类别间的视觉差异更大。

### Q5：如果训练 10 轮后验证准确率不再提升，如何判断是过拟合还是欠拟合？（难度：⭐⭐⭐）

**参考答案：**

**过拟合特征：** 训练准确率持续上升（接近 100%），验证准确率停滞或下降，训练-验证差距越来越大。

**欠拟合特征：** 训练准确率也不高（可能低于 80%），验证准确率与训练准确率接近或略低，两条曲线都在"挣扎"。

**诊断方法：**
- 如果训练 loss 还在下降但验证 loss 开始上升 → 过拟合，加正则化（Dropout、权重衰减、更多数据增强）
- 如果训练 loss 也不再下降 → 欠拟合，增加模型容量（更多层/通道）或增大初始学习率
- 如果两者都不下降 → 学习率可能太高或太低，检查学习率调度器

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Logits | "模型的原始输出" | 预 Softmax 的 C 维向量，每个值代表对应类别的原始分数；交叉熵直接接受 logits，不需要先做 Softmax |
| 交叉熵损失 | "分类用的损失函数" | 正确类别的负对数概率；融合了 log_softmax 和 NLL，数值稳定 |
| 数据增强 | "随机变换图像" | 训练时对输入施加的、保持标签不变的像素级变换；教会 CNN 那些它没有原生不变性的变换 |
| Mixup | "混合两张图" | 对两个随机样本的像素和标签做线性插值，迫使分类器学习平滑插值而非尖锐决策边界 |
| 标签平滑 | "更柔和的目标" | 用 (1-eps, eps/C, ...) 替代独热标签 [0, 0, 1, 0, ...]；改善校准，轻微提升准确率 |
| 混淆矩阵 | "哪里犯了错" | C x C 表格，(i, j) 记录真实类别 i 被预测为 j 的数量；对角线正确，非对角线是错误 |
| Top-k 准确率 | "前 k 个对不对" | 正确类别是否在前 k 个预测中；Top-5 适用于细粒度分类（如 ImageNet 1000 类） |
| DataLoader | "数据加载器" | 封装数据集的打乱、分批和多进程加载；训练流水线中最常见的 bug 来源之一 |
| BatchNorm | "批归一化" | 训练时用批次统计量，评估时用移动均值/方差；忘记切换模式会导致不可复现的结果 |
| 标准化 | "让数据归一化" | 按通道减均值除标准差；让所有特征在相似量级上，加速收敛，使用训练集的统计量 |

---

## 📚 小结

图像分类是一个从像素到概率分布的函数——理解了交叉熵如何处理 logits、数据增强如何注入不变性、混淆矩阵如何定位失败模式，你就掌握了所有视觉任务的基础。本课从零实现了完整的训练流水线，包括合成数据集、增强变换、Mixup、训练循环和混淆矩阵分析。

下一课我们将进入目标检测——从分类单张图像到同时定位和识别图像中的多个物体。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 `nn.CrossEntropyLoss` 不能接受 Softmax 的输出。写 200 字以内的说明，让一个有 Python 基础但没学过 ML 的程序员也能听懂。

2. **【实现】** 修改 `cutout` 函数，支持随机宽高比的矩形遮挡（而非正方形），并通过参数控制最大遮挡面积不超过图像面积的 25%。

3. **【实验】** 在合成数据集上运行三组对比实验（无增强 / 翻转+裁剪 / 翻转+裁剪+Cutout），每组训练 5 轮。记录验证准确率并画出训练损失曲线的对比图。解释为什么无增强的训练损失更低但验证准确率不更高。

4. **【思考】** Mixup 在自然图像（如狗/猫分类）上通常能提升准确率，但在 MNIST 上效果不明显。思考可能的原因。（提示：MNIST 的类别之间本来就比较"干净"，每个数字的变体有限。）

5. **【挑战】** 实现 CutMix：随机裁剪一个矩形区域从图像 A 粘贴到图像 B 上，标签按面积比例混合。与 Mixup 对比效果。（参考论文：[CutMix: Regularization Strategy](https://arxiv.org/abs/1905.04899)）

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 图像分类完整流水线 | `code/main.py` | 可直接运行，涵盖合成数据集、Fashion-MNIST、CIFAR-10 三种数据加载方式，支持 Mixup 和增强消融实验 |
| 分类流水线审计提示词 | `outputs/prompt-image-classification-guide.md` | 输入训练脚本，逐行检查五个训练不变量，输出结构化审计报告 |

---

## 📖 参考资料

1. [论文] He et al. "Deep Residual Learning for Image Recognition". CVPR, 2016. https://arxiv.org/abs/1512.03385
2. [论文] Zhang et al. "mixup: Beyond Empirical Risk Minimization". ICLR, 2018. https://arxiv.org/abs/1710.09412
3. [论文] Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features". ICCV, 2019. https://arxiv.org/abs/1905.04899
4. [论文] He et al. "Bag of Tricks for Image Classification with Convolutional Neural Networks". CVPR, 2019. https://arxiv.org/abs/1812.01187
5. [论文] Guo et al. "On Calibration of Modern Neural Networks". ICML, 2017. https://arxiv.org/abs/1706.04599
6. [官方文档] PyTorch `nn.CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
7. [官方文档] PyTorch `torchvision.datasets.CIFAR10`: https://pytorch.org/docs/stable/torchvision/datasets.html#cifar10
8. [课程] CS231n: Training Neural Networks. https://cs231n.github.io/neural-networks-3/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
