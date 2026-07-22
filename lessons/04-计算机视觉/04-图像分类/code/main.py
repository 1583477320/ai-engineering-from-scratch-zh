# main.py — 图像分类完整流水线
# 依赖：torch>=2.0, torchvision>=0.15, numpy>=1.24, scikit-learn>=1.2
# 安装：pip install torch torchvision numpy scikit-learn
# 对应课程：第 04 阶段 · 04（图像分类）
#
# 本文件实现了一个完整的图像分类训练流水线：
#   数据加载 -> 数据增强 -> 模型定义 -> 训练循环 -> 评估 -> 混淆矩阵分析
# 默认在 Fashion-MNIST 数据集上运行，也可切换为 CIFAR-10

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


# ====================================================================
# 常量定义
# ====================================================================

# Fashion-MNIST 的 10 个类别标签
FASHION_MNIST_CLASSES = [
    "T恤/上衣", "裤子", "套头衫", "连衣裙", "外套",
    "凉鞋", "衬衫", "运动鞋", "包", "短靴",
]

CIFAR10_CLASSES = [
    "飞机", "猫", "卡车", "鸟", "鹿",
    "狗", "青蛙", "马", "船", "鲸鱼",
]


# ====================================================================
# 第 1 步：合成数据集（用于快速验证，无需网络下载）
# ====================================================================

def synthetic_dataset(num_per_class=300, num_classes=10, seed=0):
    """生成类 CIFAR-10 格式的合成数据集（32x32 RGB）。

    每个类别使用不同的颜色频率和基础色调，加入高斯噪声迫使模型
    学习可泛化的特征，而非记忆单个像素值。
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for c in range(num_classes):
        colour_centre = rng.uniform(0, 1, (3,))
        spatial_freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(
                np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij"
            )
            r = np.sin(xx * spatial_freq) * 0.5 + colour_centre[0]
            g = np.cos(yy * spatial_freq) * 0.5 + colour_centre[1]
            b = (xx + yy) * 0.5 * colour_centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1).astype(np.float32)
            images.append(img)
            labels.append(c)
    images = np.stack(images)
    labels = np.array(labels)
    perm = rng.permutation(len(images))
    return images[perm], labels[perm]


class ArrayDataset(torch.utils.data.Dataset):
    """将 numpy 数组封装为 PyTorch Dataset，支持可选的变换。

    从 torchvision 获取的数据集自带 transform 机制，这个
    类主要用于合成数据集的快速原型验证。
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        # HWC -> CHW，PyTorch 期望通道在前
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return img, int(self.labels[idx])


# ====================================================================
# 第 2 步：Fashion-MNIST / CIFAR-10 数据加载器
# ====================================================================

def load_fashion_mnist(data_dir="./data"):
    """加载 Fashion-MNIST 数据集（28x28 灰度图，10 类服装）。

    返回训练集和验证集加载器。使用标准的随机裁剪 + 水平翻转增强。
    """
    from torchvision import datasets, transforms

    mean, std = (0.2861,), (0.3530,)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
    val_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def load_fashion_mnist_no_augment(data_dir="./data"):
    """Fashion-MNIST 无数据增强版本，用于消融实验对比。"""
    from torchvision import datasets, transforms

    mean, std = (0.2861,), (0.3530,)
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=t)
    val_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=t)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def load_cifar10(data_dir="./data"):
    """加载 CIFAR-10 数据集（32x32 彩色图，10 类物体）。

    使用 ImageNet 标准的均值/标准差进行标准化。
    reflect 填充是社区默认做法——零填充会引入黑色边框这个虚假信号。
    """
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ====================================================================
# 第 3 步：数据增强工具函数
# ====================================================================

def standardize(mean, std):
    """按通道标准化变换 (numpy 版)。"""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    """以概率 p 执行随机水平翻转。"""
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    """反射填充后随机裁剪——比零填充更自然。"""
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad + 1)
        x = np.random.randint(0, 2 * pad + 1)
        return padded[y:y + h, x:x + w, :]
    return _fn


def cutout(size=8):
    """Cutout 正则化：随机遮挡一个 size x size 的正方形区域。

    迫使模型关注全局特征，而不依赖某个局部区域做判断。
    """
    def _fn(img):
        h, w = img.shape[:2]
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        y1, y2 = max(0, cy - size // 2), min(h, cy + size // 2)
        x1, x2 = max(0, cx - size // 2), min(w, cx + size // 2)
        img = img.copy()
        img[y1:y2, x1:x2, :] = 0.0
        return img
    return _fn


def compose(*fns):
    """依次组合多个变换函数。"""
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn


# ====================================================================
# 第 4 步：Mixup 与软标签交叉熵
# ====================================================================

def mixup_batch(x, y, num_classes, alpha=0.2):
    """对一个批次执行 Mixup。

    从 Beta(alpha, alpha) 采样 lambda，对两个随机样本做线性插值。
    标签也做同样的混合，产生软目标分布。
    """
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]

    y_onehot = F.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    """对软标签计算交叉熵损失。

    当目标是独热向量时，退化为标准交叉熵。
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


# ====================================================================
# 第 5 步：模型定义
# ====================================================================

class MiniClassifier(nn.Module):
    """轻量级 CNN 分类器，适用于 32x32 合成数据。

    架构灵感来自 ResNet 设计模式：卷积块 + BN + ReLU + MaxPool。
    参数量约 30 万，在合成数据集上 5 轮即可接近 100% 准确率。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 32x32 -> 16x16

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 16x16 -> 8x8

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 -> 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class FashionCNN(nn.Module):
    """面向 Fashion-MNIST 的轻量级卷积网络（类似 LeNet 变体）。

    输入: (B, 1, 28, 28)，输出: (B, 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(dataset_name, num_classes=10, device="cpu"):
    """根据数据集选择对应的模型并移动到指定设备。"""
    if dataset_name == "fashion-mnist":
        model = FashionCNN(num_classes=num_classes).to(device)
    elif dataset_name == "cifar-10":
        model = MiniClassifier(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return model


# ====================================================================
# 第 6 步：训练与评估核心函数
# ====================================================================

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    """执行一个轮次的训练。

    训练循环五条铁律：
      1. model.train() 启用 Dropout/BatchNorm 的训练模式
      2. optimizer.zero_grad() 在 backward 之前清零梯度
      3. loss.item() 获取标量值，防止计算图驻留显存
      4. @torch.no_grad() 评估时关闭自动求导
      5. argmax 直接作用于 logits，无需先做 softmax
    """
    model.train()
    total_correct = 0
    total_samples = 0
    loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            x_mixed, y_soft = mixup_batch(images, labels, num_classes)
            logits = model(x_mixed)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        total_samples += images.size(0)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()

    return loss_sum / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, num_classes=10):
    """在验证集上评估，返回损失、Top-1 准确率和混淆矩阵。"""
    model.eval()
    total_correct = 0
    total_samples = 0
    loss_sum = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss_sum += loss.item() * images.size(0)
        total_samples += images.size(0)

        pred = logits.argmax(dim=-1)
        total_correct += (pred == labels).sum().item()

        # 逐样本更新混淆矩阵
        for true_lbl, pred_lbl in zip(labels.cpu(), pred.cpu()):
            confusion_matrix[int(true_lbl), int(pred_lbl)] += 1

    return loss_sum / total_samples, total_correct / total_samples, confusion_matrix


@torch.no_grad()
def compute_top_k_accuracy(model, loader, device, k=5, num_classes=10):
    """计算 Top-k 准确率：正确类别是否在前 k 个预测中。

    对于类别数多的数据集（如 ImageNet），Top-5 比 Top-1 更有信息量。
    Fashion-MNIST 只有 10 类，Top-5 几乎等于 Top-1。
    """
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        batch_size = images.size(0)
        total += batch_size

        _, top_k_indices = logits.topk(k, dim=-1, largest=True, sorted=True)
        expanded_labels = labels.unsqueeze(1).expand_as(top_k_indices)
        correct += (top_k_indices == expanded_labels).sum().item()

    return correct / total


# ====================================================================
# 第 7 步：混淆矩阵与逐类报告
# ====================================================================

def print_confusion_matrix(confusion, class_names=None):
    """打印可读的混淆矩阵与逐类精确率 / 召回率 / F1 分数。

    混淆矩阵的行是真实标签，列是预测标签。对角线元素是正确分类的数量，
    非对角线元素告诉你是哪两类被混淆了——这正是需要改进的地方。
    """
    c = confusion.shape[0]
    names = class_names or [str(i) for i in range(c)]

    # 表头
    label_col = "真实\\预测"
    header_str = f"{label_col:>8}"
    for name in names:
        header_str += f"{name[:4]:>6}"
    print(f"\n混淆矩阵")
    print(header_str)
    print("-" * len(header_str))

    # 各行
    for i in range(c):
        row_str = f"{names[i][:4]:>8}"
        for j in range(c):
            row_str += f"{confusion[i, j]:>6}"
        print(row_str)

    # 逐类指标
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)

    macro_p = precision.mean().item()
    macro_r = recall.mean().item()
    macro_f1 = f1.mean().item()

    print()
    print(f"逐类报告")
    print(f"{'类别':>8} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print("-" * 36)
    for i in range(c):
        print(f"{names[i]:>8} {precision[i]:>8.3f} {recall[i]:>8.3f} {f1[i]:>8.3f}")
    print("-" * 36)
    print(f"宏平均: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}")

    return precision, recall, f1


# ====================================================================
# 第 8 步：数据增强消融实验
# ====================================================================

def run_augmentation_ablation(num_classes=10, num_per_class=300, epochs=5):
    """对比三种数据增强策略的效果。

    实验组 A / B / C：
      A. 无增强（基线）
      B. 随机裁剪 + 水平翻转
      C. B + Cutout

    固定随机种子，同一模型架构，相同训练轮次，确保公平比较。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = synthetic_dataset(num_per_class=num_per_class, num_classes=num_classes, seed=42)
    split = int(0.9 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]
    eval_tf = standardize(mean, std)

    policies = {
        "A: 无增强": {
            "train_tf": standardize(mean, std),
        },
        "B: 随机翻转+裁剪": {
            "train_tf": compose(random_hflip(), random_crop(pad=4), standardize(mean, std)),
        },
        "C: 翻转+裁剪+Cutout": {
            "train_tf": compose(random_hflip(), random_crop(pad=4), cutout(8), standardize(mean, std)),
        },
    }

    results = {}

    for name, policy in policies.items():
        torch.manual_seed(42)

        train_ds = ArrayDataset(X_train, Y_train, transform=policy["train_tf"])
        val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

        model = MiniClassifier(num_classes=num_classes).to(device)
        optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9,
                        weight_decay=5e-4, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, device, num_classes, use_mixup=False,
            )
            va_loss, va_acc, _ = evaluate(model, val_loader, device, num_classes)
            scheduler.step()

        results[name] = va_acc
        print(f"  {name:25s} -> 验证准确率: {va_acc:.3f}")

    best = max(results, key=results.get)
    print(f"\n  最佳策略: {best} ({results[best]:.3f})")
    return results


# ====================================================================
# 主程序：完整训练流程
# ====================================================================

def run_training(dataset_name="fashion-mnist", data_dir="./data", num_epochs=10):
    """运行完整的训练-评估流程。

    Args:
        dataset_name: "fashion-mnist" 或 "cifar-10"
        data_dir: 数据存放目录
        num_epochs: 训练轮次
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    # 加载数据
    start_time = time.time()
    if dataset_name == "fashion-mnist":
        train_loader, val_loader = load_fashion_mnist(data_dir)
        class_names = FASHION_MNIST_CLASSES
    elif dataset_name == "cifar-10":
        train_loader, val_loader = load_cifar10(data_dir)
        class_names = CIFAR10_CLASSES
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    print(f"数据加载耗时: {time.time() - start_time:.1f}s")
    print(f"训练集: {len(train_loader.dataset)} 样本 | "
          f"验证集: {len(val_loader.dataset)} 样本\n")

    # 初始化模型
    model = get_model(dataset_name, num_classes=10, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    # 优化器与调度器
    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    print(f"开始训练 {num_epochs} 个轮次...\n")
    for epoch in range(1, num_epochs + 1):
        current_lr = scheduler.get_last_lr()[0]

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, num_classes=10, use_mixup=True,
        )
        va_loss, va_acc, confusion = evaluate(model, val_loader, device, num_classes=10)
        top5_acc = compute_top_k_accuracy(model, val_loader, device, k=5, num_classes=10)
        scheduler.step()

        print(f"轮次 {epoch:2d}/{num_epochs} | lr={current_lr:.4f} | "
              f"训练: loss={tr_loss:.3f}, acc={tr_acc:.3f} | "
              f"验证: loss={va_loss:.3f}, top1={va_acc:.3f}, top5={top5_acc:.3f}")

    # 最终报告
    print_confusion_matrix(confusion, class_names)

    top1_final = va_acc
    top5_final = top5_acc
    print(f"\n最终验证结果: Top-1 = {top1_final:.1%}, Top-5 = {top5_final:.1%}")
    print(f"总耗时: {time.time() - start_time:.1f}s")

    return model, confusion


def main():
    """入口函数。"""
    parser = argparse.ArgumentParser(description="图像分类完整流水线")
    parser.add_argument("--dataset", default="fashion-mnist", choices=["fashion-mnist", "cifar-10"])
    parser.add_argument("--epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--ablation", action="store_true", help="运行数据增强消融实验")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    if args.ablation:
        print("=== 数据增强消融实验 ===\n")
        run_augmentation_ablation(epochs=min(args.epochs, 5))
    else:
        run_training(args.dataset, args.data_dir, args.epochs)


if __name__ == "__main__":
    main()
