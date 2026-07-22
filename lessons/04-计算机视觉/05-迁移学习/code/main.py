# main.py — 迁移学习：特征提取 vs 微调完整对比
# 依赖：torch>=2.0, torchvision>=0.15, numpy
# 安装：pip install torch torchvision numpy
# 对应课程：阶段 04 · 05（迁移学习）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights


# === 1. 合成数据集 ===
# 用不同频率和颜色的纹理图案模拟 10 类图像
# 真实场景中替换为 torchvision.datasets.CIFAR10 或自定义数据集

def synthetic_dataset(num_per_class=100, num_classes=10, size=224, seed=0):
    """生成合成纹理数据集，每个类别有独特的频率和颜色模式。"""
    rng = np.random.default_rng(seed)
    X = np.empty((num_per_class * num_classes, size, size, 3), dtype=np.float32)
    Y = np.empty(num_per_class * num_classes, dtype=np.int64)
    k = 0
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(
                np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij"
            )
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1) + rng.normal(0, 0.05, (size, size, 3))
            X[k] = np.clip(img, 0, 1).astype(np.float32)
            Y[k] = c
            k += 1
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


# === 2. 数据集类 ===
# 应用 ImageNet 标准化——预训练模型期望这个分布

class ArrayDataset(Dataset):
    """将 numpy 数组包装为 PyTorch 数据集，自动应用 ImageNet 归一化。"""

    # ImageNet 的均值和标准差——所有 torchvision 预训练模型默认使用这组值
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = (self.X[i] - self.MEAN) / self.STD
        return torch.from_numpy(img).permute(2, 0, 1).float(), int(self.Y[i])


# === 3. 特征提取：冻结骨干 + 替换分类头 ===
# 这是小数据集（< 1k 样本）的首选方案

def make_feature_extractor(num_classes=10):
    """构建特征提取模型：冻结 ResNet18 骨干，仅训练新的分类头。"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 冻结所有参数——骨干网络不再更新
    for p in model.parameters():
        p.requires_grad = False

    # 替换分类头：原始 fc 输出 1000 类（ImageNet），改为任务类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# === 4. 微调：区分学习率的参数组 ===
# 早期层学习通用特征（边缘、纹理），应该用更小的学习率

def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    """为 ResNet 各阶段构建差异化学习率的参数组。

    decay=0.3 表示每个阶段的学习率是下一个阶段的 30%。
    fc 层获得 base_lr，conv1/bn1 获得 base_lr * 0.3^5 ≈ 0.00243 * base_lr。
    """
    stages = [
        ["conv1", "bn1"],      # 最早期：学习边缘和 Gabor 滤波器
        ["layer1"],             # 纹理和简单模式
        ["layer2"],             # 中层特征
        ["layer3"],             # 物体部件
        ["layer4"],             # 高级语义特征
        ["fc"],                 # 分类头
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        # 只收集需要梯度的参数
        params = [
            p for n, p in model.named_parameters()
            if any(n.startswith(k) for k in names) and p.requires_grad
        ]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups


def make_fine_tune(num_classes=10):
    """构建微调模型：所有参数可训练，配合区分学习率使用。"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.parameters():
        p.requires_grad = True
    return model


# === 5. BatchNorm 处理 ===
# 小数据集上，BN 的运行统计量可能与目标域不匹配

def freeze_bn_stats(model):
    """冻结 BatchNorm 的运行均值和方差。

    在 model.train() 之后调用，仅将 BN 层切回 eval 模式，
    其他层保持训练模式。这样 BN 使用固定的 ImageNet 统计量，
    避免在小数据集上累积噪声统计。
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model


# === 6. 训练和评估循环 ===

def train_and_eval(model, train_loader, val_loader, device,
                   epochs=2, base_lr=1e-3, freeze_bn=False):
    """训练模型并在验证集上评估。

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备（"cuda" 或 "cpu"）
        epochs: 训练轮次
        base_lr: 基础学习率
        freeze_bn: 是否冻结 BatchNorm 统计量

    Returns:
        最终验证集准确率
    """
    model = model.to(device)

    # 构建区分学习率的参数组
    groups = discriminative_param_groups(model, base_lr=base_lr)
    # 如果没有可训练参数（纯冻结），使用默认组
    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    last_val_acc = 0.0
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)

        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # 标签平滑防止模型过度自信——对小数据集特别有用
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        last_val_acc = va_correct / va_total
        print(f"  epoch {epoch}  "
              f"train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {last_val_acc:.3f}")

    return last_val_acc


# === 7. 工具函数 ===

def trainable_param_count(model):
    """统计可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_param_groups(model, base_lr=1e-3):
    """打印各参数组的学习率和参数量。"""
    groups = discriminative_param_groups(model, base_lr=base_lr)
    print(f"  {'stage':<15} {'lr':>10} {'params':>10}")
    print("  " + "-" * 37)
    for g in groups:
        count = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<15} {g['lr']:.2e} {count:>10,}")


# === 8. 主程序 ===

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # 生成合成数据集：10 类，每类 40 张，图像大小 96x96
    X, Y = synthetic_dataset(num_per_class=40, size=96)
    split = int(0.9 * len(X))
    train_ds = ArrayDataset(X[:split], Y[:split])
    val_ds = ArrayDataset(X[split:], Y[split:])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

    # --- 实验 1：特征提取（冻结骨干，仅训练分类头） ---
    print("\n" + "=" * 50)
    print("[实验 1] 特征提取：冻结骨干，仅训练分类头")
    print("=" * 50)
    fe_model = make_feature_extractor(num_classes=10)
    print(f"  可训练参数: {trainable_param_count(fe_model):,}")
    print(f"  总参数: {sum(p.numel() for p in fe_model.parameters()):,}")
    acc_fe = train_and_eval(fe_model, train_loader, val_loader, device,
                            epochs=2, base_lr=3e-2)

    # --- 实验 2：微调（区分学习率，全参数更新） ---
    print("\n" + "=" * 50)
    print("[实验 2] 微调：区分学习率，全参数更新")
    print("=" * 50)
    ft_model = make_fine_tune(num_classes=10)
    print(f"  可训练参数: {trainable_param_count(ft_model):,}")
    print("\n  各阶段学习率:")
    print_param_groups(ft_model, base_lr=1e-3)
    acc_ft = train_and_eval(ft_model, train_loader, val_loader, device,
                            epochs=2, base_lr=1e-3)

    # --- 结果对比 ---
    print("\n" + "=" * 50)
    print("结果对比")
    print("=" * 50)
    print(f"  特征提取验证准确率: {acc_fe:.3f}")
    print(f"  微调验证准确率:     {acc_ft:.3f}")

    if acc_ft > acc_fe:
        print(f"  微调比特征提取高 {(acc_ft - acc_fe)*100:.1f} 个百分点")
    else:
        print(f"  特征提取表现更好——说明数据量太小，微调破坏了预训练特征")


if __name__ == "__main__":
    main()
