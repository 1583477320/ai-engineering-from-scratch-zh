# === main.py — PyTorch 入门：从张量到完整训练流水线 ===
# 依赖：torch>=2.0, torchvision>=0.15
# 安装：pip install torch torchvision
# 对应课程：阶段 03 · 11（PyTorch 入门）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import struct
import gzip
import os
import time
import math


# ====================================================================
# 第 1 部分：Tensor 基础
# ====================================================================

def demo_tensor_basics():
    """演示 PyTorch 张量的创建、形状、数据类型和设备。"""
    print("=" * 60)
    print("  第 1 部分：Tensor 基础")
    print("=" * 60)

    # --- 创建张量的几种方式 ---
    # 从 Python 列表创建
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\n从列表创建: {x}")
    print(f"  shape={x.shape}, dtype={x.dtype}, device={x.device}")

    # 随机正态分布
    x_rand = torch.randn(3, 4)
    print(f"\n随机张量 torch.randn(3, 4):")
    print(f"  shape={x_rand.shape}, dtype={x_rand.dtype}")

    # 全零和全一
    x_zeros = torch.zeros(2, 3)
    x_ones = torch.ones(2, 3)
    print(f"\n全零: shape={x_zeros.shape}")
    print(f"全一: shape={x_ones.shape}")

    # --- 数据类型转换 ---
    # float32 是 PyTorch 的默认浮点类型
    x_fp16 = x_rand.to(torch.float16)
    x_int8 = x_rand.to(torch.int8)
    print(f"\n类型转换:")
    print(f"  float32 -> float16: dtype={x_fp16.dtype}")
    print(f"  float32 -> int8:   dtype={x_int8.dtype}")

    # --- 常见 dtype 对比 ---
    print("\n常见 dtype 对比:")
    print(f"  {'名称':<12} {'字节':>4} {'精度':>8}  {'典型用途'}")
    print(f"  {'-'*50}")
    print(f"  {'float32':<12} {'4':>4} {'~7 位':>8}  默认训练精度")
    print(f"  {'float16':<12} {'2':>4} {'~3.3 位':>8}  混合精度训练")
    print(f"  {'bfloat16':<12} {'2':>4} {'同 f32':>8}  大语言模型训练")
    print(f"  {'int8':<12} {'1':>4} {'整数':>8}  量化推理")

    # --- 形状操作 ---
    print("\n形状操作:")
    x = torch.randn(2, 3, 4)
    print(f"  原始: shape={x.shape}")

    x_view = x.view(2, 12)
    print(f"  view(2, 12): shape={x_view.shape}")

    x_unsqueeze = x.unsqueeze(0)
    print(f"  unsqueeze(0): shape={x_unsqueeze.shape}  # 新增批次维度")

    x_squeeze = x_unsqueeze.squeeze(0)
    print(f"  squeeze(0): shape={x_squeeze.shape}  # 移除大小为 1 的维度")

    x_permute = x.permute(2, 0, 1)
    print(f"  permute(2, 0, 1): shape={x_permute.shape}  # 重排维度顺序")

    # --- 张量运算 ---
    print("\n向量运算:")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print(f"  a + b = {a + b}")
    print(f"  a * b = {a * b}  # 逐元素乘法")
    print(f"  a @ b = {a @ b}  # 点积（内积）")

    # 矩阵乘法
    A = torch.randn(3, 4)
    B = torch.randn(4, 2)
    C = A @ B
    print(f"\n矩阵乘法: {A.shape} @ {B.shape} = {C.shape}")


# ====================================================================
# 第 2 部分：Autograd 自动微分
# ====================================================================

def demo_autograd():
    """演示 PyTorch 的自动微分引擎：前向传播构建计算图，反向传播自动求导。"""
    print("\n" + "=" * 60)
    print("  第 2 部分：Autograd 自动微分")
    print("=" * 60)

    # --- 基础：手动计算梯度 vs 自动计算 ---
    # 手动计算：y = x^2 + 3x，dy/dx = 2x + 3
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x
    z = y.sum()
    z.backward()  # 反向传播：自动计算 dz/dx

    print(f"\nx = {x.tolist()}")
    print(f"y = x^2 + 3x = {y.tolist()}")
    print(f"z = sum(y) = {z.item():.1f}")
    print(f"dz/dx = 2x + 3 = {x.grad.tolist()}")
    print(f"(手动验证: 2*[1,2,3] + 3 = {[5.0, 7.0, 9.0]})")

    # --- requires_grad：是否追踪梯度 ---
    print("\nrequires_grad 的作用:")
    a = torch.randn(3, requires_grad=True)   # 追踪梯度
    b = torch.randn(3, requires_grad=False)  # 不追踪
    print(f"  a.requires_grad = {a.requires_grad}")
    print(f"  b.requires_grad = {b.requires_grad}")
    print(f"  a + b 的 requires_grad = {(a + b).requires_grad}  # 任一为 True 即为 True")

    # --- 梯度累积的陷阱 ---
    print("\n梯度累积：PyTorch 默认累加梯度，不覆盖")
    w = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    for step in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f"  步骤 {step}: grad = {w.grad.tolist()}")
        # 注意：梯度是累加的，第 2 步的 grad = 第 1 步 + 第 2 步
        # 必须在下一次 backward 之前清零
        w.grad.zero_()

    # --- torch.no_grad：推理时禁用梯度 ---
    print("\ntorch.no_grad(): 推理时关闭梯度追踪，节省内存和计算")
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    print(f"  训练模式: y.requires_grad = {y.requires_grad}")
    with torch.no_grad():
        z = x * 2
        print(f"  no_grad 内: z.requires_grad = {z.requires_grad}")


# ====================================================================
# 第 3 部分：nn.Module 构建网络
# ====================================================================

class MLP(nn.Module):
    """一个简单的三层全连接网络：784 -> 256 -> 128 -> 10"""

    def __init__(self, input_dim=784, hidden1=256, hidden2=128, output_dim=10):
        super().__init__()
        # 将各层定义为类属性，PyTorch 会自动注册参数
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(hidden2, output_dim)
        # 注意：输出层不需要激活函数，因为 CrossEntropyLoss 内部处理 Softmax

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x


def demo_nn_module():
    """演示 nn.Module 的参数管理、训练/评估模式切换。"""
    print("\n" + "=" * 60)
    print("  第 3 部分：nn.Module 构建网络")
    print("=" * 60)

    model = MLP()

    # --- 参数管理 ---
    # model.parameters() 递归收集所有需要梯度的参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 查看每一层的参数形状
    print(f"\n各层参数:")
    for name, param in model.named_parameters():
        print(f"  {name:30s} shape={str(list(param.shape)):15s} params={param.numel():,}")

    # --- train / eval 模式 ---
    print(f"\n模式切换:")
    print(f"  model.training = {model.training}  # 默认为训练模式")
    model.eval()
    print(f"  model.eval() 后: model.training = {model.training}")
    model.train()
    print(f"  model.train() 后: model.training = {model.training}")

    # --- 前向传播 ---
    x = torch.randn(4, 784)  # 4 个样本，28x28 展平为 784
    output = model(x)
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}  # (batch_size, num_classes)")

    # --- nn.Sequential：层的流水线式组合 ---
    print(f"\n使用 nn.Sequential 构建等价模型:")
    seq_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )
    output_seq = seq_model(x)
    print(f"  输出形状: {output_seq.shape}")

    # --- 常用层速查 ---
    print(f"\n常用层速查:")
    print(f"  {'模块':<30} {'功能':<20} {'参数量'}")
    print(f"  {'-'*60}")
    print(f"  {'nn.Linear(in, out)':<30} {'全连接层 Wx+b':<20} {784*256+256:,}")
    print(f"  {'nn.ReLU()':<30} {'max(0, x)':<20} 0")
    print(f"  {'nn.Dropout(p)':<30} {'随机置零':<20} 0")
    print(f"  {'nn.BatchNorm1d(d)':<30} {'批归一化':<20} {256*2:,}")
    print(f"  {'nn.Embedding(v,d)':<30} {'嵌入查找表':<20} vocab*d")


# ====================================================================
# 第 4 部分：Dataset 与 DataLoader
# ====================================================================

class CustomImageDataset(Dataset):
    """自定义 Dataset 示例：加载原始 MNIST 二进制文件。

    Dataset 是一个抽象类，需要实现两个方法：
    - __len__：返回数据集大小
    - __getitem__：根据索引返回单个样本
    """

    MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    MNIST_FILES = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    def __init__(self, data_dir="./mnist_data", train=True, transform=None):
        self.transform = transform
        self._download(data_dir)

        if train:
            self.images = self._load_images(os.path.join(data_dir, self.MNIST_FILES[0]))
            self.labels = self._load_labels(os.path.join(data_dir, self.MNIST_FILES[1]))
        else:
            self.images = self._load_images(os.path.join(data_dir, self.MNIST_FILES[2]))
            self.labels = self._load_labels(os.path.join(data_dir, self.MNIST_FILES[3]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def _download(self, data_dir):
        """下载 MNIST 数据文件（如果不存在）。"""
        os.makedirs(data_dir, exist_ok=True)
        for f in self.MNIST_FILES:
            filepath = os.path.join(data_dir, f)
            if not os.path.exists(filepath):
                print(f"  正在下载 {f} ...")
                import urllib.request
                urllib.request.urlretrieve(self.MNIST_URL + f, filepath)

    def _load_images(self, filepath):
        """从 gzip 压缩的 IDX 文件加载图像，归一化到 [0, 1]。"""
        with gzip.open(filepath, "rb") as f:
            _magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
        return images

    def _load_labels(self, filepath):
        """从 gzip 压缩的 IDX 文件加载标签。"""
        with gzip.open(filepath, "rb") as f:
            _magic, num = struct.unpack(">II", f.read(8))
            data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        return labels


def demo_data_loader():
    """演示 Dataset 和 DataLoader 的使用。"""
    print("\n" + "=" * 60)
    print("  第 4 部分：Dataset 与 DataLoader")
    print("=" * 60)

    # 使用 PyTorch 内置的 torchvision MNIST（最简方式）
    transform = transforms.Compose([
        transforms.ToTensor(),           # PIL Image -> Tensor，自动归一化到 [0,1]
        transforms.Lambda(lambda t: t.view(-1))  # 展平为 784 维向量
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    print(f"\nDataset 信息:")
    print(f"  训练集大小: {len(train_dataset):,}")
    print(f"  测试集大小: {len(test_dataset):,}")
    print(f"  样本形状: {train_dataset[0][0].shape}  # 28x28 展平为 784")
    print(f"  标签范围: {train_dataset.targets.unique().tolist()}")

    # DataLoader 将 Dataset 包装为可迭代的批次流
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,       # 每批 64 个样本
        shuffle=True,        # 每轮打乱数据顺序
        num_workers=0,       # 数据加载进程数（Windows 需设为 0）
        pin_memory=True,     # 锁页内存，加速 GPU 传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,      # 推理时可以用更大的批次
        shuffle=False,
        num_workers=0,
    )

    # 查看一个批次
    images, labels = next(iter(train_loader))
    print(f"\nDataLoader 批次信息:")
    print(f"  images.shape = {images.shape}  # (batch_size, 784)")
    print(f"  labels.shape = {labels.shape}")
    print(f"  labels[:10]  = {labels[:10].tolist()}")

    return train_loader, test_loader


# ====================================================================
# 第 5 部分：训练循环
# ====================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个轮次（epoch），返回平均损失和准确率。

    这是 PyTorch 的标准训练模式，5 行核心代码：
    zero_grad -> forward -> loss -> backward -> step
    """
    model.train()  # 开启 Dropout 等训练时行为
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # 清零梯度（PyTorch 默认累加）
        outputs = model(images)      # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()              # 反向传播：自动计算所有参数的梯度
        optimizer.step()             # 用梯度更新参数

        # 统计指标
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()  # 推理时不需要梯度，节省内存
def evaluate(model, loader, criterion, device):
    """在测试集上评估模型，返回平均损失和准确率。"""
    model.eval()  # 关闭 Dropout，使用 BatchNorm 的累积统计
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ====================================================================
# 第 6 部分：GPU 加速
# ====================================================================

def demo_gpu():
    """演示 GPU 设备管理：检测、移动模型、移动数据。"""
    print("\n" + "=" * 60)
    print("  第 6 部分：GPU 加速")
    print("=" * 60)

    # 检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备检测:")
    print(f"  当前设备: {device}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 移动模型到设备
    model = MLP().to(device)
    print(f"\n模型已移动到: {next(model.parameters()).device}")

    # 移动数据到设备
    x = torch.randn(32, 784).to(device)
    output = model(x)
    print(f"输入设备: {x.device}")
    print(f"输出设备: {output.device}")

    return device


# ====================================================================
# 第 7 部分：模型保存与加载
# ====================================================================

def demo_save_load(model, device):
    """演示模型的保存（state_dict）和加载。"""
    print("\n" + "=" * 60)
    print("  第 7 部分：模型保存与加载")
    print("=" * 60)

    save_path = "mnist_mlp.pt"

    # 保存：只保存参数字典（state_dict），不保存模型结构
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存: {save_path}")

    # 加载：先创建相同结构的模型，再加载参数
    loaded_model = MLP().to(device)
    loaded_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    loaded_model.eval()
    print(f"模型已加载: {save_path}")
    print(f"加载后的模型状态: training={loaded_model.training}  # eval 模式")

    # 验证加载前后输出一致
    test_input = torch.randn(1, 784).to(device)
    with torch.no_grad():
        out_before = model(test_input)
        out_after = loaded_model(test_input)
    match = torch.allclose(out_before, out_after)
    print(f"加载前后输出一致: {match}")

    # 清理文件
    if os.path.exists(save_path):
        os.remove(save_path)


# ====================================================================
# 第 8 部分：TensorBoard 可视化（可选）
# ====================================================================

def demo_tensorboard(writer):
    """演示如何使用 TensorBoard 记录训练指标。

    启动命令：tensorboard --logdir=runs
    """
    print("\n" + "=" * 60)
    print("  第 8 部分：TensorBoard 可视化")
    print("=" * 60)
    print("\nTensorBoard 用于可视化训练曲线和模型结构。")
    print("启动命令: tensorboard --logdir=runs")
    print("浏览器访问: http://localhost:6006")


# ====================================================================
# 主程序：串联所有模块
# ====================================================================

def main():
    print("=" * 60)
    print("  PyTorch 入门 — 阶段 03 · 第 11 课")
    print("=" * 60)

    # --- 第 1 步：环境信息 ---
    print(f"\n  PyTorch 版本: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- 演示各模块 ---
    demo_tensor_basics()
    demo_autograd()
    demo_nn_module()
    train_loader, test_loader = demo_data_loader()
    device = demo_gpu()

    # --- 构建并训练模型 ---
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()  # 内部融合 LogSoftmax + NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n训练配置:")
    print(f"  模型参数量: {num_params:,}")
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  优化器: Adam (lr=1e-3)")
    print(f"  批次大小: 64")
    print(f"  训练轮次: 5")

    # --- 训练循环 ---
    print(f"\n{'='*60}")
    print(f"  开始训练")
    print(f"{'='*60}")

    start_time = time.time()
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start_time
        print(
            f"  轮次 {epoch+1:2d} | "
            f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
            f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f} | "
            f"耗时: {elapsed:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time:.1f}s")

    # --- 保存与加载 ---
    demo_save_load(model, device)

    print(f"\n{'='*60}")
    print(f"  全部演示完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
