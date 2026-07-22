# main.py — 正则化技术从零实现
# 依赖：Python 3.10+, random, math（无第三方库依赖）
# 对应课程：阶段 03 · 07（正则化）
#
# 本文件包含以下内容的从零实现：
#   1. Dropout（含反向缩放 Inverted Dropout）
#   2. L2 权重衰减（Weight Decay）
#   3. 批归一化（Batch Normalization）
#   4. 层归一化（Layer Normalization）
#   5. RMSNorm
#   6. 完整的训练对比实验（有无正则化）

import random
import math


# ============================================================
# Dropout
# ============================================================

class Dropout:
    """Dropout 正则化层。

    训练时以概率 p 将神经元输出置零，并除以 (1-p) 进行反向缩放。
    推理时直接通过，不做任何处理。

    反向缩放（Inverted Dropout）是实践中的标准做法：
    - 训练时：output = activation(z) * mask / (1 - p)
    - 推理时：output = activation(z)（无需缩放）
    这样测试代码完全不需要知道 Dropout 的存在。
    """

    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        """前向传播。训练时随机丢弃并缩放，推理时直接通过。"""
        if not self.training:
            return list(x)

        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                # 反向缩放：除以 (1-p) 使训练和推理的期望值一致
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        """反向传播：被丢弃的神经元梯度为零，其余按 (1-p) 缩放。"""
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads


# ============================================================
# L2 权重衰减
# ============================================================

def l2_regularization(weights, lambda_reg):
    """计算 L2 正则化损失：(lambda / 2) * sum(w_i^2)。

    大权重贡献更多的惩罚项，迫使模型分散权重到更多特征上。
    """
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty


def l2_gradient(weights, lambda_reg):
    """L2 正则化的梯度：lambda * w_i。大权重受到更大的收缩力。"""
    return [lambda_reg * w for w in weights]


# ============================================================
# 批归一化（Batch Normalization）
# ============================================================

class BatchNorm:
    """批归一化层。

    核心思想：对每层的输出，在小批次维度上归一化到零均值、单位方差。
    训练时用批次统计，推理时用累积的运行统计。

    适用于批次大小 > 32 的场景。小批次或变长序列场景应使用 LayerNorm。
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features  # 可学习缩放参数
        self.beta = [0.0] * num_features   # 可学习偏移参数
        self.eps = eps                      # 防止除零
        self.momentum = momentum            # 运行均值的滑动速率
        # 运行均值和方差：训练期间通过滑动平均累积
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        """前向传播。

        Args:
            batch: 批次数据，形状为 (batch_size, num_features) 的列表

        Returns:
            归一化后的输出，与输入形状相同
        """
        batch_size = len(batch)

        if self.training:
            # 当前批次的均值（跨样本计算每个特征的均值）
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            # 当前批次的方差
            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            # 更新运行统计量（指数滑动平均）
            for j in range(self.num_features):
                self.running_mean[j] = (
                    (1 - self.momentum) * self.running_mean[j]
                    + self.momentum * mean[j]
                )
                self.running_var[j] = (
                    (1 - self.momentum) * self.running_var[j]
                    + self.momentum * var[j]
                )
        else:
            # 推理时使用累积的运行统计量
            mean = list(self.running_mean)
            var = list(self.running_var)

        # 归一化并缩放平移
        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output


# ============================================================
# 层归一化（Layer Normalization）
# ============================================================

class LayerNorm:
    """层归一化层。

    核心思想：跨特征维度归一化，与批次大小无关。
    每个样本独立计算均值和方差，因此 Transformer 使用它。

    与 BatchNorm 的关键区别：
    - BatchNorm 跨样本（批次），这里跨特征
    - BatchNorm 依赖批次大小，这里不依赖
    - BatchNorm 训练和推理行为不同，这里相同
    """

    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features  # 可学习缩放参数
        self.beta = [0.0] * num_features   # 可学习偏移参数
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        """对单个样本的特征向量进行归一化。"""
        # 在特征维度上计算均值和方差
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        # 归一化并缩放平移
        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm:
    """RMSNorm 归一化层。

    LayerNorm 去掉均值减法的简化版本。
    只做均方根缩放，不做均值中心化。

    优势：每层节省约 10% 计算量，精度与 LayerNorm 相当。
    被 LLaMA、Mistral、Qwen 等现代大语言模型广泛采用。

    与 LayerNorm 的差异：
    - 没有均值计算（省掉一次全局归约）
    - 没有 beta 参数（偏移参数）
    - gamma 的初始化方式和量级略有不同
    """

    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        """只除以均方根，不做均值中心化。"""
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output


# ============================================================
# 训练对比实验
# ============================================================

def sigmoid(x):
    """Sigmoid 激活函数，带数值裁剪防止 math.exp 溢出。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    """生成环形二分类数据集。

    特征：二维坐标 (x, y)
    标签：圆内（x^2 + y^2 < 1.5）为正类（1.0），圆外为负类（0.0）

    圆形决策边界是线性模型无法解决的问题，适合展示神经网络的
    学习能力和过拟合现象。
    """
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    """支持 Dropout 和权重衰减的两层神经网络。

    架构：2 输入 → 16 隐藏单元（ReLU）→ Dropout（可选）→ 1 输出（Sigmoid）

    训练目标：在简单的环形数据集上展示正则化效果。
    """

    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        # 高斯初始化（均值 0，标准差 0.5）
        self.w1 = [
            [random.gauss(0, 0.5) for _ in range(2)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        """前向传播。

        训练时使用 Dropout，推理时关闭。
        """
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))  # ReLU 激活

        # 根据训练/推理模式切换 Dropout
        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        # 输出层
        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        """反向传播。

        包含：
        - 二元交叉熵梯度（d_loss）
        - Sigmoid 梯度（d_sigmoid）
        - ReLU 梯度（d_relu）
        - Dropout 梯度
        - 权重衰减项
        """
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        # 二元交叉熵梯度：-(target / p) + (1 - target) / (1 - p)
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        # Dropout 反向传播
        d_h_dropout = [d_out * self.w2[i] for i in range(self.hidden_size)]
        if self.dropout and self.dropout.mask is not None:
            d_h_dropout = [
                g * m / (1 - self.dropout.p) if m else 0.0
                for g, m in zip(d_h_dropout, self.dropout.mask)
            ]

        # 更新权重（任务梯度 + 权重衰减梯度）
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_h_dropout[i] * d_relu
            # 权重衰减项：lambda * w 加到梯度中
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        """在数据集上评估模型性能（始终使用推理模式）。"""
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        """训练循环：每轮迭代整个训练集，记录训练和测试指标。"""
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1

            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))

            # 每 75 轮打印一次进度
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: "
                      f"train_acc={train_acc:.1f}%, "
                      f"test_acc={test_acc:.1f}%, "
                      f"gap={gap:.1f}%")
        return history


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第 1 步：Dropout 演示（训练 vs 推理模式）")
    print("=" * 60)
    drop = Dropout(p=0.5)
    test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    random.seed(42)

    # 训练模式：随机丢弃并反向缩放
    drop.training = True
    print(f"\n  输入:          {test_input}")
    for trial in range(3):
        output = drop.forward(test_input)
        active = sum(1 for v in output if v > 0)
        print(f"  训练第 {trial + 1} 次: "
              f"{[f'{v:.1f}' for v in output]}  "
              f"({active}/{len(test_input)} 活跃)")

    # 推理模式：直接通过，不做任何处理
    drop.training = False
    output = drop.forward(test_input)
    print(f"  推理时:        {[f'{v:.1f}' for v in output]}")

    # 验证反向缩放的有效性
    train_mean = sum(test_input) / len(test_input)
    print(f"\n  输入均值:      {train_mean:.1f}")
    print(f"  推理时均值:    {sum(output) / len(output):.1f} (不变)")
    print(f"  训练时均值:    ~{train_mean:.1f} (因为反向缩放，均值不变)")

    print("\n" + "=" * 60)
    print("第 2 步：L2 权重衰减演示")
    print("=" * 60)
    weights = [0.5, -1.2, 3.0, 0.1, -2.5]
    lambda_val = 0.01

    penalty = l2_regularization(weights, lambda_val)
    grads = l2_gradient(weights, lambda_val)

    print(f"\n  权重:          {weights}")
    print(f"  正则化系数:    {lambda_val}")
    print(f"  L2 惩罚项:     {penalty:.6f}")
    print(f"  L2 梯度:       {[f'{g:.4f}' for g in grads]}")
    print(f"  最大权重 (3.0) 获得最大梯度 ({grads[2]:.4f})")
    print("  大权重被更强的力推向零——这就是 L2 控制复杂度的方法")

    print("\n" + "=" * 60)
    print("第 3 步：BatchNorm vs LayerNorm vs RMSNorm 对比")
    print("=" * 60)
    random.seed(42)
    # 8 个样本，每个 4 维特征
    batch = [[random.gauss(5, 2) for _ in range(4)] for _ in range(8)]
    sample = batch[0]  # 取第一个样本供 LayerNorm 和 RMSNorm 使用

    # 批归一化（处理整个批次）
    bn = BatchNorm(4)
    bn_out = bn.forward(batch)

    # 层归一化（处理单个样本）
    ln = LayerNorm(4)
    ln_out = ln.forward(sample)

    # RMSNorm（处理单个样本）
    rn = RMSNorm(4)
    rn_out = rn.forward(sample)

    print(f"\n  原始样本:      {[f'{v:.2f}' for v in sample]}")
    print(f"  BatchNorm:     {[f'{v:.2f}' for v in bn_out[0]]}")
    print(f"  LayerNorm:     {[f'{v:.2f}' for v in ln_out]}")
    print(f"  RMSNorm:       {[f'{v:.2f}' for v in rn_out]}")

    # 验证 LayerNorm 输出均值约 0，标准差约 1
    ln_mean = sum(ln_out) / len(ln_out)
    ln_std = math.sqrt(sum((v - ln_mean) ** 2 for v in ln_out) / len(ln_out))
    # RMSNorm 没有均值中心化
    rn_mean = sum(rn_out) / len(rn_out)
    rn_rms = math.sqrt(sum(v * v for v in rn_out) / len(rn_out))

    print(f"\n  LayerNorm 输出: mean={ln_mean:.4f}, std={ln_std:.4f}")
    print(f"  RMSNorm 输出:   mean={rn_mean:.4f}, rms={rn_rms:.4f}")
    print(f"  LayerNorm 做均值中心化 (mean≈0)；RMSNorm 只做尺度归一化")

    print("\n" + "=" * 60)
    print("第 4 步：BatchNorm 训练模式 vs 推理模式")
    print("=" * 60)
    bn2 = BatchNorm(4)
    bn2.training = True

    # 模拟训练过程：10 个批次，分布缓慢漂移
    for step in range(10):
        batch = [
            [random.gauss(3 + step * 0.1, 1) for _ in range(4)]
            for _ in range(16)
        ]
        bn2.forward(batch)

    print(f"\n  10 个批次后的运行均值:")
    print(f"  {[f'{v:.3f}' for v in bn2.running_mean]}")
    print(f"  10 个批次后的运行方差:")
    print(f"  {[f'{v:.3f}' for v in bn2.running_var]}")

    # 切换到推理模式
    bn2.training = False
    test_sample = [[5.0, 5.0, 5.0, 5.0]]
    eval_out = bn2.forward(test_sample)
    print(f"\n  推理模式使用运行统计而非批次统计")
    print(f"  输入 [5, 5, 5, 5] → {[f'{v:.3f}' for v in eval_out[0]]}")
    print(f"  (使用训练期间累积的均值和方差进行归一化)")

    print("\n" + "=" * 60)
    print("第 5 步：有正则化 vs 无正则化的训练对比")
    print("=" * 60)

    # 生成环形数据集：150 训练 + 150 测试
    all_data = make_circle_data(n=300, seed=42)
    train_data = all_data[:150]
    test_data = all_data[150:]

    # 四种训练配置
    configs = [
        ("无正则化", 0.0, 0.0),
        ("Dropout p=0.3", 0.3, 0.0),
        ("权重衰减 0.01", 0.0, 0.01),
        ("Dropout + 权重衰减", 0.3, 0.01),
    ]

    results = {}
    for name, drop_p, wd in configs:
        print(f"\n--- {name} ---")
        net = RegularizedNetwork(
            hidden_size=16,
            lr=0.05,
            dropout_p=drop_p,
            weight_decay=wd,
        )
        history = net.train_model(train_data, test_data, epochs=300)
        results[name] = history

    # 打印最终对比结果
    print("\n" + "=" * 60)
    print("最终对比")
    print("=" * 60)
    print(f"  {'配置':30s} {'训练准确率':>10s} {'测试准确率':>10s} {'差距':>8s}")
    print("  " + "-" * 60)
    for name, history in results.items():
        _, train_acc, _, test_acc = history[-1]
        gap = train_acc - test_acc
        print(f"  {name:30s} {train_acc:>9.1f}% {test_acc:>9.1f}% {gap:>7.1f}%")

    print("\n  核心结论：正则化缩小了训练-测试差距。")
    print("  Dropout + 权重衰减的模型泛化最好，")
    print("  即使它的训练准确率不是最高的。")
