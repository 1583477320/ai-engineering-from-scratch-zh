"""
mini_framework.py -- 从零构建的迷你深度学习框架

本框架实现了类似 PyTorch 的核心抽象：
  Tensor（自动微分）、Module、Linear、ReLU、Sigmoid、Dropout、
  Sequential、MSELoss、BCELoss、SGD、Adam、DataLoader

用 ~550 行纯 Python 实现，无任何第三方依赖。

依赖：无（仅标准库）
对应课程：阶段 03 · 10（迷你框架）
"""

import math
import random

# ==============================================================================
# 第一部分：Tensor 与自动微分
# ==============================================================================


class Tensor:
    """一维张量，支持自动微分。

    每个 Tensor 记录自己是如何被计算出来的（_backward_fn），
    backward() 时沿着计算图反向传播梯度。这就是 PyTorch autograd 的核心思想。
    """

    def __init__(self, data, requires_grad=False):
        # 统一将标量处理为 1 元素列表
        if isinstance(data, (int, float)):
            data = [data]
        self.data = list(data)
        self.requires_grad = requires_grad
        # 梯度：与 data 同形状
        self.grad = [0.0] * len(self.data)
        # 反向传播函数（由创建该 Tensor 的操作定义）
        self._backward_fn = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({self.data})"

    def backward(self, grad=None):
        """反向传播：累积梯度并沿计算图传播。

        调用链：loss.backward() -> 每个 Tensor 的 _backward_fn ->
        反向传播到前一层的输出 Tensor -> ...
        """
        if grad is None:
            grad = [1.0] * len(self.data)
        if not self.requires_grad:
            return
        # 累积梯度（支持多个输出路径汇聚到同一个 Tensor）
        for i in range(len(self.data)):
            self.grad[i] += grad[i]
        # 沿计算图继续反向传播
        if self._backward_fn is not None:
            self._backward_fn(grad)

    def zero_grad(self):
        """清零梯度。"""
        self.grad = [0.0] * len(self.data)


# --- 自动微分操作（autograd functions）---


def add(a, b):
    """Tensor 逐元素加法，支持自动微分。"""
    c = Tensor(
        [ai + bi for ai, bi in zip(a.data, b.data)],
        requires_grad=a.requires_grad or b.requires_grad,
    )
    if c.requires_grad:
        # d(a+b)/da = 1, d(a+b)/db = 1
        c._backward_fn = lambda g: (a.backward(g), b.backward(g))
    return c


def mul(a, b):
    """Tensor 逐元素乘法，支持自动微分。"""
    c = Tensor(
        [ai * bi for ai, bi in zip(a.data, b.data)],
        requires_grad=a.requires_grad or b.requires_grad,
    )
    if c.requires_grad:
        def _backward(grad):
            # d(a*b)/da = b, d(a*b)/db = a
            a.backward([g * bi for g, bi in zip(grad, b.data)])
            b.backward([g * ai for g, ai in zip(grad, a.data)])
        c._backward_fn = _backward
    return c


def linear_forward(input_vec, weight_rows, bias):
    """线性层前向传播。

    weight_rows: list[Tensor]，每行是权重向量（fan_in 维），行数 = fan_out
    bias: Tensor，偏置向量（fan_out 维）
    返回: Tensor，输出向量（fan_out 维）

    这是线性层的核心操作：W @ x + b，同时记录自动微分所需的反向传播信息。
    """
    fan_out = len(weight_rows)
    fan_in = len(input_vec)

    # 计算 W @ x + b
    result_data = []
    for i in range(fan_out):
        val = sum(weight_rows[i].data[j] * input_vec.data[j]
                  for j in range(fan_in))
        val += bias.data[i]
        result_data.append(val)

    requires_grad = (
        input_vec.requires_grad
        or any(w.requires_grad for w in weight_rows)
        or bias.requires_grad
    )
    c = Tensor(result_data, requires_grad=requires_grad)

    if not requires_grad:
        return c

    def _backward(grad):
        # --- 梯度反向传播的三条路径 ---

        # 1. 输入梯度：d(loss)/d(input[j]) = sum_i(grad[i] * W[i][j])
        if input_vec.requires_grad:
            input_grad = [0.0] * fan_in
            for i in range(fan_out):
                for j in range(fan_in):
                    input_grad[j] += grad[i] * weight_rows[i].data[j]
            input_vec.backward(input_grad)

        # 2. 权重梯度（直接累积）：d(loss)/d(W[i][j]) = grad[i] * input[j]
        for i in range(fan_out):
            if weight_rows[i].requires_grad:
                for j in range(fan_in):
                    weight_rows[i].grad[j] += grad[i] * input_vec.data[j]

        # 3. 偏置梯度（直接累积）：d(loss)/d(bias[i]) = grad[i]
        if bias.requires_grad:
            for i in range(fan_out):
                bias.grad[i] += grad[i]

    c._backward_fn = _backward
    return c


def relu(x):
    """ReLU 激活函数：max(0, x)。"""
    c = Tensor(
        [max(0.0, v) for v in x.data],
        requires_grad=x.requires_grad,
    )
    if c.requires_grad:
        def _backward(grad):
            # ReLU 的局部梯度：输入 > 0 时梯度为 1，否则为 0
            x.backward([g if v > 0 else 0.0 for g, v in zip(grad, x.data)])
        c._backward_fn = _backward
    return c


def sigmoid(x):
    """Sigmoid 激活函数：1 / (1 + e^{-x})。"""
    s = [
        1.0 / (1.0 + math.exp(max(-500, min(500, -v))))
        for v in x.data
    ]
    c = Tensor(s, requires_grad=x.requires_grad)
    if c.requires_grad:
        def _backward(grad):
            # Sigmoid 的局部梯度：sigmoid(x) * (1 - sigmoid(x))
            x.backward([g * sv * (1 - sv) for g, sv in zip(grad, s)])
        c._backward_fn = _backward
    return c


# ==============================================================================
# 第二部分：Module 基类
# ==============================================================================


class Module:
    """所有层的基类。

    PyTorch nn.Module 的简化版本。每个层都必须实现 forward()。
    参数管理、训练/评估模式切换由基类统一处理。
    """

    def __init__(self):
        self.training = True

    def forward(self, x):
        """前向传播。"""
        raise NotImplementedError

    def parameters(self):
        """返回所有可训练参数（Tensor 列表）。"""
        return []

    def train(self):
        """切换到训练模式。
        Dropout 会随机丢弃，BatchNorm 使用批统计量。
        """
        self.training = True

    def eval(self):
        """切换到评估模式。
        Dropout 不丢弃，BatchNorm 使用运行均值/方差。
        """
        self.training = False

    def zero_grad(self):
        """清零所有参数的梯度。"""
        for p in self.parameters():
            p.zero_grad()


# ==============================================================================
# 第三部分：层
# ==============================================================================


class Linear(Module):
    """线性层：output = W @ input + b

    权重使用 Kaiming 初始化（适配 ReLU），偏置初始化为 0。
    """

    def __init__(self, fan_in, fan_out):
        super().__init__()
        # Kaiming 初始化
        std = math.sqrt(2.0 / fan_in)
        self.weight_rows = [
            Tensor([random.gauss(0, std) for _ in range(fan_in)],
                   requires_grad=True)
            for _ in range(fan_out)
        ]
        self.bias = Tensor([0.0] * fan_out, requires_grad=True)
        self.fan_in = fan_in
        self.fan_out = fan_out

    def forward(self, x):
        return linear_forward(x, self.weight_rows, self.bias)

    def parameters(self):
        return self.weight_rows + [self.bias]


class ReLU(Module):
    """ReLU 激活层。"""

    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    """Sigmoid 激活层（二分类输出层用）。"""

    def forward(self, x):
        return sigmoid(x)


class Dropout(Module):
    """随机丢弃层。

    训练时以概率 p 随机丢弃元素，并对保留元素缩放 1/(1-p)
    以保证期望值不变。推理时直接通过。
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x):
        if not self.training:
            return x
        # 生成掩码：丢弃的位置为 0，保留的位置缩放 1/(1-p)
        scale = 1.0 / (1.0 - self.p)
        self._mask = [
            0.0 if random.random() < self.p else scale
            for _ in x.data
        ]
        c = Tensor(
            [xi * mi for xi, mi in zip(x.data, self._mask)],
            requires_grad=x.requires_grad,
        )
        if c.requires_grad and x.requires_grad:
            def _backward(grad):
                x.backward([g * m for g, m in zip(grad, self._mask)])
            c._backward_fn = _backward
        return c


# ==============================================================================
# 第四部分：顺序容器
# ==============================================================================


class Sequential(Module):
    """顺序容器。

    前向传播从左到右逐一执行所有子模块。
    自动微分保证了梯度能反向传播 —— 无需在 Sequential 中手动实现 backward。
    """

    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x):
        for module in self._modules:
            x = module.forward(x)
        return x

    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self._modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self._modules:
            module.eval()

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def count_parameters(self):
        return sum(len(p) for p in self.parameters())


# ==============================================================================
# 第五部分：损失函数
# ==============================================================================


class MSELoss:
    """均方误差损失（回归任务用）。

    返回值是一个 Tensor（支持自动微分），以便继续反向传播。
    """

    def __call__(self, predicted, target):
        n = len(predicted)
        t = target.data if isinstance(target, Tensor) else target
        # 计算 MSE
        loss_val = sum((p - t[i]) ** 2 for i, p in enumerate(predicted.data)) / n
        result = Tensor([loss_val], requires_grad=predicted.requires_grad)

        if predicted.requires_grad:
            def _backward(grad):
                pred_grad = [2.0 * (predicted.data[i] - t[i]) / n * grad[0]
                             for i in range(n)]
                predicted.backward(pred_grad)
            result._backward_fn = _backward

        return result


class BCELoss:
    """二元交叉熵损失（二分类任务用）。

    内部加入 epsilon 防止 log(0) 导致的数值问题。
    """

    def __init__(self):
        self.eps = 1e-7

    def __call__(self, predicted, target):
        n = len(predicted)
        if isinstance(target, Tensor):
            target = target.data
        t = target if isinstance(target, list) else [target]

        # 限制预测值范围，防止 log(0)
        loss_val = 0.0
        pred_grads = [0.0] * n
        for i in range(n):
            p = max(self.eps, min(1 - self.eps, predicted.data[i]))
            loss_val += -(t[i] * math.log(p) + (1 - t[i]) * math.log(1 - p))
            pred_grads[i] = (-t[i] / p + (1 - t[i]) / (1 - p)) / n

        result = Tensor([loss_val / n], requires_grad=predicted.requires_grad)

        if predicted.requires_grad:
            def _backward(grad):
                scaled_grads = [g * grad[0] for g in pred_grads]
                predicted.backward(scaled_grads)
            result._backward_fn = _backward

        return result


# ==============================================================================
# 第六部分：优化器
# ==============================================================================


class SGD:
    """随机梯度下降优化器。

    param -= lr * grad
    """

    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        self.params = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """执行一步参数更新。"""
        for param in self.params:
            for i in range(len(param)):
                decay = self.weight_decay * param.data[i]
                param.data[i] -= self.lr * (param.grad[i] + decay)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class Adam:
    """Adam 优化器（自适应矩估计）。

    结合动量（一阶矩）和 RMSProp（二阶矩）的优点。
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # 为每个参数的每个元素维护独立的一阶矩和二阶矩
        self.m = [[0.0] * len(p) for p in self.params]
        self.v = [[0.0] * len(p) for p in self.params]

    def step(self):
        """执行一步参数更新。"""
        self.t += 1
        for idx, param in enumerate(self.params):
            for i in range(len(param)):
                g = param.grad[i]

                # 更新一阶矩（动量）和二阶矩（梯度平方的指数移动平均）
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1 - self.beta1) * g
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1 - self.beta2) * g * g

                # 偏差修正（消除初始阶段的零初始化偏差）
                m_hat = self.m[idx][i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx][i] / (1 - self.beta2 ** self.t)

                # 参数更新
                param.data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


# ==============================================================================
# 第七部分：数据加载器
# ==============================================================================


class DataLoader:
    """数据加载器。

    将数据集分批，可选打乱顺序。
    数据格式：[(input_list, target_list), ...]
    """

    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# ==============================================================================
# 第八部分：XOR 问题演示
# ==============================================================================


def build_xor_data(seed=42):
    """构建 XOR 数据集（二分类）。

    XOR 问题：当两个输入不同时输出 1，相同时输出 0。
    这是感知机无法解决的经典问题（Minsky & Papert, 1969）。
    """
    random.seed(seed)
    # 原始 4 个 XOR 样本
    base = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    # 重复 100 次得到 400 个样本
    data = base * 100
    # 打乱顺序
    random.shuffle(data)
    # 80/20 划分
    split = int(len(data) * 0.8)
    return data[:split], data[split:]


def build_model():
    """构建用于 XOR 的 MLP 模型。

    架构：2（输入）-> 16 维隐藏层 -> 8 维隐藏层 -> 1（输出）
    使用 ReLU 隐藏层 + Sigmoid 输出层。
    """
    return Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )


def compute_accuracy(model, data):
    """计算模型在数据集上的准确率。"""
    correct = 0
    for x, t in data:
        output = model.forward(Tensor(x))
        predicted = 1.0 if output.data[0] >= 0.5 else 0.0
        if predicted == t[0]:
            correct += 1
    return correct / len(data), correct, len(data)


def train_xor(model, criterion, optimizer, train_data, test_data, epochs=200):
    """完整训练循环。"""
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            for x_list, t_list in zip(batch_inputs, batch_targets):
                # 前向传播（自动建立计算图）
                pred = model.forward(Tensor(x_list))
                loss = criterion(pred, t_list)

                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.data[0]
                total_samples += 1

        # 每 50 个轮次输出一次训练状态
        if epoch % 50 == 0 or epoch == epochs - 1:
            train_acc, _, _ = compute_accuracy(model, train_data)
            test_acc, _, _ = compute_accuracy(model, test_data)
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | "
                  f"Train: {train_acc:.1%} | Test: {test_acc:.1%}")

    return model


def predict_xor(model, x1, x2):
    """对单个 XOR 输入进行预测。"""
    output = model.forward(Tensor([x1, x2]))
    prediction = output.data[0]
    return prediction, 1.0 if prediction >= 0.5 else 0.0


def demo_xor_predictions(model):
    """展示模型在 XOR 数据集上的全部 4 个样本的预测结果。"""
    print("\n  所有 XOR 样本预测结果：")
    print("  " + "-" * 50)
    print(f"  {'输入':>10} | {'真实值':>6} | {'预测值':>8} | {'结果'}")
    print("  " + "-" * 50)
    for x1, x2, expected in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        prob, pred = predict_xor(model, x1, x2)
        status = "OK" if pred == expected else "WRONG"
        print(f"  ({x1}, {x2})    |   {expected}    |  {prob:.4f} ({int(pred)}) | {status}")
    print("  " + "-" * 50)


# ==============================================================================
# 主函数：三个实验对比
# ==============================================================================


def experiment_adam():
    """实验 1：Adam 优化器训练 XOR。"""
    print("-" * 70)
    print("实验 1：Adam 优化器 (lr=0.01)")
    print("-" * 70)

    train_data, test_data = build_xor_data()
    model = build_model()

    params = model.count_parameters()
    print(f"  模型参数总量: {params}")

    model = train_xor(model, BCELoss(), Adam(model.parameters(), lr=0.01),
                      train_data, test_data, epochs=200)

    model.eval()
    test_acc, correct, total = compute_accuracy(model, test_data)
    print(f"\n  Adam 测试准确率: {test_acc:.1%} ({correct}/{total})")

    return model


def experiment_sgd():
    """实验 2：SGD 优化器（相同架构）。"""
    print("\n" + "-" * 70)
    print("实验 2：SGD 优化器 (lr=0.1)")
    print("-" * 70)

    train_data, test_data = build_xor_data()
    model = build_model()

    model = train_xor(model, BCELoss(), SGD(model.parameters(), lr=0.1),
                      train_data, test_data, epochs=200)

    model.eval()
    test_acc, correct, total = compute_accuracy(model, test_data)
    print(f"\n  SGD 测试准确率: {test_acc:.1%} ({correct}/{total})")
    return test_acc


def experiment_dropout():
    """实验 3：带 Dropout 的训练。"""
    print("\n" + "-" * 70)
    print("实验 3：Adam + Dropout(p=0.2)")
    print("-" * 70)

    train_data, test_data = build_xor_data()

    # 带 Dropout 的模型
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Dropout(0.2),
        Linear(16, 8),
        ReLU(),
        Dropout(0.2),
        Linear(8, 1),
        Sigmoid(),
    )

    model = train_xor(model, BCELoss(), Adam(model.parameters(), lr=0.01),
                      train_data, test_data, epochs=200)

    model.eval()
    test_acc, correct, total = compute_accuracy(model, test_data)
    print(f"\n  Adam + Dropout 测试准确率: {test_acc:.1%} ({correct}/{total})")
    return test_acc


if __name__ == "__main__":
    print("=" * 70)
    print("从零构建的迷你深度学习框架 —— 演示")
    print("=" * 70)
    print()
    print("任务：XOR 异或问题（二分类）")
    print("模型：MLP (2 -> 16 -> 8 -> 1) + ReLU + Sigmoid")
    print("数据：400 个样本（4 个原始样本 × 100 重复）")
    print()

    # === 实验 1：Adam ===
    model_adam = experiment_adam()
    demo_xor_predictions(model_adam)

    # === 实验 2：SGD ===
    acc_sgd = experiment_sgd()

    # === 实验 3：Adam + Dropout ===
    acc_dropout = experiment_dropout()

    # === 实验对比 ===
    _, correct_adam, total_adam = compute_accuracy(model_adam, build_xor_data()[1])

    print("\n" + "=" * 70)
    print("实验结果对比")
    print("=" * 70)
    print(f"  Adam (无正则化):     {correct_adam}/{total_adam} " +
          f"({correct_adam / total_adam:.1%})")
    print(f"  SGD (无正则化):      {acc_sgd:.1%}")
    print(f"  Adam + Dropout(0.2): {acc_dropout:.1%}")

    print("\n" + "=" * 70)
    print("迷你框架组件清单")
    print("=" * 70)
    print(f"  自动微分:     Tensor + autograd ops (add, linear_forward, relu, sigmoid)")
    print(f"  核心层:       Linear, ReLU, Sigmoid")
    print(f"  正则化:       Dropout")
    print(f"  容器:         Sequential")
    print(f"  损失函数:     MSELoss, BCELoss")
    print(f"  优化器:       SGD, Adam")
    print(f"  数据加载:     DataLoader (batching + shuffle)")
    print(f"  代码量:       ~550 行纯 Python，零依赖")
    print()
    print("每个组件都可以在 PyTorch 中找到对应项 —— 区别在于 PyTorch")
    print("经过了多年 GPU 优化和生产验证。但核心架构是一样的。")
