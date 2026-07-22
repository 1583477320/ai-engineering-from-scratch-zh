# main.py — 从零实现反向传播引擎
# 依赖：Python 3.10+（仅使用标准库）
# 对应课程：阶段 03 · 03（反向传播）

import math
import random


class Value:
    """计算图中的节点，支持自动微分。

    每个 Value 存储：
    - data: 前向传播计算出的数值
    - grad: 反向传播计算出的梯度
    - _children: 参与生成当前节点的子节点（用于拓扑排序）
    - _op: 生成当前节点的操作符（用于调试）
    - _backward: 反向传播函数，将梯度传递给子节点
    """

    def __init__(self, data, children=(), op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # === 加法 ===
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 加法的局部导数为 1，梯度直接传递
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    # === 乘法 ===
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乘法：d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    # === 取负 ===
    def __neg__(self):
        return self * -1

    # === 减法 ===
    def __sub__(self, other):
        return self + (-other)

    # === 幂运算（整数指数） ===
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "指数必须是数值"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # === Sigmoid 激活函数 ===
    def sigmoid(self):
        # 裁剪输入防止 exp 溢出
        x = max(-500, min(500, self.data))
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            # sigmoid 的导数：s * (1 - s)
            self.grad += (s * (1 - s)) * out.grad

        out._backward = _backward
        return out

    # === ReLU 激活函数 ===
    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            # ReLU 的导数：x > 0 时为 1，否则为 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # === 反向传播入口 ===
    def backward(self):
        """执行反向传播，计算所有节点的梯度。"""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 损失对自身的梯度为 1
        self.grad = 1.0

        # 逆拓扑序执行反向传播
        for v in reversed(topo):
            v._backward()


def mse_loss(predicted, target):
    """均方误差损失函数。"""
    diff = predicted - target
    return diff * diff


# === 神经元 ===
class Neuron:
    """单个神经元：加权求和 + 激活函数。"""

    def __init__(self, n_inputs, activation="sigmoid"):
        # Xavier 初始化：防止 sigmoid 饱和
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        # 加权求和
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        # 应用激活函数
        if self.activation == "sigmoid":
            return act.sigmoid()
        elif self.activation == "relu":
            return act.relu()
        return act

    def parameters(self):
        return self.weights + [self.bias]


# === 层 ===
class Layer:
    """一层神经元。"""

    def __init__(self, n_inputs, n_outputs, activation="sigmoid"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


# === 网络 ===
class Network:
    """多层前馈网络。"""

    def __init__(self, sizes, activations=None):
        """sizes: 各层神经元数量，如 [2, 4, 1] 表示 2 输入、4 隐藏、1 输出。"""
        if activations is None:
            activations = ["sigmoid"] * (len(sizes) - 1)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1], activations[i]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# === 数值梯度检查 ===
def numerical_gradient(param, loss_fn, epsilon=1e-5):
    """使用有限差分法计算数值梯度，用于验证解析梯度。"""
    original = param.data

    param.data = original + epsilon
    loss_plus = loss_fn()

    param.data = original - epsilon
    loss_minus = loss_fn()

    param.data = original
    return (loss_plus - loss_minus) / (2 * epsilon)


def gradient_check(net, x, target, tolerance=1e-5):
    """对比解析梯度与数值梯度，验证反向传播实现正确性。"""
    # 先执行前向和反向传播，得到解析梯度
    pred = net(x)
    loss = mse_loss(pred, target)
    net.zero_grad()
    loss.backward()

    # 对每个参数，计算数值梯度并对比
    all_correct = True
    for i, param in enumerate(net.parameters()):
        # 定义损失函数（闭包）
        def loss_fn():
            p = net(x)
            return mse_loss(p, target).data

        num_grad = numerical_gradient(param, loss_fn)
        ana_grad = param.grad

        # 计算相对误差
        diff = abs(num_grad - ana_grad)
        denom = max(abs(num_grad), abs(ana_grad), 1e-8)
        relative_error = diff / denom

        if relative_error > tolerance:
            print(f"  参数 {i}: 数值梯度={num_grad:.6f}, 解析梯度={ana_grad:.6f}, "
                  f"相对误差={relative_error:.2e} ❌")
            all_correct = False

    if all_correct:
        print("  所有参数的解析梯度与数值梯度一致 ✓")
    return all_correct


# === 演示 1：梯度检查 ===
def demo_gradient_check():
    print("=" * 60)
    print("演示 1：梯度检查（数值梯度 vs 解析梯度）")
    print("=" * 60)

    random.seed(42)
    net = Network([2, 3, 1])

    x = [Value(0.5), Value(-0.3)]
    target = Value(0.8)

    print("\n网络结构: 2 → 3 → 1")
    print(f"输入: [{x[0].data}, {x[1].data}]")
    print(f"目标: {target.data}")
    print("\n梯度检查结果:")
    gradient_check(net, x, target)


# === 演示 2：XOR 训练 ===
def demo_xor():
    print("\n" + "=" * 60)
    print("演示 2：XOR 问题训练")
    print("=" * 60)

    random.seed(42)
    net = Network([2, 4, 1])

    xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    learning_rate = 1.0

    for epoch in range(1000):
        total_loss = Value(0.0)
        for inputs, target in xor_data:
            x = [Value(i) for i in inputs]
            pred = net(x)
            loss = mse_loss(pred, target)
            total_loss = total_loss + loss

        net.zero_grad()
        total_loss.backward()

        for p in net.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 200 == 0:
            print(f"  轮次 {epoch:4d} | 损失: {total_loss.data:.6f}")

    print("\nXOR 预测结果:")
    for inputs, target in xor_data:
        x = [Value(i) for i in inputs]
        pred = net(x)
        predicted_class = 1 if pred.data > 0.5 else 0
        status = "✓" if predicted_class == int(target) else "✗"
        print(f"  输入 {inputs} → 预测 {pred.data:.4f} (取整: {predicted_class}, "
              f"期望 {int(target)}) {status}")


# === 演示 3：圆形分类 ===
def demo_circle():
    print("\n" + "=" * 60)
    print("演示 3：圆形决策边界分类")
    print("=" * 60)

    random.seed(7)

    def generate_circle_data(n=80):
        data = []
        for _ in range(n):
            x1 = random.uniform(-1.5, 1.5)
            x2 = random.uniform(-1.5, 1.5)
            label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
            data.append(([x1, x2], label))
        return data

    circle_data = generate_circle_data(80)
    net = Network([2, 8, 1])
    learning_rate = 0.5

    for epoch in range(2000):
        random.shuffle(circle_data)
        total_loss_val = 0.0
        for inputs, target in circle_data:
            x = [Value(i) for i in inputs]
            pred = net(x)
            loss = mse_loss(pred, target)
            net.zero_grad()
            loss.backward()
            for p in net.parameters():
                p.data -= learning_rate * p.grad
            total_loss_val += loss.data

        if epoch % 500 == 0:
            correct = 0
            for inputs, target in circle_data:
                x = [Value(i) for i in inputs]
                pred = net(x)
                predicted_class = 1.0 if pred.data > 0.5 else 0.0
                if predicted_class == target:
                    correct += 1
            accuracy = correct / len(circle_data) * 100
            print(f"  轮次 {epoch:4d} | 损失: {total_loss_val:.4f} | 准确率: {accuracy:.1f}%")

    print("\n测试点预测:")
    test_points = [
        ([0.0, 0.0], "内部"),
        ([0.5, 0.5], "内部"),
        ([1.2, 1.2], "外部"),
        ([0.0, 1.2], "外部"),
        ([-0.3, 0.3], "内部"),
    ]
    for point, expected_region in test_points:
        x = [Value(i) for i in point]
        pred = net(x)
        predicted = "内部" if pred.data > 0.5 else "外部"
        status = "✓" if predicted == expected_region else "✗"
        print(f"  点 {point} → 预测 {pred.data:.4f} ({predicted}, 期望 {expected_region}) {status}")


# === 演示 4：梯度消失 ===
def demo_vanishing_gradient():
    print("\n" + "=" * 60)
    print("演示 4：梯度消失问题")
    print("=" * 60)

    random.seed(123)

    # 使用 sigmoid 的深层网络
    deep_net = Network([2, 4, 4, 4, 1])

    x = [Value(0.5), Value(-0.3)]
    target = Value(0.8)

    pred = deep_net(x)
    loss = mse_loss(pred, target)
    deep_net.zero_grad()
    loss.backward()

    # 统计各层梯度的平均绝对值
    layer_idx = 0
    param_idx = 0
    layer_grads = {}
    for layer in deep_net.layers:
        grads = []
        for n in layer.neurons:
            for w in n.weights:
                grads.append(abs(w.grad))
            grads.append(abs(n.bias.grad))
        layer_grads[layer_idx] = sum(grads) / len(grads)
        layer_idx += 1

    print("\n各层梯度平均绝对值（sigmoid 网络，4 个隐藏层）:")
    for layer_idx, avg_grad in layer_grads.items():
        print(f"  层 {layer_idx}: {avg_grad:.6f}")

    print("\n观察：越靠前的层，梯度越小（梯度消失）")


# === 演示 5：Sigmoid vs ReLU 收敛速度对比 ===
def demo_activation_comparison():
    print("\n" + "=" * 60)
    print("演示 5：Sigmoid vs ReLU 收敛速度对比")
    print("=" * 60)

    random.seed(42)
    xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    # Sigmoid 网络
    sigmoid_net = Network([2, 8, 1], activations=["sigmoid", "sigmoid"])
    # ReLU 网络（隐藏层 ReLU，输出层 sigmoid）
    relu_net = Network([2, 8, 1], activations=["relu", "sigmoid"])

    learning_rate = 1.0

    print("\nSigmoid 网络训练:")
    for epoch in range(1000):
        total_loss = Value(0.0)
        for inputs, target in xor_data:
            x = [Value(i) for i in inputs]
            pred = sigmoid_net(x)
            loss = mse_loss(pred, target)
            total_loss = total_loss + loss
        sigmoid_net.zero_grad()
        total_loss.backward()
        for p in sigmoid_net.parameters():
            p.data -= learning_rate * p.grad
        if epoch % 200 == 0:
            print(f"  轮次 {epoch:4d} | 损失: {total_loss.data:.6f}")

    print("\nReLU 网络训练:")
    for epoch in range(1000):
        total_loss = Value(0.0)
        for inputs, target in xor_data:
            x = [Value(i) for i in inputs]
            pred = relu_net(x)
            loss = mse_loss(pred, target)
            total_loss = total_loss + loss
        relu_net.zero_grad()
        total_loss.backward()
        for p in relu_net.parameters():
            p.data -= learning_rate * p.grad
        if epoch % 200 == 0:
            print(f"  轮次 {epoch:4d} | 损失: {total_loss.data:.6f}")

    print("\n观察：ReLU 网络收敛更快，因为梯度不会在正区间消失")


if __name__ == "__main__":
    demo_gradient_check()
    demo_xor()
    demo_circle()
    demo_vanishing_gradient()
    demo_activation_comparison()
