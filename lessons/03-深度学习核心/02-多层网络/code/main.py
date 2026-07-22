# main.py — 从零实现多层网络的前向传播
# 依赖：Python 3.10+ 标准库（无需第三方库）
# 对应课程：阶段 03 · 02（多层网络）

import math
import random


# === 第 1 步：Sigmoid 激活函数 ===
def sigmoid(x):
    """Sigmoid 激活函数，将任意实数压缩到 (0, 1)。

     clamp 到 [-500, 500] 防止 math.exp 溢出。
    """
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# === 第 2 步：层（Layer）===
class Layer:
    """全连接层（线性变换 + Sigmoid 激活）。

    权重矩阵形状为 (n_neurons, n_inputs)：
    - 每一行是一个神经元对所有输入的权重
    - 列数 = 输入维度，行数 = 神经元数量
    """

    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights  # 手动指定权重（用于演示）
        else:
            # 随机初始化：均匀分布 [-1, 1)
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        """前向传播：z = W·x + b，然后 a = sigmoid(z)。"""
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            # 加权求和
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            # 激活
            self.last_output.append(sigmoid(z))
        return self.last_output


# === 第 3 步：网络（Network）===
class Network:
    """多层网络：将多个 Layer 按顺序串联。

    前向传播就是逐层传递：第 k 层的输出作为第 k+1 层的输入。
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        """完整前向传播：数据从输入层流经所有隐藏层到达输出层。"""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def count_parameters(self):
        """统计可训练参数总数（权重 + 偏置）。"""
        total = 0
        for layer in self.layers:
            for neuron_weights in layer.weights:
                total += len(neuron_weights)  # 权重数量
            total += len(layer.biases)          # 偏置数量
        return total


# === 第 4 步：XOR 问题（手工调参的 2-2-1 网络）===
def demo_xor():
    """用 2-2-1 网络解决 XOR 问题。

    隐藏层第一个神经元近似 OR，第二个近似 NAND，
    输出层将两者组合为 AND，即 XOR。
    大权重（20）让 Sigmoid 接近阶跃函数。
    """
    print("=" * 60)
    print("DEMO 1：XOR 问题 — 2-2-1 网络（手工调参）")
    print("=" * 60)

    hidden = Layer(
        n_inputs=2,
        n_neurons=2,
        weights=[[20.0, 20.0], [-20.0, -20.0]],
        biases=[-10.0, 30.0],
    )
    output = Layer(
        n_inputs=2,
        n_neurons=1,
        weights=[[20.0, 20.0]],
        biases=[-30.0],
    )
    xor_net = Network([hidden, output])

    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    all_correct = True
    for inputs, expected in xor_data:
        result = xor_net.forward(inputs)
        predicted = 1 if result[0] >= 0.5 else 0
        status = "OK" if predicted == expected else "WRONG"
        if predicted != expected:
            all_correct = False
        print(f"  {inputs} -> {result[0]:.6f} (预测: {predicted}, 期望: {expected}) {status}")

    print(f"\n  XOR 解决: {all_correct}")
    print(f"  参数量: {xor_net.count_parameters()}")


# === 第 5 步：圆形分类（2-8-1 网络，随机权重）===
def demo_circle():
    """用 2-8-1 网络对平面点做圆形边界分类。

    随机权重下准确率很低——这恰恰说明前向传播只是计算，
    学习权重是下一课反向传播的任务。
    """
    print()
    print("=" * 60)
    print("DEMO 2：圆形分类 — 2-8-1 网络（随机权重）")
    print("=" * 60)

    random.seed(42)
    data = []
    for _ in range(200):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if (x * x + y * y) < 0.25 else 0
        data.append(([x, y], label))

    inside_count = sum(1 for _, label in data if label == 1)
    outside_count = len(data) - inside_count
    print(f"  数据集: {len(data)} 个点（圆内 {inside_count}，圆外 {outside_count}）")

    random.seed(7)
    circle_net = Network([
        Layer(n_inputs=2, n_neurons=8),
        Layer(n_inputs=8, n_neurons=1),
    ])

    correct = 0
    for inputs, expected in data:
        result = circle_net.forward(inputs)
        predicted = 1 if result[0] >= 0.5 else 0
        if predicted == expected:
            correct += 1

    print(f"  随机权重准确率: {correct}/{len(data)} ({100 * correct / len(data):.1f}%)")
    print(f"  参数量: {circle_net.count_parameters()}")
    print(f"  （随机权重准确率低——需要训练，下节课反向传播解决）")


# === DEMO 3：前向传播内部状态追踪 ===
def demo_forward_trace():
    """追踪 XOR 网络每一层的中间输出，理解数据如何变换。"""
    print()
    print("=" * 60)
    print("DEMO 3：前向传播内部状态追踪")
    print("=" * 60)

    hidden = Layer(
        n_inputs=2, n_neurons=2,
        weights=[[20.0, 20.0], [-20.0, -20.0]],
        biases=[-10.0, 30.0],
    )
    output = Layer(
        n_inputs=2, n_neurons=1,
        weights=[[20.0, 20.0]],
        biases=[-30.0],
    )
    xor_net = Network([hidden, output])

    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    for inputs, expected in xor_data:
        xor_net.forward(inputs)
        h = xor_net.layers[0].last_output
        o = xor_net.layers[1].last_output
        print(f"  输入: {inputs}")
        print(f"    隐藏层: [{h[0]:.6f}, {h[1]:.6f}]")
        print(f"    输出层: {o[0]:.6f} -> {'1' if o[0] >= 0.5 else '0'} (期望: {expected})")


# === DEMO 4：经典架构参数量统计 ===
def demo_parameter_count():
    """统计几种经典网络架构的可训练参数数量。"""
    print()
    print("=" * 60)
    print("DEMO 4：经典架构参数量统计")
    print("=" * 60)

    architectures = [
        ("2-3-1（本课示例）", [2, 3, 1]),
        ("2-8-1（圆形分类）", [2, 8, 1]),
        ("784-256-128-10（MNIST）", [784, 256, 128, 10]),
        ("784-512-256-128-10（深层 MNIST）", [784, 512, 256, 128, 10]),
    ]

    for name, sizes in architectures:
        layers = []
        for i in range(1, len(sizes)):
            layers.append(Layer(n_inputs=sizes[i - 1], n_neurons=sizes[i]))
        net = Network(layers)
        print(f"  {name}: {net.count_parameters():,} 个参数")


# === 主程序 ===
if __name__ == "__main__":
    demo_xor()
    demo_circle()
    demo_forward_trace()
    demo_parameter_count()
