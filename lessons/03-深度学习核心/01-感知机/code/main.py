# main.py — 感知机从零实现
# 依赖：无（仅使用 Python 标准库）
# 对应课程：阶段 03 · 01（感知机）
# 运行：python code/main.py

import math
import random


# ============================================================
# 第 1 步：感知机类（Perceptron）
# ============================================================

class Perceptron:
    """单层感知机——最简单的神经网络单元。

    感知机接收多个输入，计算加权和，加上偏置，通过阶跃函数输出 0 或 1。
    它是线性分类器，只能解决线性可分问题。
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1):
        # 权重初始化为 0，偏置初始化为 0
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs: list) -> int:
        """前向计算：计算加权和，通过阶跃函数输出 0 或 1。"""
        # 计算加权和：w1*x1 + w2*x2 + ... + b
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        # 阶跃函数：>= 0 输出 1，否则输出 0
        return 1 if total >= 0 else 0

    def train(self, training_data: list, epochs: int = 100) -> None:
        """训练感知机：对每个错误样本更新权重和偏置。"""
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    # 权重更新规则：w_i = w_i + lr * error * x_i
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    # 偏置更新规则：b = b + lr * error
                    self.bias += self.lr * error
            # 如果本轮没有错误，说明已收敛，提前停止
            if errors == 0:
                print(f"  第 {epoch + 1} 轮收敛")
                return
        print(f"  {epochs} 轮后未收敛")


# ============================================================
# 第 2 步：训练逻辑门（AND、OR、NOT）
# ============================================================

def test_gate(name: str, n_inputs: int, data: list) -> None:
    """训练并测试一个逻辑门。"""
    print(f"=== {name} ===")
    p = Perceptron(n_inputs)
    p.train(data)
    print(f"  权重: {[f'{w:.2f}' for w in p.weights]}, 偏置: {p.bias:.2f}")
    for inputs, expected in data:
        result = p.predict(inputs)
        status = "正确" if result == expected else "错误"
        print(f"  输入 {inputs} -> 输出 {result} (期望 {expected}) {status}")
    print()


# 定义逻辑门的训练数据
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]


# ============================================================
# 第 3 步：观察 XOR 失败
# ============================================================

def demonstrate_xor_failure() -> None:
    """演示单层感知机无法学习 XOR。"""
    print("=== XOR 门（单层感知机——注定失败）===")
    p_xor = Perceptron(2)
    p_xor.train(xor_data, epochs=1000)
    for inputs, expected in xor_data:
        result = p_xor.predict(inputs)
        status = "正确" if result == expected else "错误"
        print(f"  输入 {inputs} -> 输出 {result} (期望 {expected}) {status}")
    print()


# ============================================================
# 第 4 步：用多层感知机解决 XOR（手工设置权重）
# ============================================================

def xor_network(x1: int, x2: int) -> int:
    """用 OR + NAND + AND 三个感知机组合实现 XOR。

    XOR = (x1 OR x2) AND NOT(x1 AND x2)
    这证明了多层网络可以解决非线性可分问题。
    """
    # OR 神经元：当任一输入为 1 时激活
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    # NAND 神经元：当两个输入不同时为 1 时激活
    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    # AND 神经元：当两个隐藏层输出都为 1 时激活
    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    # 前向传播：输入 -> 隐藏层 -> 输出
    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


def demonstrate_xor_network() -> None:
    """演示多层感知机成功解决 XOR。"""
    print("=== XOR 门（多层网络——手工权重）===")
    for inputs, expected in xor_data:
        result = xor_network(inputs[0], inputs[1])
        status = "正确" if result == expected else "错误"
        print(f"  输入 {inputs} -> 输出 {result} (期望 {expected}) {status}")
    print()


# ============================================================
# 第 5 步：用反向传播训练双层网络（自动学习权重）
# ============================================================

class TwoLayerNetwork:
    """双层神经网络：2 输入 -> 2 隐藏神经元 -> 1 输出。

    使用 Sigmoid 激活函数（可导），通过反向传播自动学习权重。
    这是从"手工设置权重"到"自动学习"的关键一步。
    """

    def __init__(self, learning_rate: float = 2.0):
        random.seed(0)
        # 隐藏层权重：2 个神经元，每个接收 2 个输入
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        # 输出层权重：1 个神经元，接收 2 个隐藏层输出
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x: float) -> float:
        """Sigmoid 激活函数：将实数映射到 (0, 1)。"""
        # 防止 exp 溢出
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs: list) -> float:
        """前向传播：计算网络输出。"""
        self.inputs = inputs
        # 计算隐藏层输出
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        # 计算输出层
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data: list, epochs: int = 10000) -> None:
        """反向传播训练：根据误差调整权重。"""
        for epoch in range(epochs):
            total_error = 0.0
            for inputs, target in training_data:
                # 前向传播
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                # 输出层梯度：d_loss/d_z = error * sigmoid'(z)
                # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
                d_output = error * output * (1 - output)

                # 反向传播到隐藏层
                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    # 隐藏层梯度 = 输出层梯度 * 权重 * sigmoid'(z)
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                # 更新输出层权重
                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                # 更新隐藏层权重
                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]

            # 每 2000 轮打印一次误差
            if epoch % 2000 == 0:
                print(f"  第 {epoch} 轮，误差: {total_error:.4f}")


def demonstrate_trained_network() -> None:
    """演示用反向传播训练的双层网络解决 XOR。"""
    print("=== XOR 门（双层网络——反向传播自动学习）===")
    net = TwoLayerNetwork(learning_rate=2.0)
    net.train(xor_data, epochs=10000)
    print()
    for inputs, expected in xor_data:
        result = net.forward(inputs)
        predicted = 1 if result >= 0.5 else 0
        status = "正确" if predicted == expected else "错误"
        print(f"  输入 {inputs} -> 输出 {result:.4f} (四舍五入: {predicted}, 期望 {expected}) {status}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("感知机教学演示")
    print("=" * 60)
    print()

    # 第 2 步：训练基本逻辑门
    test_gate("AND 门", 2, and_data)
    test_gate("OR 门", 2, or_data)
    test_gate("NOT 门", 1, not_data)

    # 第 3 步：观察 XOR 失败
    demonstrate_xor_failure()

    # 第 4 步：多层网络解决 XOR
    demonstrate_xor_network()

    # 第 5 步：反向传播自动学习
    demonstrate_trained_network()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
