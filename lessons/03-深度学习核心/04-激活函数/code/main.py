# main.py — 激活函数教学演示
# 依赖：numpy>=1.24, matplotlib>=3.7
# 安装：pip install numpy matplotlib
# 对应课程：阶段 03 · 04（激活函数）

import math
import random
import numpy as np

# ============================================================
# 第 1 部分：激活函数定义
# ============================================================

def sigmoid(x):
    """Sigmoid 函数：将输入压缩到 (0, 1)。"""
    # 截断防止 exp 溢出
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid 的导数：σ'(x) = σ(x)(1 - σ(x))。"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_act(x):
    """Tanh 函数：将输入压缩到 (-1, 1)。"""
    return math.tanh(x)


def tanh_derivative(x):
    """Tanh 的导数：tanh'(x) = 1 - tanh²(x)。"""
    t = math.tanh(x)
    return 1 - t * t


def relu(x):
    """ReLU 函数：负值归零，正值保持不变。"""
    return max(0.0, x)


def relu_derivative(x):
    """ReLU 的导数：正区间为 1，负区间为 0。"""
    return 1.0 if x > 0 else 0.0


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU：负区间保留微小梯度，缓解死亡 ReLU 问题。"""
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU 的导数：正区间为 1，负区间为 alpha。"""
    return 1.0 if x > 0 else alpha


def gelu(x):
    """GELU 函数：高斯误差线性单元，Transformer 的默认选择。"""
    # 近似公式：0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + math.tanh(inner))


def gelu_derivative(x):
    """GELU 的导数：Φ(x) + x * φ(x)。"""
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf


def silu(x):
    """SiLU（Sigmoid Linear Unit）：x * σ(x)，也称 Swish-1。"""
    return x * sigmoid(x)


def silu_derivative(x):
    """SiLU 的导数：σ(x) + x * σ(x)(1 - σ(x))。"""
    s = sigmoid(x)
    return s + x * s * (1 - s)


def swish(x, beta=1.0):
    """Swish 函数：x * σ(βx)，β=1 时等价于 SiLU。"""
    return x * sigmoid(beta * x)


def softmax(xs):
    """Softmax 函数：将向量转换为概率分布。"""
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]  # 减去最大值防止溢出
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================
# 第 2 部分：梯度死区扫描
# ============================================================

def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    """扫描一个函数的梯度，统计"死区"（梯度接近 0）的比例。"""
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"  {name:15s}: {healthy:3d} 健康, {near_zero:3d} 近零 ({pct_dead:.0f}% 死区)")


# ============================================================
# 第 3 部分：梯度消失实验
# ============================================================

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    """模拟信号通过多层网络后的幅度变化，展示梯度消失。"""
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n  使用 {name} 通过 {n_layers} 层:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"    层 {layer + 1:2d}: 幅度 = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs


# ============================================================
# 第 4 部分：死亡神经元检测
# ============================================================

def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    """检测 ReLU 网络中"死亡"的神经元（从未被激活）。"""
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\n  死亡神经元报告 ({hidden_size} 个神经元, {n_samples} 个样本):")
    print(f"    死亡 (从未激活):  {dead}")
    print(f"    濒死 (< 5% 激活): {rarely_fire}")
    print(f"    健康:             {healthy}")
    print(f"    死亡比例:         {dead / hidden_size * 100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "死亡" if c == 0 else "濒死" if c < n_samples * 0.05 else "健康"
        bar = "#" * (c * 40 // n_samples)
        print(f"    神经元 {i:2d}: {c:4d}/{n_samples} 次激活 [{status:4s}] {bar}")


# ============================================================
# 第 5 部分：不同激活函数在分类任务上的训练对比
# ============================================================

def make_circle_data(n=200, seed=42):
    """生成圆形二分类数据集：圆内为 1，圆外为 0。"""
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    """一个简单的两层神经网络，用于对比不同激活函数。"""

    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        # 第一层：输入(2) → 隐藏层(hidden_size)
        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        # 第二层：隐藏层(hidden_size) → 输出(1)
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        """前向传播。"""
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)  # 输出层用 Sigmoid 产生概率
        return self.out

    def backward(self, target):
        """反向传播 + 参数更新。"""
        error = self.out - target
        # 输出层梯度（Sigmoid 导数）
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            # 隐藏层梯度 = 输出梯度 × w2 × 激活函数导数
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            # 更新第二层权重
            self.w2[i] -= self.lr * d_out * self.h[i]
            # 更新第一层权重
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        """训练网络，返回每轮的损失。"""
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    轮次 {epoch:3d}: 损失={avg_loss:.4f}, 准确率={accuracy:.1f}%")
        return losses


# ============================================================
# 第 6 部分：可视化（使用 matplotlib）
# ============================================================

def plot_activations():
    """绘制所有激活函数的曲线和导数曲线。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("  [跳过可视化] 未安装 matplotlib，运行 pip install matplotlib 后可查看图表")
        return

    x = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('激活函数及其导数', fontsize=14)

    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative),
        ('Tanh', tanh_act, tanh_derivative),
        ('ReLU', relu, relu_derivative),
        ('Leaky ReLU', leaky_relu, leaky_relu_derivative),
        ('GELU', gelu, gelu_derivative),
        ('SiLU/Swish', silu, silu_derivative),
    ]

    for idx, (name, func, deriv) in enumerate(activations):
        ax = axes[idx // 4][idx % 4]
        y_vals = [func(xi) for xi in x]
        d_vals = [deriv(xi) for xi in x]
        ax.plot(x, y_vals, 'b-', label='f(x)', linewidth=2)
        ax.plot(x, d_vals, 'r--', label="f'(x)", linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    # 最后一个子图：Softax 示例
    ax = axes[1][3]
    logits = np.linspace(-3, 3, 50)
    softmax_vals = np.array([softmax([float(xi), 0.0]) for xi in logits])
    ax.plot(logits, softmax_vals[:, 0], 'b-', label='softmax(x, 0)[0]', linewidth=2)
    ax.plot(logits, softmax_vals[:, 1], 'r-', label='softmax(x, 0)[1]', linewidth=2)
    ax.set_title('Softmax (二元)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    print("  图表已保存为 activation_functions.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第 1 步：激活函数值对比")
    print("=" * 60)
    test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    print(f"  {'x':>5s}  {'Sigmoid':>8s}  {'Tanh':>8s}  {'ReLU':>8s}  {'GELU':>8s}  {'SiLU':>8s}")
    for x in test_points:
        print(f"  {x:5.1f}  {sigmoid(x):8.4f}  {tanh_act(x):8.4f}  "
              f"{relu(x):8.4f}  {gelu(x):8.4f}  {silu(x):8.4f}")

    print(f"\n  softmax([2.0, 1.0, 0.1]) = [{', '.join(f'{v:.4f}' for v in softmax([2.0, 1.0, 0.1]))}]")
    print(f"  softmax([10, 10, 10])     = [{', '.join(f'{v:.4f}' for v in softmax([10.0, 10.0, 10.0]))}]")

    print("\n" + "=" * 60)
    print("第 2 步：梯度死区扫描")
    print("=" * 60)
    gradient_scan("Sigmoid", sigmoid_derivative)
    gradient_scan("Tanh", tanh_derivative)
    gradient_scan("ReLU", relu_derivative)
    gradient_scan("Leaky ReLU", leaky_relu_derivative)
    gradient_scan("GELU", gelu_derivative)
    gradient_scan("SiLU", silu_derivative)

    print("\n" + "=" * 60)
    print("第 3 步：梯度消失实验")
    print("=" * 60)
    vanishing_gradient_experiment(sigmoid, "Sigmoid")
    vanishing_gradient_experiment(relu, "ReLU")
    vanishing_gradient_experiment(gelu, "GELU")

    print("\n" + "=" * 60)
    print("第 4 步：死亡神经元检测")
    print("=" * 60)
    dead_neuron_detector()

    print("\n" + "=" * 60)
    print("第 5 步：训练对比（圆形数据集）")
    print("=" * 60)
    data = make_circle_data()

    configs = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("ReLU", relu, relu_derivative),
        ("GELU", gelu, gelu_derivative),
    ]

    results = {}
    for name, act_fn, act_d_fn in configs:
        print(f"\n  --- 使用 {name} 训练 ---")
        net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
        losses = net.train(data, epochs=200)
        results[name] = losses

    print("\n  === 最终损失对比 ===")
    for name, losses in results.items():
        improvement = (1 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0
        print(f"    {name:10s}: 起始={losses[0]:.4f} → 最终={losses[-1]:.4f} (改善: {improvement:.1f}%)")

    print("\n" + "=" * 60)
    print("第 6 步：可视化")
    print("=" * 60)
    plot_activations()

    print("\n全部演示完成。")
