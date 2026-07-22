# main.py — 权重初始化策略教学演示
# 依赖：math, random（标准库，无需额外安装）
# 对应课程：阶段 03 · 08（权重初始化）

import math
import random

# ============================================================
# 第 1 部分：初始化策略实现
# ============================================================

def zero_init(fan_in, fan_out):
    """零初始化：所有权重为 0。

    问题：每个神经元计算相同的输出，梯度相同，更新相同。
    结果：网络退化为 1 个有效神经元。
    """
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    """标准随机初始化：从 N(0, scale) 中采样。

    scale 决定了权重的方差。scale=1 时方差为 1，
    对于 fan_in=512 的层，输出方差 = 512 × 1 = 512，信号会爆炸。
    """
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    """Xavier/Glorot 初始化：Var(w) = 2 / (fan_in + fan_out)。

    适用于 Sigmoid/Tanh 激活函数。保持前向和反向传播中的方差稳定。
    Glorot & Bengio, 2010.
    """
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    """Kaiming/He 初始化：Var(w) = 2 / fan_in。

    适用于 ReLU 系列激活函数。额外的因子 2 补偿了 ReLU 将一半输出置零的效果。
    He et al., 2015.
    """
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def orthogonal_init(fan_in, fan_out):
    """正交初始化：使用 SVD 分解生成正交矩阵。

    优点：每层输出方差恰好等于输入方差，梯度不会消失或爆炸。
    缺点：计算开销较大，通常用于 RNN/LSTM。
    """
    # 生成随机矩阵
    rows = max(fan_in, fan_out)
    cols = min(fan_in, fan_out)
    mat = [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]

    # 手动 SVD 的简化版本：Gram-Schmidt 正交化
    ortho = []
    for i in range(rows):
        vec = list(mat[i])
        for prev in ortho:
            # 减去在已正交向量上的投影
            dot = sum(v * p for v, p in zip(vec, prev))
            vec = [v - dot * p for v, p in zip(vec, prev)]
        # 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm < 1e-10:
            vec = [random.gauss(0, 0.01) for _ in range(cols)]
            norm = math.sqrt(sum(v * v for v in vec))
        ortho.append([v / norm for v in vec])

    # 取前 fan_out 行、前 fan_in 列
    return [row[:fan_in] for row in ortho[:fan_out]]


# ============================================================
# 第 2 部分：激活函数
# ============================================================

def sigmoid(x):
    """Sigmoid：将输入压缩到 (0, 1)。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    """Tanh：将输入压缩到 (-1, 1)。"""
    return math.tanh(x)


def relu(x):
    """ReLU：负值归零，正值保持不变。"""
    return max(0.0, x)


# ============================================================
# 第 3 部分：50 层前向传播实验
# ============================================================

def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    """将随机数据通过 n_layers 层网络，记录每层的平均激活幅度。

    这是验证初始化策略是否有效的核心实验：
    - 幅度稳定 → 初始化正确
    - 幅度爆炸 → 初始化方差过大
    - 幅度消失 → 初始化方差过小
    """
    random.seed(42)
    layer_magnitudes = []

    # 初始输入：标准正态分布
    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        # 计算本层所有样本的平均绝对值
        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes


def magnitude_report(name, magnitudes):
    """打印激活幅度的逐层可视化报告。"""
    print(f"\n  {name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " EXPLODED"
            elif mag < 1e-6:
                bar = "." + " VANISHED"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"    Layer {i+1:3d}: {bar} ({mag:.6f})")


# ============================================================
# 第 4 部分：对称性问题演示
# ============================================================

def symmetry_demo():
    """演示零初始化的对称性问题。

    所有神经元计算相同的输出 → 梯度相同 → 更新相同。
    无论隐藏层有多宽，有效参数始终为 1。
    """
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("  对称性演示（4 个神经元，零初始化）:")
    for i, out in enumerate(outputs):
        print(f"    神经元 {i}: 输出 = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"    全部相同: {all_same}")
    print(f"    有效参数: 1（而非 {len(weights) * len(weights[0])}）")


# ============================================================
# 第 5 部分：方差传播分析
# ============================================================

def variance_analysis():
    """分析不同初始化策略的方差传播。

    核心公式：Var(z) = fan_in × Var(w) × Var(x)
    目标：让 Var(z) = Var(x)，即输入输出方差相等。
    """
    fan_in = 64
    n_trials = 10000

    configs = [
        ("Random N(0,1)", 1.0),
        ("Random N(0,0.01)", 0.01),
        ("Xavier std", math.sqrt(2.0 / (fan_in + fan_in))),
        ("Kaiming std", math.sqrt(2.0 / fan_in)),
    ]

    print("\n  方差分析（fan_in=64，单层）:")
    print(f"  {'策略':<25} {'权重方差':>12} {'输出方差':>12} {'输出/输入比':>12}")
    print("  " + "-" * 62)

    for name, std in configs:
        random.seed(42)
        output_vars = []
        for _ in range(n_trials):
            inputs = [random.gauss(0, 1) for _ in range(fan_in)]
            weights = [random.gauss(0, std) for _ in range(fan_in)]
            z = sum(w * x for w, x in zip(weights, inputs))
            output_vars.append(z * z)

        mean_output_var = sum(output_vars) / len(output_vars)
        weight_var = std * std
        ratio = mean_output_var
        print(f"  {name:<25} {weight_var:>12.6f} {mean_output_var:>12.4f} {ratio:>12.4f}")


# ============================================================
# 第 6 部分：主实验——50 层网络 + 初始化组合
# ============================================================

def run_experiment():
    """运行所有初始化策略 × 激活函数组合的 50 层前向传播实验。

    关键观察：
    - Zero + Sigmoid：信号稳定但无意义（所有神经元相同）
    - Random(1) + ReLU：信号爆炸
    - Random(0.01) + ReLU：信号消失
    - Xavier + Sigmoid/Tanh：信号稳定
    - Kaiming + ReLU：信号稳定
    """
    configs = [
        ("Zero + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("Random N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("Random N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"\n  {'策略':<28} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("  " + "-" * 78)

    all_results = {}
    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        all_results[name] = mags
        row = f"  {name:<28}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)

    return all_results


# ============================================================
# 第 7 部分：正交初始化对比实验
# ============================================================

def orthogonal_experiment():
    """对比正交初始化与 Kaiming 初始化在 ReLU 网络上的表现。"""
    configs = [
        ("Kaiming + ReLU", kaiming_init, relu),
        ("Orthogonal + ReLU", orthogonal_init, relu),
    ]

    print(f"\n  {'策略':<28} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("  " + "-" * 78)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"  {name:<28}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)


# ============================================================
# 第 8 部分：残差连接缩放演示（GPT-2 风格）
# ============================================================

def residual_scaling_demo():
    """演示 GPT-2 的残差缩放：1/sqrt(2N) 防止信号在 N 层后爆炸。

    残差连接：x = x + sublayer(x)
    每次加法都会增加方差，N 层后方差增长约 N 倍。
    缩放 sublayer 输出为 1/sqrt(2N)，保持信号稳定。
    """
    n_layers = 50
    width = 64
    random.seed(42)

    # 无缩放的残差流
    signal_no_scale = [random.gauss(0, 1) for _ in range(width)]
    mags_no_scale = []
    for layer in range(n_layers):
        sublayer_out = [random.gauss(0, 1) for _ in range(width)]
        signal_no_scale = [s + o for s, o in zip(signal_no_scale, sublayer_out)]
        avg_mag = sum(abs(s) for s in signal_no_scale) / width
        mags_no_scale.append(avg_mag)

    # 有缩放的残差流
    signal_scaled = [random.gauss(0, 1) for _ in range(width)]
    mags_scaled = []
    for layer in range(n_layers):
        scale = 1.0 / math.sqrt(2.0 * (layer + 1))
        sublayer_out = [random.gauss(0, 1) * scale for _ in range(width)]
        signal_scaled = [s + o for s, o in zip(signal_scaled, sublayer_out)]
        avg_mag = sum(abs(s) for s in signal_scaled) / width
        mags_scaled.append(avg_mag)

    print(f"\n  残差缩放对比（{n_layers} 层）:")
    print(f"  {'层':>5s}  {'无缩放':>12s}  {'有缩放 (1/√2N)':>16s}")
    print("  " + "-" * 38)
    for i in [0, 9, 19, 29, 39, 49]:
        print(f"  {i+1:5d}  {mags_no_scale[i]:12.4f}  {mags_scaled[i]:16.4f}")

    print(f"\n  第 {n_layers} 层: 无缩放={mags_no_scale[-1]:.4f}, 有缩放={mags_scaled[-1]:.4f}")
    print(f"  无缩放增长倍数: {mags_no_scale[-1] / mags_no_scale[0]:.2f}x")
    print(f"  有缩放增长倍数: {mags_scaled[-1] / mags_scaled[0]:.2f}x")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("第 1 步：对称性问题——零初始化")
    print("=" * 70)
    symmetry_demo()

    print("\n" + "=" * 70)
    print("第 2 步：方差传播分析")
    print("=" * 70)
    variance_analysis()

    print("\n" + "=" * 70)
    print("第 3 步：50 层前向传播实验")
    print("=" * 70)
    all_results = run_experiment()

    print("\n" + "=" * 70)
    print("第 4 步：逐层激活幅度报告")
    print("=" * 70)
    for name, mags in all_results.items():
        magnitude_report(name, mags)

    print("\n" + "=" * 70)
    print("第 5 步：正交初始化对比")
    print("=" * 70)
    orthogonal_experiment()

    print("\n" + "=" * 70)
    print("第 6 步：残差缩放演示（GPT-2 风格）")
    print("=" * 70)
    residual_scaling_demo()

    print("\n" + "=" * 70)
    print("全部演示完成。")
    print("=" * 70)
