# main.py — 从零实现七种优化器并进行数值对比
# 依赖：math, random（标准库，无第三方依赖）
# 对应课程：阶段 03 · 06（优化器）

import math
import random


# ==============================================================================
# 第一部分：七种优化器的从零实现
# ==============================================================================

class SGD:
    """随机梯度下降（Stochastic Gradient Descent）

    最基础的优化器：沿梯度的反方向更新参数。
    w = w - lr * gradient
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class SGDMomentum:
    """带动量的 SGD

    维护速度的指数移动平均：m_t = beta * m_{t-1} + gradient
    动量可以抑制震荡方向、加速一致方向，实现更快的收敛。
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]


class NesterovMomentum:
    """Nesterov 加速梯度

    与标准动量的区别：先按动量"预看"一步，在预看位置计算梯度。
    这个"先走再看"的策略提供了更好的矫正能力，通常比标准动量收敛更快。
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)

        for i in range(len(params)):
            # 先按旧动量更新到"预看位置"
            lookahead = params[i] - self.lr * self.beta * self.velocities[i]
            # 在预看位置计算梯度的效应（这里简化为直接使用当前梯度，
            # 并在此基础上加入动量）
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]


class AdaGrad:
    """自适应梯度算法

    核心思想：为每个参数维护梯度平方的累积和，
    梯度大的参数学习率被自动缩小，梯度小的参数学习率相对放大。
    适用于稀疏数据（如 NLP 中的词嵌入训练）。
    缺点：学习率单调递减，训练后期可能过小。
    """

    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.sum_squares = None

    def step(self, params, grads):
        if self.sum_squares is None:
            self.sum_squares = [0.0] * len(params)
        for i in range(len(params)):
            self.sum_squares[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (math.sqrt(self.sum_squares[i]) + self.epsilon)


class RMSProp:
    """均方根传播

    AdaGrad 的改进版：用指数移动平均替代累积平方和，
    解决了学习率单调递减的问题。beta 控制历史梯度的记忆长度。
    由 Hinton 在 Coursera 课程中提出（未正式发表）。
    """

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = [0.0] * len(params)
        for i in range(len(params)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grads[i] ** 2
            params[i] -= self.lr * grads[i] / (math.sqrt(self.s[i]) + self.epsilon)


class Adam:
    """自适应矩估计 = 动量 + RMSProp + 偏差校正

    同时维护梯度的一阶矩（均值，处理震荡）和二阶矩（方差，处理自适应学习率），
    并通过偏差校正补偿前几步的冷启动偏差。
    默认超参：lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩（梯度均值）
        self.v = None  # 二阶矩（梯度平方均值）
        self.t = 0     # 时间步计数器

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            # 更新一阶矩（动量）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # 更新二阶矩（自适应学习率）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            # 偏差校正：补偿零初始化带来的偏差
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)


class AdamW:
    """解耦权重衰减的 Adam

    与 Adam + L2 正则化的关键区别：
    Adam + L2 会将正则化项通过自适应学习率缩放，导致不同参数的正则化强度不同。
    AdamW 将权重衰减直接作用于参数，每个参数获得均匀的正则化。
    这是训练 Transformer、大语言模型的默认优化器。
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 先做 Adam 更新
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            # 再做解耦权重衰减（直接作用于参数，不经过梯度）
            params[i] -= self.lr * self.weight_decay * params[i]


# ==============================================================================
# 第二部分：辅助函数
# ==============================================================================

def sigmoid(x):
    """Sigmoid 激活函数，带数值裁剪防止溢出。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    """生成环形二分类数据集。

    圆内（x^2 + y^2 < 1.5）为正类，圆外为负类。
    这个数据集线性不可分，需要非线性模型才能正确分类。
    """
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


# ==============================================================================
# 第三部分：训练用神经网络（手动实现前向传播和反向传播）
# ==============================================================================

class OptimizerTestNetwork:
    """两层前馈网络：2 → hidden_size → 1

    用于对比不同优化器的训练效果。
    结构：线性层1 → ReLU → 线性层2 → Sigmoid → 二分类输出
    """

    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)  # 固定种子确保不同优化器从相同初始化出发
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        # 第一层权重和偏置
        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        # 第二层权重和偏置
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        """将所有参数展平为一维列表，便于优化器统一处理。"""
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        """将一维参数列表还原为网络结构。"""
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        """前向传播，缓存中间变量供反向传播使用。"""
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))  # ReLU 激活

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        """反向传播计算所有参数的梯度。

        损失函数：二元交叉熵
        梯度流：d_loss → d_sigmoid → d_out → d_w2, d_h → d_w1, d_b1
        """
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        # 交叉熵对 sigmoid 输出的梯度
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0

        # 第一层权重的梯度
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        # 第一层偏置的梯度
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        # 第二层权重的梯度
        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        # 第二层偏置的梯度
        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        """训练网络并记录每轮的损失和准确率。"""
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


# ==============================================================================
# 第四部分：偏差校正演示
# ==============================================================================

def bias_correction_demo():
    """展示 Adam 偏差校正如何补偿零初始化的冷启动问题。

    在训练初期，一阶矩和二阶矩都被初始化为 0，
    除以 (1 - beta^t) 可以将早期的有偏估计拉回到正确值。
    """
    beta1 = 0.9
    beta2 = 0.999
    gradient = 1.0  # 假设梯度恒为 1，便于观察偏差校正效果

    print("  Step | m_raw  | m_corrected | v_raw    | v_corrected")
    print("  " + "-" * 55)

    m = 0.0
    v = 0.0
    for t in range(1, 11):
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        print(f"  {t:4d} | {m:.4f} | {m_hat:.4f}      | {v:.6f} | {v_hat:.6f}")


# ==============================================================================
# 第五部分：Rosenbrock 函数上的数值对比
# ==============================================================================

def rosenbrock(x, y):
    """Rosenbrock 函数（香蕉函数）：f(x,y) = (1-x)^2 + 100*(y-x^2)^2

    全局最小值在 (1, 1) 处，f(1,1) = 0。
    这个函数的特殊之处：有一个狭窄的弯曲山谷，
    梯度方向与山谷方向几乎垂直，导致普通梯度下降反复震荡。
    是检验优化器性能的经典测试函数。
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    """Rosenbrock 函数的梯度。

    df/dx = -2(1-x) - 400x(y-x^2)
    df/dy = 200(y-x^2)
    """
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return [dx, dy]


def rosenbrock_comparison():
    """在 Rosenbrock 函数上对比七种优化器的收敛行为。

    所有优化器从同一点 (-1, 1) 出发，经过 5000 步优化，
    对比最终到达的位置和损失值。
    """
    optimizers = {
        "SGD (lr=0.001)": SGD(lr=0.001),
        "SGD+Momentum (lr=0.0005)": SGDMomentum(lr=0.0005, beta=0.9),
        "Nesterov (lr=0.0005)": NesterovMomentum(lr=0.0005, beta=0.9),
        "AdaGrad (lr=0.05)": AdaGrad(lr=0.05),
        "RMSProp (lr=0.001)": RMSProp(lr=0.001, beta=0.9),
        "Adam (lr=0.001)": Adam(lr=0.001),
        "AdamW (lr=0.001, wd=0.001)": AdamW(lr=0.001, weight_decay=0.001),
    }

    steps = 5000
    print(f"\n  起始点: (-1, 1)，目标点: (1, 1)，运行 {steps} 步")
    print(f"  {'优化器':<35} {'最终 x':>8} {'最终 y':>8} {'最终损失':>12}")
    print("  " + "-" * 65)

    for name, opt in optimizers.items():
        params = [-1.0, 1.0]
        for step in range(steps):
            grads = rosenbrock_grad(params[0], params[1])
            opt.step(params, grads)
        final_loss = rosenbrock(params[0], params[1])
        print(f"  {name:<35} {params[0]:>8.4f} {params[1]:>8.4f} {final_loss:>12.6f}")


# ==============================================================================
# 第六部分：权重衰减效果对比
# ==============================================================================

def weight_decay_demo():
    """对比 Adam 和 AdamW 的权重衰减效果。

    Adam + L2 正则化中，自适应学习率会缩放正则化项，
    导致梯度方差大的参数被正则化得更少。
    AdamW 将权重衰减解耦，所有参数获得均匀的正则化。
    """
    random.seed(42)
    large_weights = [random.uniform(-5, 5) for _ in range(10)]
    weights_adam = list(large_weights)
    weights_adamw = list(large_weights)

    opt_adam = Adam(lr=0.001)
    opt_adamw = AdamW(lr=0.001, weight_decay=0.1)

    initial_norm = math.sqrt(sum(w * w for w in large_weights))
    print(f"  初始权重 L2 范数: {initial_norm:.4f}")

    for step in range(100):
        # 随机梯度（模拟训练中的噪声）
        grads = [random.gauss(0, 0.1) for _ in range(10)]
        opt_adam.step(weights_adam, list(grads))
        opt_adamw.step(weights_adamw, list(grads))

    norm_adam = math.sqrt(sum(w * w for w in weights_adam))
    norm_adamw = math.sqrt(sum(w * w for w in weights_adamw))
    print(f"  100 步后：")
    print(f"    Adam  权重 L2 范数: {norm_adam:.4f}")
    print(f"    AdamW 权重 L2 范数: {norm_adamw:.4f}")
    print(f"    AdamW 将权重收缩了 {norm_adam / max(0.001, norm_adamw):.1f} 倍")


# ==============================================================================
# 第七部分：主程序
# ==============================================================================

if __name__ == "__main__":
    # ---- 实验 1：SGD 在简单函数上的收敛 ----
    print("=" * 60)
    print("实验 1：SGD 最小化 f(x) = (x-3)^2")
    print("=" * 60)
    print("  从 x=10 出发，目标 x=3")
    x = [10.0]
    sgd = SGD(lr=0.1)
    for step in range(20):
        grad = [2.0 * (x[0] - 3.0)]
        sgd.step(x, grad)
        loss = (x[0] - 3.0) ** 2
        if step % 5 == 0 or step == 19:
            print(f"    Step {step:2d}: x={x[0]:.6f}, loss={loss:.6f}")

    # ---- 实验 2：Adam 偏差校正演示 ----
    print("\n" + "=" * 60)
    print("实验 2：Adam 偏差校正")
    print("=" * 60)
    print("  观察前几步 m_raw 与 m_corrected 的差距")
    bias_correction_demo()

    # ---- 实验 3：七种优化器在环形数据集上的对比 ----
    print("\n" + "=" * 60)
    print("实验 3：优化器在环形数据集上的训练对比")
    print("=" * 60)
    data = make_circle_data()

    configs = [
        ("SGD (lr=0.05)", SGD(lr=0.05)),
        ("SGD+Momentum (lr=0.05, beta=0.9)", SGDMomentum(lr=0.05, beta=0.9)),
        ("Nesterov (lr=0.05, beta=0.9)", NesterovMomentum(lr=0.05, beta=0.9)),
        ("AdaGrad (lr=0.05)", AdaGrad(lr=0.05)),
        ("RMSProp (lr=0.001)", RMSProp(lr=0.001)),
        ("Adam (lr=0.001)", Adam(lr=0.001)),
        ("AdamW (lr=0.001, wd=0.01)", AdamW(lr=0.001, weight_decay=0.01)),
    ]

    results = {}
    for name, opt in configs:
        print(f"\n--- {name} ---")
        net = OptimizerTestNetwork(opt, hidden_size=8)
        history = net.train(data, epochs=300)
        results[name] = history

    # 汇总对比
    print("\n" + "=" * 60)
    print("最终对比")
    print("=" * 60)
    for name, history in results.items():
        final_loss, final_acc = history[-1]
        first_85 = None
        for epoch, (loss, acc) in enumerate(history):
            if acc >= 85.0:
                first_85 = epoch
                break
        reached = f"epoch {first_85}" if first_85 is not None else "未达到"
        print(f"  {name:42s}: acc={final_acc:.1f}%, loss={final_loss:.4f}, 达到85%: {reached}")

    # ---- 实验 4：权重衰减效果 ----
    print("\n" + "=" * 60)
    print("实验 4：Adam vs AdamW 权重衰减效果")
    print("=" * 60)
    weight_decay_demo()

    # ---- 实验 5：Rosenbrock 函数对比 ----
    print("\n" + "=" * 60)
    print("实验 5：Rosenbrock 函数优化对比")
    print("=" * 60)
    rosenbrock_comparison()
