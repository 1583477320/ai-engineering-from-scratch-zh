# 随机过程 — 从零实现核心算法
# 依赖：numpy>=1.24
# 对应课程：阶段 01 · 22（随机过程）
#
# 本文件涵盖以下随机过程的核心算法：
# 1. 一维/二维随机游走（Random Walk）
# 2. 马尔可夫链（Markov Chain）与平稳分布
# 3. 朗之万动力学（Langevin Dynamics）
# 4. 梅特罗波利斯-黑斯廷斯采样（Metropolis-Hastings MCMC）
# 5. 前向扩散过程（Forward Diffusion）

import numpy as np


# === 第 1 步：随机游走 ===

def random_walk_1d(n_steps: int, seed: int = None) -> np.ndarray:
    """一维随机游走。

    每一步等概率向左（-1）或向右（+1）移动。

    Args:
        n_steps: 步数
        seed: 随机种子（用于复现）

    Returns:
        长度为 n_steps + 1 的位置数组，positions[0] = 0

    理论：
        经过 n 步后，位置 S_n = X_1 + ... + X_n，
        E[S_n] = 0，Var(S_n) = n，标准差 = sqrt(n)。
    """
    rng = np.random.RandomState(seed)
    # 每步等概率取 +1 或 -1
    steps = rng.choice([-1, 1], size=n_steps)
    # 累积求和得到位置序列，第 0 步在原点
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps: int, seed: int = None) -> tuple:
    """二维随机游走。

    每一步等概率向上、下、左、右四个方向移动。

    Args:
        n_steps: 步数
        seed: 随机种子

    Returns:
        (x, y) 两个长度为 n_steps + 1 的位置数组
    """
    rng = np.random.RandomState(seed)
    # 0=右, 1=左, 2=上, 3=下
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1
    dx[directions == 1] = -1
    dy[directions == 2] = 1
    dy[directions == 3] = -1
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y


# === 第 2 步：马尔可夫链 ===

class MarkovChain:
    """离散时间马尔可夫链。

    系统在有限个状态之间按照固定概率转移。核心性质：下一状态只依赖当前状态（马尔可夫性质 / 无后效性）。

    Args:
        transition_matrix: 转移矩阵 P，P[i][j] = 从状态 i 转移到 j 的概率，
                           每一行求和为 1
        state_names: 状态名称列表（可选，用于可读输出）
    """

    def __init__(self, transition_matrix: list, state_names: list = None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]
        # 断言每行概率和为 1（允许浮点误差）
        assert np.allclose(self.P.sum(axis=1), 1.0), "转移矩阵每行概率和必须等于 1"

    def step(self, current_state: int, rng: np.random.RandomState = None) -> int:
        """从当前状态走出一步，按转移概率采样下一个状态。"""
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state: int, n_steps: int, seed: int = None) -> list:
        """模拟 n 步转移过程。

        Args:
            start_state: 起始状态编号
            n_steps: 模拟步数
            seed: 随机种子

        Returns:
            状态序列列表，长度为 n_steps + 1（包含起始状态）
        """
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self) -> np.ndarray:
        """计算平稳分布（解析法）。

        平稳分布 pi 满足 pi * P = pi，即 P^T 的特征值 1 对应的左特征向量。

        Returns:
            平稳分布向量，各元素之和为 1
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        # 找到离 1.0 最近的特征值索引
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        # 取绝对值并归一化（特征向量方向不唯一，可能有负分量）
        stationary = np.abs(stationary)
        total = stationary.sum()
        if total > 0:
            stationary = stationary / total
        return stationary

    def empirical_distribution(self, states: list) -> np.ndarray:
        """从状态序列中统计经验分布。

        Args:
            states: 模拟产生的状态序列

        Returns:
            各状态出现的频率
        """
        counts = np.zeros(self.n_states)
        for s in states:
            counts[s] += 1
        return counts / len(states)


# === 第 3 步：朗之万动力学 ===

def langevin_dynamics(
    grad_U, x0: np.ndarray, dt: float,
    temperature: float, n_steps: int, seed: int = None
) -> np.ndarray:
    """朗之万动力学采样。

    在能量函数 U(x) 的梯度下降中加入可控噪声，
    平衡时样本的分布正比于 exp(-U(x) / temperature)。

    Args:
        grad_U: 能量函数的梯度函数 grad_U(x) -> gradient array
        x0: 初始位置
        dt: 步长（离散化时间增量）
        temperature: 温度参数（控制探索强度）
        n_steps: 模拟步数
        seed: 随机种子

    Returns:
        轨迹数组，形状 (n_steps + 1, *x0.shape)

    更新公式：
        x_{t+1} = x_t - dt * grad_U(x_t) + sqrt(2 * T * dt) * z_t
        其中 z_t ~ N(0, 1)

    设计原理：
        - 梯度项 -dt * grad_U(x_t)：将粒子推向低能量区域（开发）
        - 噪声项 sqrt(2 * T * dt) * z_t：提供随机扰动（探索）
        - 温度 T = 0 时退化为纯梯度下降；温度极高时退化为随机游走
    """
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        # 梯度下降 + 噪声扰动
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)


# === 第 4 步：梅特罗波利斯-黑斯廷斯采样 ===

def metropolis_hastings(
    target_log_prob, proposal_std: float,
    x0: np.ndarray, n_samples: int, seed: int = None
) -> tuple:
    """梅特罗波利斯-黑斯廷斯算法（MCMC）。

    构造一个马尔可夫链，使其平稳分布等于目标分布 p(x)。
    只需要计算目标分布的对数值（无需归一化常数）。

    Args:
        target_log_prob: 目标分布的对数概率（无需归一化）
        proposal_std: 提议分布的标准差（高斯提议 N(x, std^2)）
        x0: 初始样本
        n_samples: 采样次数
        seed: 随机种子

    Returns:
        (samples, acceptance_rate)
        - samples: 样本数组，形状 (n_samples, *x0.shape)
        - acceptance_rate: 接受率（理想范围约 23%~50%）

    算法流程：
        1. 从当前位置 x 生成候选点 x' = x + N(0, std^2)
        2. 计算接受比 a = p(x') / p(x)
        3. 以概率 min(1, a) 接受 x'，否则保持在 x
    """
    if n_samples < 1:
        raise ValueError("n_samples 至少为 1")
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        # 生成候选点（对称高斯提议）
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        # 计算对数接受比（归一常数被抵消）
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        # 以概率 min(1, exp(log_ratio)) 接受候选点
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / max(n_samples - 1, 1)
    return np.array(samples), acceptance_rate


# === 第 5 步：前向扩散过程 ===

def diffusion_forward(
    signal: np.ndarray, n_steps: int,
    beta_start: float = 0.0001, beta_end: float = 0.02,
    seed: int = None
) -> tuple:
    """前向扩散过程。

    沿马尔可夫链逐步向信号添加高斯噪声，最终信号退化为纯噪声。
    这是扩散模型（如 DDPM）的前向过程。

    Args:
        signal: 原始信号（一维或多维）
        n_steps: 扩散步数
        beta_start: 噪声调度起始值
        beta_end: 噪声调度结束值
        seed: 随机种子

    Returns:
        (trajectory, betas)
        - trajectory: 信号退化轨迹，形状 (n_steps + 1, *signal.shape)
        - betas: 噪声调度数组，长度 n_steps

    更新公式：
        x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise_t

    经过 T 步后，x_T 近似为 N(0, I) 的纯高斯噪声。
    """
    rng = np.random.RandomState(seed)
    # 线性噪声调度
    betas = np.linspace(beta_start, beta_end, n_steps)
    trajectory = [signal.copy()]
    x = signal.copy()
    for t in range(n_steps):
        noise = rng.randn(*x.shape)
        x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * noise
        trajectory.append(x.copy())
    return np.array(trajectory), betas


# === 输出展示 ===

def demo_random_walks():
    """演示：一维随机游走的 sqrt(n) 缩放规律。"""
    print("=" * 60)
    print("演示 1：一维随机游走")
    print("=" * 60)

    n_walks = 5
    n_steps = 1000
    print(f"\n{n_walks} 条随机游走，每条 {n_steps} 步：\n")

    final_positions = []
    for i in range(n_walks):
        walk = random_walk_1d(n_steps, seed=i)
        final_positions.append(walk[-1])
        print(f"  游走 {i+1}: 终点位置 = {walk[-1]:+4d}, "
              f"最大值 = {walk.max():+4d}, 最小值 = {walk.min():+4d}")

    print(f"\n理论：E[位置] = 0, std(位置) = sqrt({n_steps}) = {np.sqrt(n_steps):.1f}")

    # 大量游走验证统计规律
    n_many = 10000
    finals = np.array([random_walk_1d(n_steps, seed=i)[-1] for i in range(n_many)])
    print(f"\n{n_many} 条游走的统计：均值 = {finals.mean():.2f}, "
          f"标准差 = {finals.std():.2f}（预期 {np.sqrt(n_steps):.2f}）")


def demo_markov_chain():
    """演示：马尔可夫链模拟与平稳分布计算。"""
    print("\n" + "=" * 60)
    print("演示 2：天气马尔可夫链")
    print("=" * 60)

    # 状态：晴天(0)、雨天(1)、多云(2)
    P = [[0.7, 0.1, 0.2],
         [0.3, 0.4, 0.3],
         [0.4, 0.2, 0.4]]
    names = ["晴天", "雨天", "多云"]
    mc = MarkovChain(P, state_names=names)

    # 解析法计算平稳分布
    pi = mc.stationary_distribution()
    print("\n平稳分布（解析解）：")
    for i, name in enumerate(names):
        print(f"  {name}: {pi[i]:.4f}")

    # 模拟验证
    states = mc.simulate(start_state=0, n_steps=100000, seed=42)
    empirical = mc.empirical_distribution(states)
    print("\n经验分布（100000 步，从晴天开始）：")
    for i, name in enumerate(names):
        print(f"  {name}: {empirical[i]:.4f}")

    # 收敛速度验证
    print("\n收敛检验：")
    for length in [100, 1000, 10000, 100000]:
        states = mc.simulate(start_state=1, n_steps=length, seed=42)
        emp = mc.empirical_distribution(states)
        error = np.abs(emp - pi).max()
        print(f"  {length:>7d} 步: 最大误差 = {error:.4f}")

    # 展示一条样本轨迹
    short = mc.simulate(start_state=0, n_steps=20, seed=42)
    sequence = " -> ".join(names[s] for s in short[:15])
    print(f"\n样本轨迹：{sequence}...")


def demo_langevin():
    """演示：朗之万动力学采样高斯分布。"""
    print("\n" + "=" * 60)
    print("演示 3：朗之万动力学 —— 从高斯分布采样")
    print("=" * 60)

    target_mean = 3.0
    target_var = 2.0

    # 目标分布为 N(3, 2)，对应的能量函数梯度为 (x - 3) / 2
    def grad_U(x):
        return (x - target_mean) / target_var

    trajectory = langevin_dynamics(
        grad_U=grad_U,
        x0=np.array([0.0]),
        dt=0.1,
        temperature=1.0,
        n_steps=50000,
        seed=42
    )

    # 丢弃前 5000 步作为预烧期（burn-in）
    samples = trajectory[5000:, 0]
    print(f"\n目标分布：均值 = {target_mean}, 方差 = {target_var}")
    print(f"采样结果（丢弃前 5000 步，保留 {len(samples)} 个样本）：")
    print(f"  均值:   {samples.mean():.4f}（预期 {target_mean}）")
    print(f"  方差:   {samples.var():.4f}（预期 {target_var}）")
    print(f"  标准差: {samples.std():.4f}（预期 {np.sqrt(target_var):.4f}）")


def demo_metropolis_hastings():
    """演示：Metropolis-Hastings 从双峰分布采样。"""
    print("\n" + "=" * 60)
    print("演示 4：Metropolis-Hastings —— 双峰分布采样")
    print("=" * 60)

    # 双峰目标分布：N(-3,1) 和 N(+3,1) 的等比例混合
    def bimodal_log_prob(x):
        v = np.asarray(x).ravel()[0]
        log_p1 = -0.5 * (v - 3) ** 2
        log_p2 = -0.5 * (v + 3) ** 2
        return np.logaddexp(log_p1, log_p2) - np.log(2)

    samples, acc_rate = metropolis_hastings(
        target_log_prob=bimodal_log_prob,
        proposal_std=2.0,
        x0=np.array([0.0]),
        n_samples=100000,
        seed=42
    )

    # 丢弃前 10000 步
    samples_flat = samples[10000:, 0]
    print("\n目标：N(-3,1) 和 N(+3,1) 的等比例混合")
    print(f"接受率: {acc_rate:.2%}")
    print(f"样本均值: {samples_flat.mean():.4f}（预期约 0.0）")
    print(f"样本标准差: {samples_flat.std():.4f}")

    left_mode = samples_flat[samples_flat < 0]
    right_mode = samples_flat[samples_flat >= 0]
    print(f"\n左峰 (x < 0):  均值 = {left_mode.mean():.4f}, 样本数 = {len(left_mode)}")
    print(f"右峰 (x >= 0): 均值 = {right_mode.mean():.4f}, 样本数 = {len(right_mode)}")
    print(f"各峰占比: {len(left_mode)/len(samples_flat):.2%} / "
          f"{len(right_mode)/len(samples_flat):.2%}（预期约 50/50）")

    # 不同提议方差的对比
    print("\n不同提议标准差下的接受率：")
    for std in [0.1, 0.5, 2.0, 5.0, 20.0]:
        _, rate = metropolis_hastings(bimodal_log_prob, std, np.array([0.0]), 10000, seed=42)
        print(f"  std = {std:5.1f}: 接受率 = {rate:.2%}")


def demo_diffusion():
    """演示：前向扩散过程将信号退化为纯噪声。"""
    print("\n" + "=" * 60)
    print("演示 5：前向扩散过程")
    print("=" * 60)

    # 原始信号：两个正弦波的叠加
    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)
    signal = np.sin(t) + 0.5 * np.sin(3 * t)

    trajectory, betas = diffusion_forward(
        signal,
        n_steps=100,
        beta_start=0.001,
        beta_end=0.05,
        seed=42
    )

    print(f"\n原始信号：sin(t) + 0.5*sin(3t)，共 {n_points} 个点")
    print(f"噪声调度：beta 从 {betas[0]:.4f} 到 {betas[-1]:.4f}")

    checkpoints = [0, 10, 25, 50, 75, 100]
    print("\n信号退化过程：")
    print(f"{'步数':>6s} | {'均值':>8s} | {'标准差':>8s} | {'信噪比(dB)':>10s} | {'相关系数':>10s}")
    print("-" * 55)
    for step in checkpoints:
        x = trajectory[step]
        noise_power = np.mean((x - signal) ** 2)
        signal_power = np.mean(signal ** 2)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        corr = np.corrcoef(signal, x)[0, 1]
        print(f"{step:>6d} | {x.mean():>8.4f} | {x.std():>8.4f} | "
              f"{snr:>10.2f} | {corr:>10.4f}")

    print("\n第 0 步：完美信号（相关系数 = 1.0）")
    print("第 100 步：近似纯噪声（相关系数接近 0）")
    print("这就是扩散模型的前向过程。")


if __name__ == "__main__":
    demo_random_walks()
    demo_markov_chain()
    demo_langevin()
    demo_metropolis_hastings()
    demo_diffusion()
