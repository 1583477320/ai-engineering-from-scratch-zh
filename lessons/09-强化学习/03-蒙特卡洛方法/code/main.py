# 蒙特卡洛方法从零实现
# 演示 first-visit MC 评估、ε-greedy MC 控制、off-policy MC

import random
import math
from collections import defaultdict
matplotlib.use("Agg")


# ============================================================================
# 环境：简化版 4×4 GridWorld
# ============================================================================

GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())


class GridWorldEnv:
    """GridWorld 环境——实现 reset/step API。"""

    def __init__(self, stochastic=False):
        self.grid = GRID
        self.terminal = TERMINAL
        self.stochastic = stochastic

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, state, action):
        if state == self.terminal:
            return state, 0.0, True
        dr, dc = ACTIONS[action]
        if self.stochastic and random.random() < 0.1:
            alt = [a for a in ACTION_LIST if a != action]
            dr, dc = ACTIONS[random.choice(alt)]
        new_r = min(self.grid - 1, max(0, state[0] + dr))
        new_c = min(self.grid - 1, max(0, state[1] + dc))
        new_state = (new_r, new_c)
        return new_state, -1.0, new_state == self.terminal


# ============================================================================
# 第 1 步：Rollout 和回报计算
# ============================================================================

def rollout(env, policy, max_steps=200):
    """运行策略生成一条轨迹。"""
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory


def returns_from(trajectory, gamma=0.99):
    """反向遍历计算每个时间步的折扣回报。"""
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))


# ============================================================================
# 第 2 步：First-visit MC 评估
# ============================================================================

def mc_policy_evaluation(env, policy_fn, num_episodes=5000, gamma=0.99):
    """First-visit 蒙特卡洛策略评估。"""
    V = defaultdict(float)
    counts = defaultdict(int)

    for ep in range(num_episodes):
        trajectory = rollout(env, policy_fn)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, _, _), G in zip(trajectory, returns):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]

    return V


# ============================================================================
# 第 3 步：ε-greedy 策略
# ============================================================================

def make_epsilon_greedy_policy(Q, epsilon=0.1):
    """构造 ε-greedy 策略——关于 Q(s,a) 的软策略。"""
    def policy_fn(s):
        if random.random() < epsilon:
            return random.choice(ACTION_LIST)
        return max(ACTION_LIST, key=lambda a: Q[s][a])
    return policy_fn


# ============================================================================
# 第 4 步：MC 控制
# ============================================================================

def mc_control(env, num_episodes=20000, gamma=0.99, epsilon=0.1):
    """ε-greedy 蒙特卡洛控制。"""
    Q = defaultdict(lambda: {a: 0.0 for a in ACTION_LIST})
    counts = defaultdict(lambda: {a: 0 for a in ACTION_LIST})
    policy_fn = make_epsilon_greedy_policy(Q, epsilon)

    all_returns = []  # 收敛监控
    for ep in range(num_episodes):
        trajectory = rollout(env, policy_fn)
        returns = returns_from(trajectory, gamma)
        all_returns.append(returns[0] if returns else 0.0)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]

        # 每 1000 回合打印进度
        if (ep + 1) % 5000 == 0:
            avg_r = sum(all_returns[-1000:]) / min(1000, len(all_returns[-1000:]))
            print(f"  Episode {ep+1}: 平均回报={avg_r:.2f}")

    return Q, policy_fn


# ============================================================================
# 第 5 步：Off-policy MC 评估（重要度采样）
# ============================================================================

def off_policy_mc_evaluation(env, target_policy, num_episodes=10000,
                              gamma=0.99, behavior_policy=None):
    """
    Off-policy 蒙特卡洛评估——使用重要度采样。
    从 behavior_policy 收集数据，评估 target_policy 的价值。
    """
    if behavior_policy is None:
        # 默认行为策略：均匀随机策略
        behavior_policy = lambda s: random.choice(ACTION_LIST)

    V = defaultdict(float)
    C = defaultdict(float)

    for ep in range(num_episodes):
        trajectory = rollout(env, behavior_policy)
        returns = returns_from(trajectory, gamma)
        G = 0.0
        W = 1.0  # 重要度采样权重

        for (s, a, _), G_t in zip(trajectory, returns):
            G = G_t
            # 更新 V(s): 加权 IS
            C[s] += W
            V[s] += (W / C[s]) * (G - V[s])
            # 如果行为策略选择了目标策略不可能的动作→提前停止
            pi_prob = sum(1.0 for _ in [target_policy(s) if target_policy(s) == a else None])
            if pi_prob == 0:
                break
            mu_prob = 1.0 / len(ACTION_LIST)  # 均匀随机策略
            W *= pi_prob / mu_prob
            if W < 1e-9:
                break

    return V


# ============================================================================
# 可视化工具
# ============================================================================

def print_values(V, title="V^π"):
    print(f"\n{title}:")
    for r in range(GRID):
        row = "  ".join([f"{V.get((r, c), 0.0):6.2f}" for c in range(GRID)])
        print(f"  {row}")


def print_policy_from_Q(Q, title="π*"):
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    print(f"\n{title}:")
    for r in range(GRID):
        row = "  ".join([
            arrow.get(max(ACTION_LIST, key=lambda a: Q[(r,c)][a]), "?")
            if (r,c) in Q else "?"
            for c in range(GRID)
        ])
        print(f"  {row}")


# ============================================================================
# DP 精确解（用于对比）
# ============================================================================

def dp_policy_evaluation(gamma=0.99):
    """DP 精确解（已知模型）。"""
    V = {s: 0.0 for s in [(r,c) for r in range(GRID) for c in range(GRID)]}
    V[TERMINAL] = 0.0

    for _ in range(1000):
        delta = 0.0
        for r in range(GRID):
            for c in range(GRID):
                s = (r, c)
                v = 0.0
                for a in ACTION_LIST:
                    dr, dc = ACTIONS[a]
                    nr = min(GRID-1, max(0, r+dr))
                    nc = min(GRID-1, max(0, c+dc))
                    ns = (nr, nc)
                    v += 0.25 * (-1.0 + gamma * V[ns])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
        if delta < 1e-6:
            break
    return V


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("蒙特卡洛方法演示")
    print("=" * 55)

    env = GridWorldEnv(stochastic=False)

    # 1. DP 精确解（基准）
    print("\n--- 1. DP 精确解（基准）---")
    V_dp = dp_policy_evaluation()
    print_values(V_dp, "V* (DP 精确)")

    # 2. First-visit MC 评估
    print("\n--- 2. First-visit MC 评估（均匀随机策略）---")
    V_mc = mc_policy_evaluation(env, lambda s: random.choice(ACTION_LIST),
                                 num_episodes=5000)
    print_values(V_mc, "V^π (MC, 5000 回合)")

    # 3. 对比 MC vs DP
    print("\n--- 3. MC vs DP 对比 ---")
    states = [(r,c) for r in range(GRID) for c in range(GRID)]
    diff = sum(abs(V_mc.get(s, 0) - V_dp.get(s, 0)) for s in states) / len(states)
    print(f"  平均绝对误差: {diff:.4f}")
    print(f"  V_dp(0,0): {V_dp[(0,0)]:.4f}")
    print(f"  V_mc(0,0): {V_mc.get((0,0), 0):.4f}")

    # 4. MC 控制
    print("\n--- 4. ε-greedy MC 控制 ---")
    Q, policy = mc_control(env, num_episodes=15000, epsilon=0.1)
    print_policy_from_Q(Q, "π* (MC 控制)")

    # 5. 测试学习到的策略
    print("\n--- 5. 策略测试 ---")
    greedy_policy = lambda s: max(Q[s], key=Q[s].get) if s in Q else random.choice(ACTION_LIST)
    total = 0.0
    for _ in range(1000):
        traj = rollout(env, greedy_policy)
        ret = returns_from(traj, 1.0)
        total += ret[0] if ret else 0.0
    avg = total / 1000
    print(f"  1000 回合平均回报: {avg:.2f}（最优=-6.00）")
