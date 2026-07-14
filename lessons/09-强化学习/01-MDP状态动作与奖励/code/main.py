# GridWorld MDP：从零实现马尔可夫决策过程

import random
from collections import defaultdict

# ============================================================================
# 第 1 步：定义 4×4 GridWorld 环境
# ============================================================================

GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())


def step(state, action):
    """
    执行动作，返回 (下一个状态, 奖励, 是否结束)。
    确定性转移：每步奖励 -1，终点奖励 0。
    """
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    new_r = min(GRID - 1, max(0, state[0] + dr))
    new_c = min(GRID - 1, max(0, state[1] + dc))
    new_state = (new_r, new_c)
    return new_state, -1.0, new_state == TERMINAL


def all_states():
    """返回所有非终止状态。"""
    return [(r, c) for r in range(GRID) for c in range(GRID) if (r, c) != TERMINAL]


# ============================================================================
# 第 2 步：策略定义
# ============================================================================

def uniform_policy(state):
    """均匀随机策略——每个动作概率 0.25。"""
    return {a: 0.25 for a in ACTION_LIST}


def deterministic_policy(action_name):
    """确定性策略——所有状态采取同一个动作。"""
    def policy(state):
        return {a: (1.0 if a == action_name else 0.0) for a in ACTION_LIST}
    return policy


# ============================================================================
# 第 3 步：策略评估（迭代贝尔曼方程）
# ============================================================================

def policy_evaluation(policy_fn, gamma=0.99, tol=1e-6, max_iter=1000):
    """
    迭代贝尔曼方程计算 V^π(s)。
    Args:
        policy_fn: 策略函数，输入状态，返回 {动作: 概率}
        gamma: 折扣因子
        tol: 收敛阈值
        max_iter: 最大迭代次数
    Returns:
        V: 价值函数字典 {(r, c): 价值}
    """
    V = {s: 0.0 for s in all_states()}
    V[TERMINAL] = 0.0

    for iteration in range(max_iter):
        delta = 0.0
        for s in all_states():
            v = 0.0
            for a, pi_a in policy_fn(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            print(f"  策略评估收敛: {iteration+1} 次迭代, Δ={delta:.2e}")
            return V

    print(f"  警告: 达到最大迭代次数 {max_iter}, Δ={delta:.2e}")
    return V


# ============================================================================
# 第 4 步：策略改进
# ============================================================================

def policy_improvement(V, gamma=0.99):
    """根据 V^π 改进策略——对每个状态选使贝尔曼方程最大化的动作。"""
    new_policy = {}
    for s in all_states():
        best_a = max(
            ACTION_LIST,
            key=lambda a: sum(
                p * (r + gamma * V[s_next])
                for s_next, r, p in [step(s, a)]  # 确定性环境：只有一个结果
            ),
        )
        new_policy[s] = best_a
    return new_policy


# ============================================================================
# 第 5 步：策略迭代
# ============================================================================

def policy_iteration(gamma=0.99):
    """策略迭代：评估→改进→重复直到策略不变。"""
    # 从均匀随机策略开始
    policy = {s: "up" for s in all_states()}

    for outer in range(100):
        # 评估
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        # 改进
        new_policy = policy_improvement(V, gamma)
        # 检查收敛
        if new_policy == policy:
            print(f"  策略迭代收敛: {outer+1} 轮外层迭代")
            return V, policy
        policy = new_policy

    print("  警告: 策略迭代未收敛")
    return V, policy


# ============================================================================
# 第 6 步：价值迭代
# ============================================================================

def value_iteration(gamma=0.99, tol=1e-6, max_iter=1000):
    """价值迭代：直接迭代贝尔曼最优方程。"""
    V = {s: 0.0 for s in all_states()}
    V[TERMINAL] = 0.0

    for iteration in range(max_iter):
        delta = 0.0
        for s in all_states():
            v = max(
                sum(p * (r + gamma * V[s_next])
                    for s_next, r, p in [step(s, a)])
                for a in ACTION_LIST
            )
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            print(f"  价值迭代收敛: {iteration+1} 次迭代, Δ={delta:.2e}")
            break

    # 提取最优策略
    policy = policy_improvement(V, gamma)
    return V, policy


# ============================================================================
# 第 7 步：随机策略的 Rollout
# ============================================================================

def rollout(policy_fn, max_steps=200):
    """用给定策略执行一个回合。"""
    s = (0, 0)
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        dist = policy_fn(s)
        # 按概率采样动作
        rand_val = random.random()
        cumul = 0.0
        chosen_a = ACTION_LIST[0]
        for a, p in dist.items():
            cumul += p
            if rand_val < cumul:
                chosen_a = a
                break

        s_next, r, done = step(s, chosen_a)
        total_reward += r
        steps += 1
        s = s_next
        if done:
            break

    return total_reward, steps


# ============================================================================
# 可视化工具
# ============================================================================

def print_values(V, title="价值函数"):
    """将价值函数打印为 4×4 网格。"""
    print(f"\n{title}:")
    print("  " + "    ".join([f"  c{c} " for c in range(GRID)]))
    for r in range(GRID):
        row = "  ".join([f"{V.get((r, c), 0.0):6.2f}" for c in range(GRID)])
        print(f"r{r} {row}")


def print_policy(policy, title="策略"):
    """将策略打印为 4×4 网格（箭头）。"""
    arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    print(f"\n{title}:")
    for r in range(GRID):
        row = "  ".join([
            arrow_map.get(policy.get((r, c), ""), "?")
            for c in range(GRID)
        ])
        print(f"  {row}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GridWorld MDP 演示")
    print("=" * 60)

    # 1. 随机策略的 Rollout
    print("\n--- 1. 随机策略 Rollout ---")
    rewards = []
    for _ in range(1000):
        total_r, steps = rollout(uniform_policy)
        rewards.append(total_r)
    avg_reward = sum(rewards) / len(rewards)
    print(f"  1000 次 Rollout 平均回报: {avg_reward:.2f}")
    print(f"  最优回报（最短路径）: -6.00")

    # 2. 策略评估
    print("\n--- 2. 策略评估（均匀随机策略，γ=0.99）---")
    V_uniform = policy_evaluation(uniform_policy, gamma=0.99)
    print_values(V_uniform, "V^π (均匀随机)")
    print(f"  V(start) = {V_uniform[(0, 0)]:.2f}")

    # 3. 不同 γ 下的 V(start)
    print("\n--- 3. 折扣因子 γ 对 V(start) 的影响 ---")
    for gamma in [0.5, 0.9, 0.95, 0.99]:
        V = policy_evaluation(uniform_policy, gamma=gamma)
        print(f"  γ={gamma:.2f}: V(start)={V[(0, 0)]:.2f}")

    # 4. 策略迭代
    print("\n--- 4. 策略迭代 ---")
    V_pi, policy_pi = policy_iteration(gamma=0.99)
    print_values(V_pi, "V* (策略迭代)")
    print_policy(policy_pi, "最优策略 π*")
    print(f"  V*(start) = {V_pi[(0, 0)]:.2f} (应≈-6)")

    # 5. 价值迭代
    print("\n--- 5. 价值迭代 ---")
    V_vi, policy_vi = value_iteration(gamma=0.99)
    print_values(V_vi, "V* (价值迭代)")
    print_policy(policy_vi, "最优策略 π*")

    # 6. 对比
    print("\n--- 6. 对比 ---")
    print(f"  策略迭代 V*(0,0): {V_pi[(0, 0)]:.4f}")
    print(f"  价值迭代 V*(0,0): {V_vi[(0, 0)]:.4f}")
    print(f"  差异: {abs(V_pi[(0, 0)] - V_vi[(0, 0)]):.6f}")
