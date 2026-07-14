# 动态规划求解器：策略迭代 + 价值迭代
# 对比两种方法的收敛行为

import random
from collections import defaultdict

# ============================================================================
# 环境定义
# ============================================================================

GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())
SLIP_PROB = 0.1  # 随机版 GridWorld 的滑动概率


def all_states():
    return [(r, c) for r in range(GRID) for c in range(GRID) if (r, c) != TERMINAL]


def step_deterministic(state, action):
    """确定性转移。"""
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    new_r = min(GRID - 1, max(0, state[0] + dr))
    new_c = min(GRID - 1, max(0, state[1] + dc))
    return (new_r, new_c), -1.0, (new_r, new_c) == TERMINAL


def step_stochastic(state, action, slip=SLIP_PROB):
    """随机转移：slip 概率滑到垂直方向。"""
    if state == TERMINAL:
        return state, 0.0, True

    # 确定性部分
    dr, dc = ACTIONS[action]
    # 滑动部分：如果滑动，垂直方向随机选择
    if random.random() < slip:
        perpendicular = [a for a in ACTION_LIST if a != action and
                         ACTIONS[a][0] == -ACTIONS[action][0]]
        if not perpendicular:
            perpendicular = [a for a in ACTION_LIST if a != action]
        chosen = random.choice(perpendicular)
        dr, dc = ACTIONS[chosen]

    new_r = min(GRID - 1, max(0, state[0] + dr))
    new_c = min(GRID - 1, max(0, state[1] + dc))
    return (new_r, new_c), -1.0, (new_r, new_c) == TERMINAL


# ============================================================================
# DP 算法
# ============================================================================

def policy_evaluation(policy_fn, gamma=0.99, tol=1e-6, max_iter=1000, in_place=True):
    """迭代贝尔曼方程计算 V^π(s)。支持就地更新和同步更新。"""
    V = {s: 0.0 for s in all_states()}
    V[TERMINAL] = 0.0

    for iteration in range(max_iter):
        delta = 0.0
        for s in all_states():
            v = sum(
                pi_a * (r + gamma * V[s_next])
                for a, pi_a in policy_fn(s).items()
                for s_next, r in [step_deterministic(s, a)]
            )
            delta = max(delta, abs(v - V[s]))
            if in_place:
                V[s] = v  # 就地更新
            else:
                pass  # 同步更新需在循环结束后统一替换
        if not in_place:
            # 同步更新：一次性替换
            V_new = {}
            for s in all_states():
                V_new[s] = sum(
                    pi_a * (r + gamma * V[s_next])
                    for a, pi_a in policy_fn(s).items()
                    for s_next, r in [step_deterministic(s, a)]
                )
            V = V_new
        if delta < tol:
            return V, iteration + 1
    return V, max_iter


def policy_improvement(V, gamma=0.99):
    """根据 V^π 改进策略。"""
    new_policy = {}
    for s in all_states():
        best_a = max(
            ACTION_LIST,
            key=lambda a: step_deterministic(s, a)[1] + gamma * V[step_deterministic(s, a)[0]]
        )
        new_policy[s] = best_a
    return new_policy


def policy_iteration(gamma=0.99):
    """策略迭代：评估→改进→重复。"""
    policy = {s: "up" for s in all_states()}
    outer_iters = 0

    for outer in range(100):
        V, eval_iters = policy_evaluation(
            lambda s: {policy[s]: 1.0}, gamma
        )
        new_policy = policy_improvement(V, gamma)
        outer_iters += 1
        if new_policy == policy:
            return V, policy, outer_iters, eval_iters
        policy = new_policy

    return V, policy, outer_iters, 0


def value_iteration(gamma=0.99, tol=1e-6, max_iter=1000):
    """价值迭代：直接迭代贝尔曼最优方程。"""
    V = {s: 0.0 for s in all_states()}
    V[TERMINAL] = 0.0

    for iteration in range(max_iter):
        delta = 0.0
        for s in all_states():
            best_v = float("-inf")
            for a in ACTION_LIST:
                s_next, r = step_deterministic(s, a)
                va = r + gamma * V[s_next]
                if va > best_v:
                    best_v = va
            delta = max(delta, abs(best_v - V[s]))
            V[s] = best_v
        if delta < tol:
            break

    policy = policy_improvement(V, gamma)
    return V, policy, iteration + 1


# ============================================================================
# 可视化
# ============================================================================

def print_values(V, title="V"):
    print(f"\n{title}:")
    for r in range(GRID):
        row = "  ".join([f"{V.get((r, c), 0.0):6.2f}" for c in range(GRID)])
        print(f"  {row}")


def print_policy(policy, title="π"):
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    print(f"\n{title}:")
    for r in range(GRID):
        row = "  ".join([arrow.get(policy.get((r, c), ""), "?") for c in range(GRID)])
        print(f"  {row}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("动态规划对比：策略迭代 vs 价值迭代")
    print("=" * 50)

    gamma = 0.99

    # 1. 策略迭代
    print("\n--- 策略迭代 ---")
    V_pi, policy_pi, outer, inner = policy_iteration(gamma)
    print(f"  外层迭代: {outer} 轮")
    print(f"  内层评估: {inner} 步")
    print(f"  V*(0,0): {V_pi[(0,0)]:.4f}")
    print_values(V_pi, "V* (策略迭代)")
    print_policy(policy_pi, "π* (策略迭代)")

    # 2. 价值迭代
    print("\n--- 价值迭代 ---")
    V_vi, policy_vi, iters = value_iteration(gamma)
    print(f"  迭代次数: {iters} 轮")
    print(f"  V*(0,0): {V_vi[(0,0)]:.4f}")
    print_values(V_vi, "V* (价值迭代)")
    print_policy(policy_vi, "π* (价值迭代)")

    # 3. 对比
    print("\n--- 对比 ---")
    diff = abs(V_pi[(0, 0)] - V_vi[(0, 0)])
    print(f"  策略迭代 V*(0,0): {V_pi[(0, 0)]:.4f}")
    print(f"  价值迭代 V*(0,0): {V_vi[(0, 0)]:.4f}")
    print(f"  差异: {diff:.6f}")

    # 4. 不同 γ 下的行为
    print("\n--- 折扣因子影响 ---")
    for gamma in [0.5, 0.9, 0.95, 0.99]:
        V, _, iters = value_iteration(gamma)
        print(f"  γ={gamma:.2f}: V*(0,0)={V[(0,0)]:.4f}, 收敛步数={iters}")
