# REINFORCE 从零实现——CartPole 上的策略梯度

import random
import math
import gymnasium as gym


# 第 1 步：Softmax 策略
def softmax_policy(features, theta, n_actions):
    """线性 softmax 策略。theta: (n_actions, n_features)"""
    scores = [sum(theta[a][j] * features[j] for j in range(len(features)))
              for a in range(n_actions)]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1


# 第 2 步：Rollout
def rollout(env, theta, n_actions):
    trajectory = []
    s, _ = env.reset()
    done = False
    while not done:
        s_list = s.tolist() if hasattr(s, 'tolist') else list(s)
        probs = softmax_policy(s_list, theta, n_actions)
        a = sample_action(probs)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        trajectory.append((s_list, a, r, probs))
        s = s2
    return trajectory


# 第 3 步：回报计算
def compute_returns(trajectory, gamma=0.99):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))


# 第 4 步：REINFORCE 更新
def reinforce_update(theta, trajectory, gamma=0.99, lr=0.01, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    n_actions, n_features = len(theta), len(theta[0])

    for (features, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad = [-p for p in probs]
        grad[a] += 1.0
        for i in range(n_actions):
            for j in range(n_features):
                theta[i][j] += lr * advantage * grad[i] * features[j]

    return theta


# 第 5 步：REINFORCE 完整循环
def reinforce(env, n_actions, n_features, gamma=0.99, lr=0.01, num_episodes=2000):
    theta = [[0.0] * n_features for _ in range(n_actions)]
    baseline = 0.0
    all_returns = []

    for ep in range(num_episodes):
        trajectory = rollout(env, theta, n_actions)
        returns = compute_returns(trajectory, gamma)
        ep_return = returns[0] if returns else 0.0
        all_returns.append(ep_return)

        baseline = 0.9 * baseline + 0.1 * ep_return
        theta = reinforce_update(theta, trajectory, gamma, lr, baseline)

        if (ep + 1) % 200 == 0:
            avg = sum(all_returns[-100:]) / min(100, len(all_returns[-100:]))
            print(f"Episode {ep+1}: avg={avg:.1f}, baseline={baseline:.1f}")

    return theta, all_returns


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]

    print(f"状态维度: {n_features}, 动作空间: {n_actions}")
    theta, rewards = reinforce(env, n_actions, n_features, num_episodes=2000)
    avg = sum(rewards[-100:]) / 100
    print(f"\nFinal avg(100): {avg:.1f}")
