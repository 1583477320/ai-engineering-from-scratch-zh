# PPO 从零实现——CartPole 上的近端策略优化

import random
import math
import gymnasium as gym


EPS = 0.2


# ============================================================================
# 第 1 步：策略和价值网络
# ============================================================================

def softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def policy(theta, features):
    scores = [sum(theta[a][j] * features[j] for j in range(len(features)))
              for a in range(len(theta))]
    return softmax(scores)


def critic_predict(w, features):
    return sum(w[j] * features[j] for j in range(len(w)))


# ============================================================================
# 第 2 步：轨迹收集
# ============================================================================

def collect_rollout(env, theta, w, n_steps=128):
    buffer = []
    s, _ = env.reset()
    for _ in range(n_steps):
        features = s.tolist() if hasattr(s, 'tolist') else list(s)
        probs = policy(theta, features)
        a = sample_action(probs)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.append({
            "s": features, "a": a, "r": r, "done": done,
            "log_pi_old": math.log(probs[a] + 1e-12),
            "probs": probs,
            "v": critic_predict(w, features),
        })
        if done:
            s, _ = env.reset()
        else:
            s = s2
    return buffer


def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1


# ============================================================================
# 第 3 步：GAE 优势
# ============================================================================

def compute_gae(buffer, gamma=0.99, lam=0.95):
    n = len(buffer)
    rewards = [rec["r"] for rec in buffer]
    values = [rec["v"] for rec in buffer]

    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [a + v for a, v in zip(advantages, values)]
    # 归一化优势
    if len(advantages) > 1:
        m = sum(advantages) / n
        s = math.sqrt(sum((a - m)**2 for a in advantages) / n + 1e-8)
        advantages = [(a - m) / s for a in advantages]
    return advantages, returns


# ============================================================================
# 第 4 步：PPO 更新
# ============================================================================

def ppo_update(theta, w, buffer, gamma=0.99, lam=0.95, lr=0.01,
               k_epochs=4, clip_eps=0.2):
    n_actions = len(theta)
    advantages, returns = compute_gae(buffer, gamma, lam)
    clip_fraction = 0.0
    total_samples = 0

    for _ in range(k_epochs):
        for i, rec in enumerate(buffer):
            features, a = rec["s"], rec["a"]
            adv, target_v = advantages[i], returns[i]

            # 当前策略
            probs = policy(theta, features)
            logp = math.log(probs[a] + 1e-12)

            # 重要度比率
            ratio = math.exp(logp - rec["log_pi_old"])

            # 裁剪代理损失
            surr1 = ratio * adv
            surr2 = max(min(ratio, 1 + clip_eps), 1 - clip_eps) * adv
            pg_loss = -min(surr1, surr2)

            # 更新策略
            if (adv > 0 and ratio >= 1 + clip_eps) or (adv < 0 and ratio <= 1 - clip_eps):
                pg_grad = 0.0
                clip_fraction += 1.0
            else:
                pg_grad = ratio * adv
            total_samples += 1

            grad = [-p for p in probs]
            grad[a] += 1.0
            n_features = len(features)
            for action_i in range(n_actions):
                for j in range(n_features):
                    theta[action_i][j] += lr * pg_grad * grad[action_i] * features[j]

            # 价值更新
            v_hat = critic_predict(w, features)
            v_err = target_v - v_hat
            for j in range(len(w)):
                w[j] += lr * 0.5 * v_err * features[j]

    return theta, w, clip_fraction / max(total_samples, 1)


# ============================================================================
# 第 5 步：PPO 训练
# ============================================================================

def train_ppo(env, n_actions, n_features, n_steps=256, gamma=0.99, lam=0.95,
              lr=0.01, k_epochs=4, clip_eps=0.2, iterations=100):
    theta = [[0.0] * n_features for _ in range(n_actions)]
    w = [0.0] * n_features
    all_rewards = []

    for it in range(iterations):
        buffer = collect_rollout(env, theta, w, n_steps)
        theta, w, clip_frac = ppo_update(theta, w, buffer, gamma, lam,
                                          lr, k_epochs, clip_eps)

        # 评估
        eval_reward = 0.0
        s, _ = env.reset()
        for _ in range(200):
            features = s.tolist() if hasattr(s, 'tolist') else list(s)
            probs = policy(theta, features)
            a = max(range(len(probs)), key=lambda i: probs[i])
            s, r, term, trunc, _ = env.step(a)
            eval_reward += r
            if term or trunc:
                break
        all_rewards.append(eval_reward)

        if (it + 1) % 20 == 0:
            print(f"Iteration {it+1}: eval={eval_reward:.1f}, clip={clip_frac:.2f}")

    return theta, w, all_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    print(f"状态维度: {n_features}, 动作维度: {n_actions}")
    theta, w, rewards = train_ppo(env, n_actions, n_features, iterations=100)
    print(f"最终 eval_reward: {sum(rewards[-5:])/5:.1f}")
