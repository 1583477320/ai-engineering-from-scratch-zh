# A2C 从零实现——CartPole 上的同步演员-评论家

import random
import math
import gymnasium as gym


# ============================================================================
# 第 1 步：Softmax 策略
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


def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1


# ============================================================================
# 第 2 步：评论家（线性）
# ============================================================================

def critic_predict(w, features):
    return sum(w[j] * features[j] for j in range(len(w)))


def critic_update(w, features, target, lr=0.01):
    v_hat = critic_predict(w, features)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * features[j]
    return v_hat


# ============================================================================
# 第 3 步：GAE 优势计算
# ============================================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_val=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_val
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    # 归一化优势
    if len(advantages) > 1:
        m = sum(advantages) / len(advantages)
        s = math.sqrt(sum((a - m)**2 for a in advantages) / len(advantages) + 1e-8)
        advantages = [(a - m) / s for a in advantages]
    return advantages, returns


# ============================================================================
# 第 4 步：Rollout
# ============================================================================

def rollout(env, theta, w, n_steps=128):
    s, _ = env.reset()
    buffer = []
    for _ in range(n_steps):
        features = s.tolist() if hasattr(s, 'tolist') else list(s)
        probs = policy(theta, features)
        a = sample_action(probs)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        v = critic_predict(w, features)
        buffer.append((features, a, r, probs, v))
        if done:
            s, _ = env.reset()
        else:
            s = s2
    return buffer


# ============================================================================
# 第 5 步：A2C 更新
# ============================================================================

def a2c_update(theta, w, buffer, gamma=0.99, lam=0.95, lr_a=0.01, lr_c=0.01):
    s_list = [t[0] for t in buffer]
    a_list = [t[1] for t in buffer]
    r_list = [t[2] for t in buffer]
    p_list = [t[3] for t in buffer]
    v_list = [t[4] for t in buffer]

    advantages, returns = compute_gae(r_list, v_list, gamma, lam, 0.0)

    for t in range(len(buffer)):
        features, a, probs = s_list[t], a_list[t], p_list[t]
        adv, target_v = advantages[t], returns[t]
        n_actions = len(theta)
        n_features = len(features)

        # 评论家
        critic_update(w, features, target_v, lr_c)

        # 演员
        grad = [-p for p in probs]
        grad[a] += 1.0
        for i in range(n_actions):
            for j in range(n_features):
                theta[i][j] += lr_a * adv * grad[i] * features[j]

    return theta, w


# ============================================================================
# 第 6 步：A2C 训练
# ============================================================================

def train_a2c(env, n_actions, n_features, n_steps=128, gamma=0.99, lam=0.95,
              lr_a=0.01, lr_c=0.01, episodes=500):
    theta = [[0.0] * n_features for _ in range(n_actions)]
    w = [0.0] * n_features
    all_rewards = []
    total_steps = 0

    for ep in range(episodes):
        buffer = rollout(env, theta, w, n_steps)
        theta, w = a2c_update(theta, w, buffer, gamma, lam, lr_a, lr_c)
        total_steps += n_steps

        # 评估
        if (ep + 1) % 20 == 0:
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
            print(f"Episode {ep+1}: eval_reward={eval_reward:.1f}")

    return theta, w, all_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    print(f"状态维度: {n_features}, 动作维度: {n_actions}")
    theta, w, rewards = train_a2c(env, n_actions, n_features, episodes=200)
    print(f"最终 eval_reward (平均): {sum(rewards[-5:])/5:.1f}")
