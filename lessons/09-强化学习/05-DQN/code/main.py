# DQN 从零实现——CartPole 上的深度 Q 学习

import random
import math
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 第 1 步：Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# 第 2 步：经验回放
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s = torch.FloatTensor([x[0] for x in batch])
        a = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
        r = torch.FloatTensor([x[2] for x in batch])
        s2 = torch.FloatTensor([x[3] for x in batch])
        d = torch.FloatTensor([x[4] for x in batch])
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# 第 3 步：训练
def train_dqn(env, episodes=500, batch_size=64, gamma=0.99, lr=1e-3,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
              target_update=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = epsilon_start
    all_rewards = []

    for ep in range(episodes):
        s, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(s))
                    a = q.argmax().item()

            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_clipped = max(-1.0, min(1.0, r))
            buffer.push(s.tolist(), a, r_clipped, s2.tolist(), done)
            total_reward += r
            s = s2

            if len(buffer) >= batch_size:
                s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                q_vals = q_net(s_b).gather(1, a_b).squeeze()
                with torch.no_grad():
                    q_targets = r_b + gamma * target_net(s2_b).max(1)[0] * (1 - d_b)
                loss = F.mse_loss(q_vals, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = sum(all_rewards[-100:]) / min(100, len(all_rewards[-100:]))
            print(f"Episode {ep+1}: avg={avg:.1f}, ε={epsilon:.3f}")

    return all_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    rewards = train_dqn(env, episodes=500)
    print(f"Final avg(100): {sum(rewards[-100:])/100:.1f}")
