# Q 学习与 SARSA 从零实现

import random
import math
from collections import defaultdict

GRID = 4
TERMINAL = (3, 3)
ACTIONS = ["up", "down", "left", "right"]


class GridWorld:
    def __init__(self, cliff=False):
        self.grid = GRID
        self.terminal = TERMINAL
        self.cliff = cliff

    def reset(self):
        return (0, 0)

    def step(self, state, action):
        if state == self.terminal:
            return state, 0.0, True
        dr, dc = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}[action]
        r, c = state
        nr = min(self.grid-1, max(0, r+dr))
        nc = min(self.grid-1, max(0, c+dc))
        if self.cliff and (nr==3 and nc==3):
            return self.reset(), -100.0, False
        return (nr, nc), -1.0, (nr,nc)==self.terminal


def choose(Q, s, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(Q[s], key=Q[s].get)


def q_learning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    all_rewards = []
    for _ in range(episodes):
        s, total = env.reset(), 0.0
        while True:
            a = choose(Q, s, epsilon)
            s2, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s2].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            total += r
            s = s2
            if done: break
        all_rewards.append(total)
    return Q, all_rewards


def sarsa(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    all_rewards = []
    for _ in range(episodes):
        s = env.reset()
        a = choose(Q, s, epsilon)
        total = 0.0
        while True:
            s2, r, done = env.step(s, a)
            a2 = choose(Q, s2, epsilon) if not done else None
            target = r + (gamma * Q[s2][a2] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            total += r
            s, a = s2, a2
            if done: break
        all_rewards.append(total)
    return Q, all_rewards


if __name__ == "__main__":
    env = GridWorld()
    print("Q-Learning...")
    Q_q, r_q = q_learning(env)
    print(f"  平均回报(最后100回合): {sum(r_q[-100:])/100:.2f}")
    print("SARSA...")
    Q_s, r_s = sarsa(env)
    print(f"  平均回报(最后100回合): {sum(r_s[-100:])/100:.2f}")
