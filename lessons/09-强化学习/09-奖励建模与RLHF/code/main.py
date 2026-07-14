# RLHF 从零实现——Bradley-Terry RM + PPO with KL

import math, random
from collections import defaultdict


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))


def dot(a, b):
    return sum(a.get(k, 0) * v for k, v in b.items())


def bag(texts):
    words = defaultdict(int)
    for t in texts:
        for w in t.split():
            words[w] += 1
    return dict(words)


PROMPTS = ["help me", "explain this", "solve this"]
GOOD = {"clear", "specific", "kind", "thorough", "helpful"}
BAD = {"vague", "rude", "wrong", "short", "lazy"}


def make_pref(rng):
    x = rng.choice(PROMPTS)
    y_pos = rng.choice(list(GOOD)) + " " + rng.choice(list(GOOD))
    y_neg = rng.choice(list(BAD)) + " " + rng.choice(list(BAD))
    return x, y_pos, y_neg


def train_rm(w, steps=500, lr=0.05, rng=None):
    correct = 0
    total = 0
    for _ in range(steps):
        x, y_pos, y_neg = make_pref(rng)
        r_pos = dot(w, bag([y_pos]))
        r_neg = dot(w, bag([y_neg]))
        p = sigmoid(r_pos - r_neg)
        for tok, cnt in bag([y_pos]).items():
            w[tok] = w.get(tok, 0) + lr * (1 - p) * cnt
        for tok, cnt in bag([y_neg]).items():
            w[tok] = w.get(tok, 0) - lr * (1 - p) * cnt
        total += 1
        if r_pos > r_neg:
            correct += 1
    return w, correct / total


if __name__ == "__main__":
    rng = random.Random(42)
    w = {}
    w, acc = train_rm(w, steps=1000, rng=rng)
    print(f"RM 训练完成，准确率: {acc:.2%}")
    # 测试
    r_good = dot(w, bag(["clear", "specific"]))
    r_bad = dot(w, bag(["vague", "rude"]))
    print(f"好回复得分: {r_good:.3f}, 坏回复得分: {r_bad:.3f}")
