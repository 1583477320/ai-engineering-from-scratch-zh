# GRPO 多臂赌博机——组内相对优势 + KL 惩罚

import random, math


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return [e / sum(exps) for e in exps]


def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1


def grpo_update(theta, prompt_idx, correct, G=8, beta=0.01, lr=0.1):
    """GRPO 更新——组内相对优势。"""
    probs = softmax(theta[prompt_idx])
    n_actions = len(probs)

    samples = [sample_action(probs) for _ in range(G)]
    rewards = [1.0 if s == correct else 0.0 for s in samples]

    mean_r = sum(rewards) / G
    std_r = (sum((r - mean_r)**2 for r in rewards) / G + 1e-8) ** 0.5
    advantages = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advantages):
        grad = [0.0] * n_actions
        grad[a] = 1.0
        for i in range(n_actions):
            theta[prompt_idx][i] += lr * A * (grad[i] - probs[i])

    # KL 惩罚到参考策略
    for i in range(n_actions):
        theta[prompt_idx][i] -= beta * (theta[prompt_idx][i] - reference[prompt_idx][i])

    return theta


if __name__ == "__main__":
    # 2 个问题，4 个可能答案
    correct_answers = [2, 0]
    n_prompts = 2
    n_actions = 4

    theta = [[0.0] * n_actions for _ in range(n_prompts)]
    reference = [list(t) for t in theta]  # 拷贝参考策略

    for step in range(2000):
        for p in range(n_prompts):
            theta = grpo_update(theta, p, correct_answers[p], G=8, beta=0.01, lr=0.05)

        if (step + 1) % 200 == 0:
            probs0 = softmax(theta[0])
            probs1 = softmax(theta[1])
            acc0 = 1.0 if probs0.index(max(probs0)) == correct_answers[0] else 0.0
            acc1 = 1.0 if probs1.index(max(probs1)) == correct_answers[1] else 0.0
            print(f"Step {step+1}: Q0 acc={acc0:.0f}, Q1 acc={acc1:.0f}")
