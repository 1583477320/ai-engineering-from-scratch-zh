# main.py — 推测解码教学实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 16（推测解码）
#
# 核心思想：小模型（草稿模型）快速生成 K 个候选词元，
# 大模型（验证模型）一次前向传播并行验证——正确接受，错误重采。
# 输出分布与完全用大模型生成完全等价。

import numpy as np
from typing import List, Tuple

VOCAB_SIZE = 10
TOKEN_LABELS = ["the", "a", "is", "not", "and",
                "in", "to", "of", "it", "you"]


# ============================================================
# 模型模拟（用条件概率表代替真实神经网络）
# ============================================================

class SimpleLM:
    """简化的语言模型：基于转移概率表生成词元。

    Args:
        sharpness: Dirichlet 浓度参数。值越小，分布越尖锐，
                   模拟更大、更确定的模型。
    """

    def __init__(self, sharpness=1.0, seed=42):
        rng = np.random.default_rng(seed)
        alpha = np.ones(VOCAB_SIZE) * sharpness
        self.transition = rng.dirichlet(alpha, size=VOCAB_SIZE)
        self.start_probs = rng.dirichlet(alpha)

    def predict_next(self, context: List[int]) -> np.ndarray:
        """根据上下文预测下一个词元的概率分布。"""
        if not context:
            return self.start_probs
        return self.transition[context[-1]]

    def generate_token(self, context: List[int]) -> int:
        """采样一个词元。"""
        return int(np.random.choice(VOCAB_SIZE, p=self.predict_next(context)))

    def parallel_verify(self, prefix: List[int],
                        draft_tokens: List[int]) -> np.ndarray:
        """一次前向验证所有候选词元（推测解码加速的关键）。"""
        n = len(draft_tokens)
        probs = np.zeros((n, VOCAB_SIZE))
        for i in range(n):
            probs[i] = self.predict_next(prefix + draft_tokens[:i])
        return probs


# ============================================================
# 推测解码核心算法
# ============================================================

def rejection_sample(draft_prob, target_prob, draft_token) -> Tuple[bool, int]:
    """拒绝采样：决定是否接受草稿词元。

    数学保证：最终采样分布与直接用目标模型生成完全一致。
    接受概率 = min(1, target_prob(draft_token) / draft_prob(draft_token))。
    """
    p_draft = draft_prob[draft_token]
    p_target = target_prob[draft_token]
    if np.random.random() < min(1.0, p_target / max(p_draft, 1e-12)):
        return True, draft_token
    # 拒绝后从 (target - draft)_+ 重采样
    residual = np.maximum(target_prob - draft_prob, 0)
    residual /= residual.sum() + 1e-12
    return False, int(np.random.choice(VOCAB_SIZE, p=residual))


def speculative_decode(draft, target, prefix, max_new, K=4):
    """推测解码：草稿模型猜 K 个词元，大模型一次并行验证。

    Returns:
        (生成的词元列表, 大模型前向次数)
    """
    generated = list(prefix)
    target_fwd = 0

    while len(generated) - len(prefix) < max_new:
        # 第 1 步：草稿模型逐词元生成 K 个候选
        draft_tokens = []
        for _ in range(K):
            draft_tokens.append(draft.generate_token(generated + draft_tokens))

        # 第 2 步：目标模型一次前向验证全部
        target_probs = target.parallel_verify(generated, draft_tokens)
        target_fwd += 1

        # 第 3 步：逐位置拒绝采样
        accepted_count = 0
        for i in range(K):
            draft_prob = draft.predict_next(generated + draft_tokens[:i])
            accepted, token = rejection_sample(
                draft_prob, target_probs[i], draft_tokens[i])
            generated.append(token)
            if accepted:
                accepted_count += 1
            else:
                break

        # 全部接受时，额外从目标模型生成一个词元
        if accepted_count == K:
            extra = target.generate_token(generated)
            generated.append(extra)
            target_fwd += 1

    return generated[len(prefix):max_new + len(prefix)], target_fwd


def standard_decode(target, prefix, max_new):
    """标准自回归解码：每个词元都需一次大模型前向（基线）。"""
    generated = list(prefix)
    for _ in range(max_new):
        generated.append(target.generate_token(generated))
    return generated[len(prefix):], max_new


# ============================================================
# 演示
# ============================================================

def token_str(tokens):
    return " ".join(TOKEN_LABELS[t] for t in tokens)


def demo():
    draft = SimpleLM(sharpness=1.0, seed=42)   # 草稿：分布较平
    target = SimpleLM(sharpness=0.3, seed=99)   # 目标：分布更尖锐
    prefix = [0, 1]
    K, max_new = 5, 20

    print("=" * 56)
    print("  推测解码 — 教学演示")
    print("=" * 56)

    # --- 拒绝采样单步演示 ---
    print("\n[1] 拒绝采样演示")
    dp = draft.predict_next(prefix)
    tp = target.predict_next(prefix)
    dt = draft.generate_token(prefix)
    accepted, ft = rejection_sample(dp, tp, dt)
    print(f"  草稿分布: {np.round(dp, 3)}")
    print(f"  目标分布: {np.round(tp, 3)}")
    print(f"  草稿采样: {TOKEN_LABELS[dt]} (P_draft={dp[dt]:.3f}, "
          f"P_target={tp[dt]:.3f})")
    result = "接受" if accepted else "拒绝→重采样"
    print(f"  结果: {result}  最终={TOKEN_LABELS[ft]}")

    # --- 推测解码 vs 标准解码 ---
    print(f"\n[2] 推测解码 vs 标准解码 (生成 {max_new} 词元)")
    std_tokens, std_fwd = standard_decode(target, list(prefix), max_new)
    sp_tokens, sp_fwd = speculative_decode(draft, target, list(prefix),
                                           max_new, K)

    print(f"  标准解码: {std_fwd} 次大模型前向 → {token_str(std_tokens)}")
    print(f"  推测解码: {sp_fwd} 次大模型前向 → {token_str(sp_tokens)}")
    print(f"  加速比: {std_fwd}/{sp_fwd} ≈ {std_fwd/sp_fwd:.1f}x "
          f"(K={K})")

    # --- 蒙特卡洛验证分布等价性 ---
    print(f"\n[3] 分布等价性验证 (蒙特卡洛, 各2000次)")
    N, L = 2000, 3
    std_counts, sp_counts = {}, {}
    for _ in range(N):
        key = tuple(standard_decode(target, [0], L)[0])
        std_counts[key] = std_counts.get(key, 0) + 1
        key = tuple(speculative_decode(draft, target, [0], L, K=4)[0])
        sp_counts[key] = sp_counts.get(key, 0) + 1

    all_keys = sorted(set(std_counts) | set(sp_counts))[:6]
    max_diff = 0
    for key in all_keys:
        sp = std_counts.get(key, 0) / N
        sq = sp_counts.get(key, 0) / N
        d = abs(sp - sq)
        max_diff = max(max_diff, d)
        label = token_str(list(key))
        if std_counts.get(key, 0) > 0 or sp_counts.get(key, 0) > 0:
            print(f"  {label:<16} 标准={sp:.3f}  推测={sq:.3f}  差={d:.3f}")

    eq = "等价 (最大差异 < 0.05)" if max_diff < 0.05 else "存在波动"
    print(f"\n  结论: 最大差异={max_diff:.4f} → {eq}")
    print(f"  理论上差异随试验次数→∞ 趋向 0")


if __name__ == "__main__":
    demo()
