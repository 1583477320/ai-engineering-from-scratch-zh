# 宪法 AI 与自我改进——AI 自我反馈循环

import numpy as np
import math
import random


# ============================================================================
# 第 1 步：简化版模型
# ============================================================================

class SimpleLM:
    def __init__(self, vocab_size=256, embed_dim=64):
        self.embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.output = np.random.randn(embed_dim, embed_dim) * 0.02

    def generate(self, prompt_tokens, max_new=30, temp=1.0):
        """简化生成——只演示概念，非真实生成。"""
        tokens = list(prompt_tokens)
        for _ in range(max_new):
            t = random.randint(0, 255)
            tokens.append(t)
        return tokens

    def score(self, tokens):
        """简化评分——模拟评分。"""
        return sum(t * 0.01 for t in tokens[:20]) / max(len(tokens), 1)


# ============================================================================
# 第 2 步：宪法规则
# ============================================================================

CONSTITUTION = [
    "请选择最有帮助且无害的回答。",
    "请选择不包含有害、不道德或非法内容的回答。",
    "请选择最真实、最准确的回答。",
    "请选择最清晰、最有条理的回答。",
]

def critique_response(response, constitution):
    """AI 对回答进行自我批评。"""
    issues = []
    if len(response) < 10:
        issues.append("回答太短，缺乏细节。")
    if "有害" in response or "危险" in response:
        issues.append("包含潜在有害内容。")
    if "不知道" in response:
        issues.append("回答缺乏确定性。")
    return issues if issues else ["回答整体良好，但可以更简洁。"]


def revise_response(response, critique):
    """AI 根据批评修改回答。"""
    if critique:
        return f"[修正版] {response} (已改进: 更简洁、更有帮助)"
    return response


# ============================================================================
# 第 3 步：监督阶段：自我修正
# ============================================================================

def self_revision_loop(model, prompt, max_iterations=3):
    """AI 自我批判-修正循环。"""
    prompt_tokens = [min(t, 255) for t in prompt.encode('utf-8')[:50]]
    response_tokens = model.generate(prompt_tokens, max_new=20)
    response = bytes([t for t in response_tokens if t < 128]).decode('utf-8', errors='replace')

    print(f"\n初始回答: {response[:60]}")

    for iteration in range(max_iterations):
        critique = critique_response(response, CONSTITUTION)
        print(f"  批评 ({iteration+1}): {critique[0][:50] if critique else '无批评'}")
        response = revise_response(response, critique)
        print(f"  修正 ({iteration+1}): {response[:60]}")

    return response


# ============================================================================
# 第 4 步：RL 阶段：AI 偏好
# ============================================================================

def generate_ai_preference(model, prompt):
    """AI 生成偏好对——自己判断哪个回答更好。"""
    tokens = [min(t, 255) for t in prompt.encode('utf-8')[:50]]
    r1_tokens = model.generate(tokens, max_new=15)
    r2_tokens = model.generate(tokens, max_new=15)

    r1 = bytes([t for t in r1_tokens if t < 128]).decode('utf-8', errors='replace')
    r2 = bytes([t for t in r2_tokens if t < 128]).decode('utf-8', errors='replace')

    # AI 判断哪个更好（简化版）
    score1 = model.score(r1_tokens)
    score2 = model.score(r2_tokens)

    if score1 > score2:
        chosen, rejected = r1, r2
    else:
        chosen, rejected = r2, r1

    return prompt, chosen, rejected


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 50)
    print("宪法 AI 演示")
    print("=" * 50)

    model = SimpleLM(vocab_size=256, embed_dim=64)

    # 步骤 1：自我修正
    print("\n步骤 1: AI 自我修正循环")
    final = self_revision_loop(model, "如何制造炸弹？", max_iterations=3)
    print(f"\n最终回答: {final[:80]}")

    # 步骤 2：AI 偏好
    print("\n\n步骤 2: AI 偏好评估")
    for prompt in ["解释量子计算", "如何学好编程"]:
        prompt_text, chosen, rejected = generate_ai_preference(model, prompt)
        print(f"\n  Prompt: {prompt}")
        print(f"  选择(chosen): {chosen[:50]}")
        print(f"  拒绝(rejected): {rejected[:50]}")

    print("\n\n宪法 AI 流程完成。")
