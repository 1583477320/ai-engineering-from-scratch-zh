# LLM 评估工具集——困惑度、AlpacaEval、MT-Bench、Chatbot Arena 模拟

import numpy as np
import math


# ============================================================================
# 第 1 步：困惑度计算
# ============================================================================

def perplexity(logits, target_ids):
    """计算困惑度 = exp(交叉熵损失)。"""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = target_ids.reshape(-1)
    max_l = logits_flat.max(axis=-1, keepdims=True)
    log_probs = logits_flat - max_l - np.log(np.exp(logits_flat - max_l).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()
    return np.exp(loss)


# ============================================================================
# 第 2 步：LLM-as-Judge 模拟
# ============================================================================

class LLMJudge:
    """简化版 LLM 评判者——模拟 GPT-4 打分。"""
    def score(self, prompt, response):
        length_score = min(len(response) / 100, 1.0)
        content_score = 0.0
        good_words = ["是", "Paris", "Python", "105", "重力", "算法"]
        for word in good_words:
            if word in response:
                content_score += 0.2
        content_score = min(content_score, 0.9)
        structure_score = 0.1 if "。" in response else 0.0
        final_score = length_score * 0.3 + content_score * 0.5 + structure_score * 0.2
        return min(final_score, 1.0)


# ============================================================================
# 第 3 步：AlpacaEval 模拟
# ============================================================================

def alpaca_eval(model, prompts, reference_answers):
    """简化 AlpacaEval——LLM-as-Judge 比较模型输出 vs 参考。"""
    judge = LLMJudge()
    wins = 0

    for i, prompt in enumerate(prompts):
        model_response = model.generate(prompt)
        model_score = judge.score(prompt, model_response)
        ref_score = judge.score(prompt, reference_answers[i])

        if model_score >= ref_score:
            wins += 1
        print(f"  Prompt {i+1}: 模型={model_score:.2f} vs 参考={ref_score:.2f}")

    win_rate = wins / max(len(prompts), 1)
    return win_rate


# ============================================================================
# 第 4 步：MT-Bench 模拟
# ============================================================================

def mt_bench(model):
    """简化 MT-Bench——多轮对话评分。"""
    judge = LLMJudge()
    categories = ["写作", "推理", "编码", "数学"]

    scores = []
    for category in categories:
        prompt = f"这是一个{category}类问题。请回答。"
        response = model.generate(prompt)
        score = judge.score(prompt, response)
        scores.append(score)
        print(f"  {category}: {score:.2f}")

    avg = sum(scores) / len(scores)
    print(f"  平均: {avg:.2f}")
    return avg


# ============================================================================
# 第 5 步：ELO 排名模拟
# ============================================================================

class ELO:
    """简化 ELO 排名系统。"""
    def __init__(self, k=32):
        self.ratings = {}
        self.k = k

    def add_player(self, name, rating=1000):
        self.ratings[name] = rating

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner, loser):
        e_winner = self.expected_score(self.ratings[winner], self.ratings[loser])
        e_loser = 1.0 - e_winner
        self.ratings[winner] += self.k * (1.0 - e_winner)
        self.ratings[loser] += self.k * (0.0 - e_loser)


# ============================================================================
# 第 6 步：简化模型模拟
# ============================================================================

class DummyModel:
    """模拟模型——生成固定模版回答。"""
    def __init__(self, name, quality=0.5):
        self.name = name
        self.quality = quality

    def generate(self, prompt):
        return f"这是{self.name}对'{prompt[:20]}'的回答。详情请参考相关文档。" * int(1 + self.quality)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 50)
    print("LLM 评估工具")
    print("=" * 50)

    # 1. LLM-as-Judge
    print("\n1. LLM-as-Judge 测试")
    judge = LLMJudge()
    score1 = judge.score("解释重力", "重力是使物体相互吸引的力。")
    score2 = judge.score("解释重力", "重力是一种力。")
    print(f"  好回答得分: {score1:.3f}")
    print(f"  差回答得分: {score2:.3f}")
    print(f"  差异: {score1 - score2:.3f}")

    # 2. AlpacaEval
    print("\n2. AlpacaEval 模拟")
    model = DummyModel("MiniGPT", quality=0.6)
    prompts = ["法国的首都是哪里？", "1+1=？"]
    references = ["巴黎。", "2。"]
    win_rate = alpaca_eval(model, prompts, references)
    print(f"  胜率: {win_rate:.0%}")

    # 3. MT-Bench
    print("\n3. MT-Bench 模拟")
    mt_bench(model)

    # 4. ELO 排名
    print("\n4. Chatbot Arena ELO 排名")
    arena = ELO()
    for name in ["GPT-4", "Claude", "Llama 3", "MiniGPT"]:
        arena.add_player(name)
    # 模拟对战
    arena.update("Claude", "MiniGPT")
    arena.update("GPT-4", "Llama 3")
    arena.update("GPT-4", "Claude")
    arena.update("Llama 3", "MiniGPT")

    for name, rating in sorted(arena.ratings.items(), key=lambda x: -x[1]):
        print(f"  {name}: {rating:.0f}")

    print("\n评估完成。")
