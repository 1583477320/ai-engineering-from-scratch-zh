# 少样本、思维链与自洽性

import random
from collections import Counter


# ============================================================================
# 第 1 步：少样本提示
# ============================================================================

def few_shot_prompt(query, examples):
    """构建少样本提示。"""
    prompt = ""
    for i, (q, a) in enumerate(examples, 1):
        prompt += f"示例 {i}:\n问题: {q}\n答案: {a}\n\n"
    prompt += f"现在回答:\n问题: {query}\n答案:"
    return prompt


def zero_shot_cot_prompt(query):
    """零样本思维链。"""
    return f"{query}\n\n让我们一步步思考。"


# ============================================================================
# 第 2 步：自洽性
# ============================================================================

def extract_answer(text):
    """从回复中提取最终答案。"""
    import re
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None


def self_consistency(model_fn, prompt, n_paths=5, temperature=0.7):
    """采样多个路径，取多数答案。"""
    responses = [model_fn(prompt, temperature=temperature) for _ in range(n_paths)]
    answers = [extract_answer(r) for r in responses if extract_answer(r) is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


# ============================================================================
# 第 3 步：模拟生成器
# ============================================================================

def mock_model(prompt, temperature=0.0):
    """模拟模型——简化版演示。"""
    if "一步步" in prompt:
        # 模拟 CoT 推理
        return "我们先列出数量。\n1. 开始有 12 个\n2. 给了 3 个: 12-3=9\n3. 买了 5 个: 9+5=14\n答案是 14"
    return "答案是 14"


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("少样本与思维链演示\n")

    # 1. 少样本
    examples = [
        ("2+2=?", "4"),
        ("3×3=?", "9"),
        ("10-4=?", "6"),
    ]
    prompt = few_shot_prompt("5+5=?", examples)
    print(f"少样本提示:\n{prompt}\n")

    # 2. CoT
    cot_prompt = zero_shot_cot_prompt("小明有 12 个苹果，给了小红 3 个，买了 5 个。现在有几个？")
    response = mock_model(cot_prompt)
    print(f"CoT 回复:\n{response}\n")
    print(f"提取答案: {extract_answer(response)}")

    # 3. 自洽性
    ans = self_consistency(mock_model, cot_prompt, n_paths=5, temperature=0.5)
    print(f"自洽性多数决策: {ans}")
