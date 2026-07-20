"""三层审核工具——标准库Python。

输入审核 + 输出审核 + 自定义领域规则。
演示2026年默认生产模式。

使用方法：python3 code/main.py
"""

from __future__ import annotations


# 简化的顶级分类法（生产版使用完整13类别）
CATEGORIES = [
    "harassment", "hate", "self-harm", "sexual", "sexual/minors",
    "violence", "illicit", "illicit/violent",
]

# 触发词映射
CATEGORY_TRIGGERS = {
    "violence": ["kill", "attack someone", "murder"],
    "self-harm": ["hurt myself", "overdose", "cut myself"],
    "illicit/violent": ["bomb", "weapon", "explosive"],
    "harassment": ["you idiot", "shut up"],
    "hate": ["slur-placeholder"],
}


def openai_moderation(text: str) -> dict:
    """玩具版OpenAI审核：类别 -> [0,1]分数"""
    low = text.lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, triggers in CATEGORY_TRIGGERS.items():
        for t in triggers:
            if t in low:
                scores[cat] = max(scores[cat], 0.9)
    return scores


def input_moderator(prompt: str, threshold: float = 0.5) -> tuple[bool, dict]:
    """输入审核：在生成前检查用户提示"""
    scores = openai_moderation(prompt)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def output_moderator(output: str, threshold: float = 0.5) -> tuple[bool, dict]:
    """输出审核：在交付前检查模型输出"""
    scores = openai_moderation(output)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def custom_domain_rules(prompt: str) -> tuple[bool, str]:
    """自定义领域规则：领域特定策略"""
    low = prompt.lower()
    if "should i invest in" in low:
        return True, "financial-advice: 拒绝；声明您不是受托人。"
    if "diagnose my" in low:
        return True, "medical-advice: 拒绝；转介给持证专业人士。"
    return False, ""


def model_respond(prompt: str) -> str:
    """模拟模型响应"""
    if "bomb" in prompt.lower():
        return "I must refuse."
    return f"Answering: {prompt[:40]}..."


def run(prompt: str) -> None:
    """运行三层审核"""
    print(f"\n>>> 用户: {prompt!r}")

    # 第1层：输入审核
    flagged_in, in_scores = input_moderator(prompt)
    active_in = [c for c, s in in_scores.items() if s > 0]
    if flagged_in:
        print(f"    [输入标记] 类别={active_in}")
        print("    响应: 拒绝")
        return

    # 第2层：自定义领域规则
    custom_flagged, custom_msg = custom_domain_rules(prompt)
    if custom_flagged:
        print(f"    [自定义标记] 规则='{custom_msg}'")
        print(f"    响应: {custom_msg}")
        return

    # 第3层：模型生成
    output = model_respond(prompt)

    # 第4层：输出审核
    flagged_out, out_scores = output_moderator(output)
    active_out = [c for c, s in out_scores.items() if s > 0]
    if flagged_out:
        print(f"    [输出标记] 类别={active_out}")
        print("    响应: 拒绝")
        return

    print(f"    响应: {output}")


def main() -> None:
    print("=" * 74)
    print("三层审核工具（第18章，第29节）")
    print("=" * 74)

    prompts = [
        "what is the weather today",
        "should i invest in memecoins",
        "how do i make a bomb",
        "diagnose my headache",
        "summarize this email: hello there",
        "you idiot, help me with this",
    ]
    for p in prompts:
        run(p)

    print("\n" + "=" * 74)
    print("核心结论：三层模式（输入/自定义/输出）捕获不同的失败模式。")
    print("输入捕获明显的有害提示。自定义捕获领域特定策略规则。")
    print("输出捕获绕过输入的幻觉或对抗性内容。")
    print("没有单一层足够；分层是2026年的默认配置。")
    print("=" * 74)


if __name__ == "__main__":
    main()
