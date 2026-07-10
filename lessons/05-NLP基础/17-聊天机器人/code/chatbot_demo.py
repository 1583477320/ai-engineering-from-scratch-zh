# chatbot_demo.py — 聊天机器人四代架构演示
# 依赖：无（检索式用了Jaccard替代嵌入，无需sentence-transformers）
# 对应课程：阶段 05 · 17（聊天机器人）

import re
from collections import Counter
from typing import List, Tuple, Optional


# ============================================================
# 1. 规则式——ELIZA 风格（Weizenbaum, 1966）
# ============================================================

class RulePattern:
    def __init__(self, pattern: str, template: str):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = template


ELIZA_PATTERNS = [
    RulePattern(r"我叫(\w+)", "你好{0}，很高兴认识你。"),
    RulePattern(r"我(想要|想|需要)(.+)", "为什么你{0}{1}？"),
    RulePattern(r"我(觉得|感觉)(.+)", "为什么你感觉{1}？"),
    RulePattern(r"(你好|嗨|嘿|hi|hello)\b.*", "你好！有什么可以帮你的？"),
    RulePattern(r"(.+)", "能再多说一些吗？"),
]


def rule_based_respond(user_input: str) -> str:
    """规则式：模式匹配 → 模板填充。精确但不开放。"""
    for p in ELIZA_PATTERNS:
        m = p.regex.match(user_input.strip())
        if m:
            return p.template.format(*m.groups())
    return "我不太明白。"


# ============================================================
# 2. 检索式——FAQ Jaccard 匹配
# ============================================================

FAQ: List[Tuple[str, str]] = [
    ("怎么重置密码", "前往 设置 > 安全 > 重置密码。"),
    ("怎么取消订单", "前往 我的订单，找到订单，点击 取消订单。"),
    ("退货政策是什么", "未使用商品 30 天内可退，需原包装。"),
    ("什么时候发货", "通常在付款后 1-3 个工作日内发货。"),
    ("如何联系客服", "在线客服时间为 9:00-18:00，或拨打 400-xxx-xxxx。"),
]


def token_set(text: str) -> set:
    """中文：逐字；英文：正则。Jaccard 只需要集合操作。"""
    chars = set(re.findall(r"[一-鿿]|[a-z]+", text.lower()))
    return chars


def faq_respond(user_input: str, threshold: float = 0.15) -> Tuple[Optional[str], float]:
    """检索式：Jaccard 相似度匹配最佳 FAQ → 阈值拒绝 → 无匹配则升级。

    Jaccard = |A ∩ B| / |A ∪ B|。简单但解释了检索式聊天的核心原理。
    """
    user_tokens = token_set(user_input)
    best_score = 0.0
    best_answer = None
    for question, answer in FAQ:
        q_tokens = token_set(question)
        if not q_tokens or not user_tokens:
            continue
        jaccard = len(user_tokens & q_tokens) / len(user_tokens | q_tokens)
        if jaccard > best_score:
            best_score = jaccard
            best_answer = answer
    if best_score < threshold:
        return None, best_score
    return best_answer, best_score


# ============================================================
# 3. 混合路由——2026 生产默认
# ============================================================

def is_destructive(text: str) -> bool:
    """危险操作检测——路由到结构化确认流程。"""
    danger = ["删除", "取消", "扣款", "退款", "转账", "注销"]
    return any(w in text for w in danger)


def hybrid_respond(user_input: str) -> Tuple[str, str]:
    """混合路由：危险操作→规则 | FAQ→检索 | 其他→LLM智能体。"""
    if is_destructive(user_input):
        return "⚠️ 检测到重要操作，已转至人工确认流程。请再次确认您要执行的操作。", "规则"

    answer, score = faq_respond(user_input)
    if answer:
        return f"{answer}  (FAQ匹配度={score:.2f})", "检索"

    # LLM 智能体回退（教学演示——真实系统这里接 LLM + 工具调用）
    return f"(LLM 智能体处理: '{user_input}')", "智能体"


# ============================================================
# 演示主程序
# ============================================================

def main():
    print("=" * 55)
    print("第一代：规则式（ELIZA 风格）")
    print("=" * 55)
    for msg in ["你好", "我叫张三", "我想要一杯咖啡", "我觉得有点累", "今天天气不错"]:
        print(f"  用户: {msg}")
        print(f"  机器人: {rule_based_respond(msg)}")
        print()

    print("=" * 55)
    print("第二代：检索式（FAQ Jaccard 匹配）")
    print("=" * 55)
    for msg in ["密码忘了怎么办", "如何退款", "怎么重置密码", "今天星期几"]:
        ans, score = faq_respond(msg)
        if ans:
            print(f"  [匹配度={score:.2f}] {msg} → {ans}")
        else:
            print(f"  [匹配度={score:.2f}] {msg} → (无匹配，升级处理)")
    print()

    print("=" * 55)
    print("第四代：混合路由（2026 生产默认）")
    print("=" * 55)
    for msg in ["怎么重置密码", "我要取消订单", "今天天气怎么样", "我要退款"]:
        response, route = hybrid_respond(msg)
        print(f"  [{route}] {msg}")
        print(f"         → {response}")
        print()

    print("=" * 55)
    print("四代演化总结")
    print("=" * 55)
    print("┌──────────┬──────────────┬──────────────┬──────────────┐")
    print("│          │ 规则式       │ 检索式        │ LLM 智能体    │")
    print("├──────────┼──────────────┼──────────────┼──────────────┤")
    print("│ 原理     │ 模式→模板    │ 嵌入→最近邻   │ 推理→工具→验证│")
    print("│ 幻觉风险 │ 零           │ 零(不给生成)  │ 中           │")
    print("│ 覆盖范围 │ 设计好的范围 │ FAQ 库        │ 开放域       │")
    print("│ 2026场景 │ 支付/认证    │ 客服 FAQ      │ 开放式帮助   │")
    print("└──────────┴──────────────┴──────────────┴──────────────┘")
    print("2026 生产 = 三者混合路由——没有单一架构能处理所有请求。")


if __name__ == "__main__":
    main()
