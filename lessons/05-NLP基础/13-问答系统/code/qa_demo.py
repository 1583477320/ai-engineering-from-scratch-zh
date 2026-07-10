# qa_demo.py — 问答系统：抽取式+检索增强+生成式 + EM/F1评估
# 依赖：无
# 对应课程：阶段 05 · 13（问答系统）

import re
from collections import Counter
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """词元级 F1——EM 太严格时给部分分数。"""
    p_tokens = tokenize(normalize(pred))
    g_tokens = tokenize(normalize(gold))
    if not p_tokens or not g_tokens:
        return 0.0 if (p_tokens or g_tokens) else 1.0
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# 玩具检索——模拟 BM25
CORPUS = [
    "苹果公司于2007年6月29日发布了第一代iPhone。",
    "Macworld 2007大会上，Steve Jobs宣布了iPhone。",
    "Android于2008年作为Google的移动操作系统推出。",
    "第一代iPod于2001年发布。",
]


def toy_retrieve(query: str, top_k: int = 2) -> List[Tuple[float, str]]:
    """简单词汇重叠得分——教学用的 BM25 近似。"""
    q_tokens = set(tokenize(query))
    scored = []
    for doc in CORPUS:
        d_tokens = tokenize(doc)
        d_counts = Counter(d_tokens)
        score = sum(d_counts.get(qt, 0) / (1 + len(d_tokens) / 10)
                    for qt in q_tokens)
        scored.append((score, doc))
    scored.sort(reverse=True)
    return scored[:top_k]


def main():
    # EM vs F1
    print("=" * 50)
    print("QA 评估：EM（完全匹配）vs F1（部分分数）")
    print("=" * 50)
    cases = [
        ("2007年6月29日", "2007年6月29日"),
        ("2007年6月29日", "June 29, 2007"),
        ("2007年", "2007年6月29日"),
        ("2008年", "2007年6月29日"),
    ]
    for pred, gold in cases:
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        print(f"  预测={pred:<16s} 真实={gold:<16s} EM={em:.0f} F1={f1:.2f}")
    print("  EM=1 → 完全相同。F1 → 部分匹配给部分分。两者都不捕获语义。")

    # 检索
    print(f"\n{'='*50}")
    print("检索增强 QA——先搜后读")
    print("=" * 50)
    q = "第一代iPhone什么时候发布的？"
    results = toy_retrieve(q)
    print(f"  问题: {q}")
    print(f"  检索结果:")
    for score, doc in results:
        print(f"    [{score:.3f}] {doc}")

    # 三种 QA 架构对比
    print(f"\n{'='*50}")
    print("三种 QA 架构——2026 的生产选择")
    print("=" * 50)
    print("┌──────────────┬──────────┬──────────┬──────────┐")
    print("│              │ 抽取式   │ RAG      │ 生成式   │")
    print("├──────────────┼──────────┼──────────┼──────────┤")
    print("│ 答案来源     │ 原文span │ 检索+span│ 模型参数 │")
    print("│ 幻觉风险     │ 零       │ 低       │ 中-高    │")
    print("│ 需要检索     │ 是(给定) │ 是       │ 否       │")
    print("│ 覆盖范围     │ 1篇文章  │ 语料库   │ 训练数据 │")
    print("│ 事实准确性   │ 最高     │ 中       │ 最低     │")
    print("└──────────────┴──────────┴──────────┴──────────┘")


if __name__ == "__main__":
    main()
