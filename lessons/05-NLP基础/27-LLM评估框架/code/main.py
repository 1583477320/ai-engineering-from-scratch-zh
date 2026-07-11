# 玩具评估指标——BLEU/ROUGE 风格 + 忠实度检查
# 对应课程：阶段 05 · 27

import re
from collections import Counter

STOP = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
        "to", "for", "with", "by", "and", "or", "but", "it", "its", "this", "that"}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(tokens):
    return [t for t in tokens if t not in STOP]


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# === 忠实度 = 答案中的声明有多少比例在上下文中被支撑 ===
def faithfulness(answer, context):
    a_tokens = set(content_tokens(tokenize(answer)))
    c_tokens = set(content_tokens(tokenize(context)))
    if not a_tokens: return 0.0
    return len(a_tokens & c_tokens) / len(a_tokens)


# === ROUGE-1 召回率 ===
def rouge1_recall(hypothesis, reference):
    h_ngrams = ngrams(tokenize(hypothesis), 1)
    r_ngrams = ngrams(tokenize(reference), 1)
    if not r_ngrams: return 0.0
    overlap = sum((h_ngrams & r_ngrams).values())
    return overlap / sum(r_ngrams.values())


# === 完全匹配 ===
def exact_match(pred, gold):
    return 1.0 if tokenize(pred) == tokenize(gold) else 0.0


def main():
    # 模拟 RAG 评估
    context = "Apple Inc released the first iPhone on June 29 2007."
    answers = [
        ("The first iPhone was released on June 29 2007.", "忠实且正确"),
        ("Apple launched a revolutionary product in 2007.", "部分忠实"),
        ("Steve Jobs invented the iPhone in 2008.", "包含错误信息"),
    ]
    reference = "June 29 2007"

    print("=== RAG 评估指标演示 ===")
    print(f"上下文: {context}\n")
    for ans, desc in answers:
        faith = faithfulness(ans, context)
        r1 = rouge1_recall(ans, reference)
        em = exact_match(ans, reference)
        print(f"  [{desc}]")
        print(f"    答案: '{ans}'")
        print(f"    忠实度: {faith:.2f}  ROUGE-1召回: {r1:.2f}  EM: {em:.0f}")
        print()

    print("注意：玩具指标——生产中用 RAGAS (NLI) + DeepEval (G-Eval)。")
    print("EM 过于严格（'2007年6月29日' vs 'June 29, 2007' = 0），忠实度给部分分。")


if __name__ == "__main__":
    main()
