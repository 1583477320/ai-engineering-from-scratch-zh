# 玩具 NLI 分类器——词汇重叠 + 否定检测
# 对应课程：阶段 05 · 21
# 仅用标准库——展示任务形态：(前提, 假设) → {蕴含, 矛盾, 中立}

import re
from collections import Counter

NEGATIONS = {"not", "no", "never", "nobody", "nothing", "neither", "nor", "none", "without"}
STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "as", "and", "or", "but",
        "there", "this", "that", "it", "its", "i", "he", "she", "we", "they",
        "do", "does", "did", "has", "have", "had", "will", "would", "could", "should"}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_words(tokens):
    """去掉停用词和否定词——后者单独处理。"""
    return [t for t in tokens if t not in STOP and t not in NEGATIONS]


def has_negation(tokens):
    return any(t in NEGATIONS for t in tokens)


def lexical_overlap(prem_tokens, hyp_tokens):
    """假设中的内容词有多少比例在前提中出现。"""
    p_content = set(content_words(prem_tokens))
    h_content = content_words(hyp_tokens)
    if not h_content: return 0.0
    return sum(1 for t in h_content if t in p_content) / len(h_content)


def predict_nli(premise, hypothesis):
    """两个浅层特征：(1)词汇重叠度 (2)前提/假设中是否存在否定词。"""
    p_tokens, h_tokens = tokenize(premise), tokenize(hypothesis)
    overlap = lexical_overlap(p_tokens, h_tokens)
    p_neg, h_neg = has_negation(p_tokens), has_negation(h_tokens)

    if overlap >= 0.5 and p_neg != h_neg:
        return "contradiction", overlap
    if overlap >= 0.5:
        return "entailment", overlap
    if overlap > 0 and p_neg != h_neg:
        return "contradiction", overlap
    return "neutral", overlap


def evaluate(examples):
    correct, confusion = 0, Counter()
    for premise, hypothesis, gold in examples:
        pred, conf = predict_nli(premise, hypothesis)
        ok = pred == gold
        correct += int(ok)
        confusion[(gold, pred)] += 1
        tag = "  OK" if ok else "MISS"
        print(f"  [{tag}] gold={gold:<13} pred={pred:<13} conf={conf:.2f}")
        print(f"         p: {premise}")
        print(f"         h: {hypothesis}")
    return correct, len(examples), confusion


def main():
    examples = [
        ("A cat is sleeping on the couch.", "There is a cat in the room.", "entailment"),
        ("A cat is sleeping on the couch.", "There is no cat in the room.", "contradiction"),
        ("A cat is sleeping on the couch.", "The dog chased the ball.", "neutral"),
        ("John walked his dog in the park.", "John has a dog.", "entailment"),
        ("John walked his dog in the park.", "John has no dog.", "contradiction"),
        ("The stock market rallied today.", "Stocks went up today.", "entailment"),
        ("The stock market rallied today.", "Stocks did not move today.", "contradiction"),
        ("She finished the marathon in three hours.", "She ran a marathon.", "entailment"),
        ("Birds were singing outside the window.", "The room was silent.", "neutral"),
        ("The chef served a tasty meal.", "The chef never cooked anything.", "contradiction"),
    ]
    print("=== 玩具 NLI 分类器（词汇重叠 + 否定检测）===\n")
    correct, total, confusion = evaluate(examples)
    print(f"\n准确率: {correct}/{total} ({100 * correct / total:.1f}%)")
    print("\n混淆矩阵 (真实→预测):")
    for (gold, pred), count in sorted(confusion.items()):
        print(f"  {gold:<14} → {pred:<14} {count}")

    print("\n注意：此分类器仅利用两个浅层特征。")
    print("生产 NLI 使用 DeBERTa-v3-MNLI——在 MNLI-matched 上约 91%。")
    print("任务形态——(前提, 假设) → 标签——完全相同。")


if __name__ == "__main__":
    main()
