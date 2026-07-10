# ngram_lm.py — N-gram 语言模型：Laplace + Kneser-Ney 平滑 + 困惑度
# 依赖：无
# 对应课程：阶段 05 · 16（Transformer 之前的文本生成）

import math, random
from collections import Counter, defaultdict
from typing import List, Callable


def tokenize(text: str) -> List[str]:
    return text.lower().replace(".", " .").replace(",", " ,").split()


def train_bigrams(sentences: List[List[str]]):
    bigrams = Counter()
    unigrams = Counter()
    unigram_contexts = defaultdict(set)
    for s in sentences:
        padded = ["<s>"] + s + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)
    return bigrams, unigrams, unigram_contexts


def kneser_ney_prob(bigrams, unigram_contexts, context_totals,
                    unique_follow, total_unique_bigrams,
                    prev, w, discount=0.75):
    """Kneser-Ney 平滑——N-gram 语言模型的精髓。

    核心洞察：'Francisco' 在语料中高频，但几乎只出现在 'San' 之后。
    普通的绝对折扣会给 'Francisco' 高概率，
    Kneser-Ney 用接续概率（出现在多少个不同的上下文中）替代原始频率。
    """
    count = bigrams.get((prev, w), 0)
    denom = context_totals.get(prev, 0)
    continuation = len(unigram_contexts.get(w, set())) / max(total_unique_bigrams, 1)
    if denom == 0:
        return continuation or 1e-9
    first = max(count - discount, 0) / denom
    lam = discount * len(unique_follow[prev]) / denom
    return first + lam * continuation


def laplace_prob(bigrams, unigrams, vocab_size, prev, w):
    num = bigrams.get((prev, w), 0) + 1
    return num / (unigrams.get(prev, 0) + vocab_size)


def perplexity(prob_fn: Callable, sentences: List[List[str]]) -> float:
    """困惑度 = exp(-平均log概率)。越低越好——更少的'惊讶'。"""
    total_log, total = 0.0, 0
    for s in sentences:
        padded = ["<s>"] + s + ["</s>"]
        for i in range(1, len(padded)):
            total_log += math.log(max(prob_fn(padded[i - 1], padded[i]), 1e-12))
            total += 1
    return math.exp(-total_log / total)


def sample_sentence(prob_fn, vocab, max_len=15, seed=0):
    rng = random.Random(seed)
    tokens = ["<s>"]
    for _ in range(max_len):
        probs = [(w, prob_fn(tokens[-1], w)) for w in vocab if w != "<s>"]
        total = sum(p for _, p in probs)
        r, acc = rng.random() * total, 0.0
        for w, p in probs:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            return tokens[1:-1]
    return tokens[1:]


def main():
    raw = [
        "the cat sat on the mat .",
        "the cat ran across the room .",
        "the dog sat by the window .",
        "a cat chased the mouse .",
        "the dog ran after the cat .",
        "a mouse hid under the table .",
        "the cat watched the birds .",
        "the dog chased the ball .",
        "a bird sat on the branch .",
        "the cat slept on the couch .",
    ]
    train = [tokenize(s) for s in raw]
    test = [tokenize("the cat sat on the couch ."),
            tokenize("the dog watched the mouse .")]

    bigrams, unigrams, unigram_contexts = train_bigrams(train)
    vocab = list(unigrams.keys())
    vocab_size = len(vocab)
    context_totals = Counter()
    unique_follow = defaultdict(set)
    for (prev, w), c in bigrams.items():
        context_totals[prev] += c
        unique_follow[prev].add(w)
    total_unique_bigrams = sum(len(s) for s in unigram_contexts.values())

    def kn(prev, w):
        return kneser_ney_prob(bigrams, unigram_contexts, context_totals,
                               unique_follow, total_unique_bigrams, prev, w)
    def lap(prev, w):
        return laplace_prob(bigrams, unigrams, vocab_size, prev, w)

    print(f"词表大小: {vocab_size}")
    print(f"困惑度 (Laplace):     {perplexity(lap, test):.2f}")
    print(f"困惑度 (Kneser-Ney):  {perplexity(kn, test):.2f}")
    print("  → 越低越好。Kneser-Ney 平滑给未见过的搭配更低的概率。")
    print("\n=== 采样生成 (Kneser-Ney) ===")
    for seed in [1, 7, 42]:
        sent = sample_sentence(kn, vocab, seed=seed)
        print(f"  seed={seed}: {' '.join(sent)}")


if __name__ == "__main__":
    main()
