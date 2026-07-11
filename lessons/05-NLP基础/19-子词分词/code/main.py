# BPE 子词分词——从零训练 + 编码
# 对应课程：阶段 05 · 19

import re
from collections import Counter


def word_counts(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return Counter(words)


def init_vocab(counts):
    return {tuple(word) + ("</w>",): freq for word, freq in counts.items()}


def pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(vocab, pair):
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged); i += 2
            else:
                new_symbols.append(symbols[i]); i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def train_bpe(text, num_merges):
    """BPE 训练：字符级词表出发→贪心合并最高频相邻对→重复 k 次。"""
    counts = word_counts(text)
    if not counts: raise ValueError("语料未产生任何词")
    vocab = init_vocab(counts)
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(vocab)
        if not pairs: break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = merge_pair(vocab, best)
    final_tokens = set()
    for symbols in vocab:
        final_tokens.update(symbols)
    return merges, sorted(final_tokens)


def encode_bpe(word, merges):
    """用学到的合并规则编码新词。合并顺序=训练顺序——order matters。"""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        merged, i = a + b, 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [merged] + symbols[i + 2:]
            else: i += 1
    return symbols


def main():
    corpus = """
    the quick brown fox jumps over the lazy dog
    language models learn from statistical patterns in text
    tokenization splits text into smaller units called tokens
    subword tokenization lets rare words decompose into known pieces
    byte pair encoding is the dominant tokenization algorithm today
    """
    merges_30, tokens_30 = train_bpe(corpus, num_merges=30)
    merges_150, tokens_150 = train_bpe(corpus, num_merges=150)

    print(f"=== BPE, 30 次合并 ===")
    print(f"词表大小: {len(tokens_30)}")
    print("前 10 条合并规则:")
    for i, m in enumerate(merges_30[:10]):
        print(f"  {i}: {m[0]!r} + {m[1]!r} → {m[0] + m[1]!r}")

    print(f"\n=== BPE, 150 次合并 ===")
    print(f"词表大小: {len(tokens_150)}")

    print(f"\n=== 留出词编码（150-合并模型）===")
    for word in ["tokenizable", "unlearnable", "patterns", "languages"]:
        pieces = encode_bpe(word, merges_150)
        tag = "OK" if len(pieces) == 1 else f"拆分({len(pieces)}块)"
        print(f"  {word:<14} → {' | '.join(pieces)}  [{tag}]")

    print("\n注意：玩具语料上大多数留出词会被拆分。")
    print("生产词表在数十亿 token 上训练——常见词保持完整。")


if __name__ == "__main__":
    main()
