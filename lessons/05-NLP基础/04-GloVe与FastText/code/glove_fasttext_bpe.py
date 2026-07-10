# glove_fasttext_bpe.py — GloVe 共现矩阵 + FastText 子词嵌入 + BPE 分词
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 05 · 04（GloVe 与 FastText）

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set


# ============================================================
# 1. GloVe：共现矩阵构建
# ============================================================

def build_cooccurrence(docs: List[List[str]],
                       window: int = 5) -> Tuple[Dict[str, int],
                                                  Counter]:
    """构建词-词共现矩阵。

    与 Word2Vec 的 skipgram 对不同，GloVe 使用距离衰减权重：
    距离越远的上下文词，权重越低（1/distance）。

    X[i][j] = 词 j 在词 i 的上下文窗口中出现的加权次数。
    """
    vocab: Dict[str, int] = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    pair_counts: Counter = Counter()
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window),
                           min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    # 距离越远，权重越低
                    pair_counts[(center, indexed[j])] += 1.0 / distance

    return vocab, pair_counts


# ============================================================
# 2. GloVe：加权回归训练
# ============================================================

def glove_train(vocab: Dict[str, int],
                pair_counts: Counter,
                dim: int = 16,
                epochs: int = 100,
                lr: float = 0.05,
                x_max: int = 100,
                alpha: float = 0.75,
                seed: int = 0) -> np.ndarray:
    """训练 GloVe 嵌入。

    GloVe 的核心思想：
      向量点积 + 偏置 ≈ log(共现次数)

    即：W[i] · W_tilde[j] + b[i] + b_tilde[j] ≈ log(X[i][j])

    损失函数：weight(X_ij) * (预测 - 实际)^2

    weight 函数：f(x) = (x/x_max)^alpha  当 x < x_max，否则 1.0
    这个函数压制高频共现对（如 the-and 出现百万次）对损失的支配。
    """
    n = len(vocab)
    rng = np.random.default_rng(seed)

    # 两个嵌入表 + 偏置
    W = rng.normal(0, 0.1, size=(n, dim))        # 中心词嵌入
    W_tilde = rng.normal(0, 0.1, size=(n, dim))  # 上下文嵌入
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            # 权重：压制高频对
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0

            # 预测值 vs 目标值
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            # 梯度更新
            W[i] -= lr * coef * W_tilde[j]
            W_tilde[j] -= lr * coef * W[i]
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    # 最终嵌入 = 中心词表 + 上下文表（公开的技巧：求和比单用效果好）
    return W + W_tilde


# ============================================================
# 3. FastText：字符 n-gram
# ============================================================

def char_ngrams(word: str,
                n_min: int = 3,
                n_max: int = 6) -> Set[str]:
    """提取一个词的字符 n-gram 集合。

    词 "<where>" 的 3-6 gram：
      <wh, whe, her, ere, re>, <whe, wher, here, ere>,
      <wher, where, here>, <where>

    每个词还保留其完整形式 <word> 作为一个特殊的 n-gram。
    """
    wrapped = f"<{word}>"
    grams = {wrapped}  # 完整词形
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams


# ============================================================
# 4. BPE：从零学习 + 应用
# ============================================================

def learn_bpe(corpus: Dict[str, int],
              k_merges: int) -> List[Tuple[str, str]]:
    """从词-频词典中学习 BPE 合并规则。

    输入 corpus = {"low": 5, "lower": 2, "newest": 6, ...}
    表示语料中 "low" 出现 5 次，"lower" 出现 2 次...

    算法：
    1. 初始词表 = 所有字符 + 词尾标记 </w>
    2. 统计所有相邻字符对的频率（乘以词频）
    3. 合并频率最高的一对 → 新 token
    4. 重复 k 次
    """
    # 初始化：将每个词拆为字符序列
    vocab: Dict[Tuple[str, ...], int] = {}
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges: List[Tuple[str, str]] = []
    for _ in range(k_merges):
        # 统计所有相邻对的频率
        pair_freq: Counter = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq

        if not pair_freq:
            break

        # 选择最高频的一对
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        # 将选中的对合并
        new_vocab: Dict[Tuple[str, ...], int] = {}
        for tokens, freq in vocab.items():
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if (i + 1 < len(tokens) and
                        tokens[i] == best[0] and
                        tokens[i + 1] == best[1]):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab

    return merges


def apply_bpe(word: str,
              merges: List[Tuple[str, str]]) -> List[str]:
    """将学到的 BPE 合并规则应用于一个新词。

    合并顺序 == 学习顺序——这就是为什么 merge table 的 order 至关重要。
    """
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens: List[str] = []
        i = 0
        while i < len(tokens):
            if (i + 1 < len(tokens) and
                    tokens[i] == a and
                    tokens[i + 1] == b):
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === FastText 演示 ===
    print("=" * 60)
    print("FastText：字符 n-gram 拆解")
    print("=" * 60)

    for word in ["where", "whereupon", "playing", "played"]:
        grams = sorted(char_ngrams(word))
        print(f"\n{word} → {len(grams)} 个 n-gram")
        print(f"  示例: {grams[:6]}...")

    # 关键对比：OOV 词与其已知近亲共享多少 n-gram
    g_where = char_ngrams("where")
    g_whereupon = char_ngrams("whereupon")
    shared = g_where & g_whereupon
    print(f"\n'where' 与 'whereupon' 共享 {len(shared)} 个 n-gram: "
          f"{sorted(shared)[:5]}...")
    print("→ 这就是 FastText 能为 OOV 词生成合理向量的原因。")

    # Jaccard 相似度：playing vs played（英语时态变化）
    g_playing = char_ngrams("playing")
    g_played = char_ngrams("played")
    jaccard = len(g_playing & g_played) / len(g_playing | g_played)
    print(f"\n'playing' vs 'played' Jaccard 重叠度: {jaccard:.2%}")
    print("→ 英文时态变体共享大量子词，FastText 天然对形态变化有鲁棒性。")

    # === GloVe 演示 ===
    print("\n" + "=" * 60)
    print("GloVe：共现矩阵构建")
    print("=" * 60)

    docs = [
        ["the", "cat", "sat", "on", "mat"],
        ["the", "dog", "sat", "on", "rug"],
        ["the", "cat", "ate", "food"],
        ["a", "dog", "chased", "a", "cat"],
    ]
    vocab, pair_counts = build_cooccurrence(docs, window=3)

    # 展示共现矩阵
    words = sorted(vocab, key=lambda w: vocab[w])
    print(f"词表: {words}")
    print(f"共现对数量: {len(pair_counts)}")
    print("Top 5 共现对:")
    for (i, j), count in pair_counts.most_common(5):
        print(f"  ({words[i]}, {words[j]}): {count:.1f}")

    # === BPE 演示 ===
    print("\n" + "=" * 60)
    print("BPE：子词合并学习")
    print("=" * 60)

    corpus = Counter({
        "low": 5, "lower": 2, "newest": 6, "widest": 3,
        "lowest": 4, "newer": 2,
    })
    merges = learn_bpe(corpus, k_merges=10)
    print(f"学习了 {len(merges)} 条合并规则:")
    for a, b in merges:
        print(f"  '{a}' + '{b}' → '{a + b}'")

    print("\nBPE 分词结果:")
    for test in ["lowest", "slowest", "lower", "newish"]:
        tokens = apply_bpe(test, merges)
        print(f"  {test:10s} → {tokens}")

    # 关键洞察
    print(f"\n关键洞察：")
    print(f"  'lowest' 被切成 {apply_bpe('lowest', merges)}")
    print(f"  'slowest' 被切成 {apply_bpe('slowest', merges)} "
          f"— 虽然没见过，但 'low' + 'est' 的规则可以复用")

    # === 中文 BPE 演示 ===
    print("\n" + "=" * 60)
    print("中文 BPE 演示（以字为基本单元）")
    print("=" * 60)
    zh_corpus = Counter({
        "机器学习": 10, "深度学习": 8, "学习算法": 6,
        "机器翻译": 5, "学习数据": 4, "深度网络": 3,
    })
    zh_merges = learn_bpe(zh_corpus, k_merges=8)
    print(f"学习了 {len(zh_merges)} 条合并规则:")
    for a, b in zh_merges:
        print(f"  '{a}' + '{b}' → '{a + b}'")

    print("\n中文 BPE 分词结果:")
    for test in ["机器学习", "深度学习", "强化学习", "机器视觉"]:
        tokens = apply_bpe(test, zh_merges)
        print(f"  {test:10s} → {tokens}")

    print("\n注意：'强化学习'和'机器视觉'未出现在训练语料中，")
    print("但 BPE 自动将已知子词模式复用——它们被合理切分。")


if __name__ == "__main__":
    main()
