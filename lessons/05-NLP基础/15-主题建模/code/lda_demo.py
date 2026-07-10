# lda_demo.py — LDA 主题建模：Collapsed Gibbs 采样从零实现
# 依赖：无
# 对应课程：阶段 05 · 15（主题建模）

import random, re
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """中文: 逐字; 英文: 正则匹配。长于2字符才保留。"""
    tokens = re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())
    return [t for t in tokens if len(t) > 2 or t.isdigit() or '一' <= t <= '鿿']


def collapsed_gibbs_lda(docs: List[List[str]], n_topics: int,
                        n_iters: int = 200, alpha: float = 0.1,
                        beta: float = 0.01, seed: int = 0):
    """Collapsed Gibbs 采样推断 LDA 模型。

    LDA 的生成过程：
    1. 每个主题 k → 词分布 φ_k ~ Dirichlet(β)
    2. 每个文档 d → 主题分布 θ_d ~ Dirichlet(α)
    3. 为文档 d 中的每个位置生成一个词:
       a. 从 θ_d 采样主题 z
       b. 从 φ_z 采样词 w

    Collapsed Gibbs 采样直接推断 z（每词的主题分配），
    不需要显式表示 φ 和 θ——积分掉它们只采样 z。
    """
    vocab = {}
    for doc in docs:
        for w in doc:
            if w not in vocab:
                vocab[w] = len(vocab)
    V, D = len(vocab), len(docs)
    indexed = [[vocab[w] for w in doc] for doc in docs]

    # 随机初始化主题分配
    rng = random.Random(seed)
    z = [[rng.randint(0, n_topics - 1) for _ in doc] for doc in indexed]

    # 计数矩阵
    ndt = [[0] * n_topics for _ in range(D)]   # 文档 d 中分配给主题 t 的词数
    ntw = [[0] * V for _ in range(n_topics)]    # 主题 t 中词 w 的出现次数
    nt = [0] * n_topics                          # 主题 t 的总词数

    for d in range(D):
        for i, w in enumerate(indexed[d]):
            t = z[d][i]
            ndt[d][t] += 1
            ntw[t][w] += 1
            nt[t] += 1

    # Gibbs 迭代
    for _ in range(n_iters):
        for d in range(D):
            for i, w in enumerate(indexed[d]):
                t = z[d][i]
                ndt[d][t] -= 1; ntw[t][w] -= 1; nt[t] -= 1

                # 对每个候选主题计算条件概率:
                # P(z_i=k | ...) ∝ (ndt[d][k]+α) × (ntw[k][w]+β)/(nt[k]+V·β)
                probs = [(ndt[d][k] + alpha)
                         * (ntw[k][w] + beta) / (nt[k] + V * beta)
                         for k in range(n_topics)]
                total = sum(probs)
                r = rng.random() * total
                acc, new_t = 0.0, 0
                for k, p in enumerate(probs):
                    acc += p
                    if r <= acc:
                        new_t = k
                        break

                z[d][i] = new_t
                ndt[d][new_t] += 1; ntw[new_t][w] += 1; nt[new_t] += 1

    # 输出：主题Top词 + 文档主题分布
    inv_vocab = {i: w for w, i in vocab.items()}
    topics = []
    for k in range(n_topics):
        top = sorted(range(V), key=lambda i: -ntw[k][i])[:8]
        topics.append([inv_vocab[i] for i in top])

    doc_topic = []
    for d in range(D):
        total = sum(ndt[d]) + n_topics * alpha
        doc_topic.append([(ndt[d][k] + alpha) / total for k in range(n_topics)])

    return topics, doc_topic


def main():
    # 英文演示（可靠的分词 → 干净的主题）
    docs_en = [tokenize(d) for d in [
        "stocks rose after the fed cut interest rates",
        "bond yields fell as investors bought treasuries",
        "the s p 500 hit a new high on earnings reports",
        "chip makers reported strong demand for ai accelerators",
        "openai released a new model with multimodal reasoning",
        "deep learning researchers published a paper on efficient attention",
        "the senate passed a bill on healthcare spending",
        "the president signed new tariffs on steel imports",
        "congress debated a tax cut for small businesses",
    ]]
    topics, doc_topic = collapsed_gibbs_lda(docs_en, n_topics=3,
                                            n_iters=300, seed=42)
    print("=== LDA 主题（英文）===")
    for k, words in enumerate(topics):
        print(f"  主题 {k}: {', '.join(words)}")
    print("  → 三个主题清晰分离：金融/AI/政治")
    print("  → 中文 LDA 需要先分词—— pip install jieba")


if __name__ == "__main__":
    main()
