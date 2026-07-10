# lda_demo.py — LDA 主题建模：Collapsed Gibbs 采样从零实现
# 依赖：无
# 对应课程：阶段 05 · 15（主题建模）

import random, re
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 or t.isdigit()]


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
    docs_raw = [
        "股票在美联储降息后上涨",
        "债券收益率因投资者买入国债而下跌",
        "标普500指数在企业财报推动下创出新高",
        "芯片制造商报告AI加速器的强劲需求",
        "OpenAI发布了具有多模态推理能力的新模型",
        "深度学习研究人员发表了一篇关于高效注意力机制的论文",
        "参议院通过了一项关于医疗支出的法案",
        "总统签署了对钢铁进口的新关税",
        "国会讨论了针对小企业的减税方案",
    ]

    try:
        import jieba
        docs = [list(jieba.cut(d)) for d in docs_raw]
    except ImportError:
        docs = [tokenize(d) for d in docs_raw]

    topics, doc_topic = collapsed_gibbs_lda(docs, n_topics=3,
                                            n_iters=300, seed=42)

    print("=== LDA 主题（Collapsed Gibbs, 300 迭代）===")
    labels = ["金融", "AI", "政策"]
    for k, words in enumerate(topics):
        print(f"  主题 {k} ({labels[k]}): {', '.join(words)}")
    print(f"\n=== 文档主题分布 ===")
    for doc, mix in zip(docs_raw, doc_topic):
        top = max(range(3), key=lambda k: mix[k])
        print(f"  [{', '.join(f'{p:.2f}' for p in mix)}] → 主题{top} | {doc[:45]}")

    print(f"\nLDA vs BERTopic: LDA=文档可混合多主题; BERTopic=1文档1主题+聚类")


if __name__ == "__main__":
    main()
