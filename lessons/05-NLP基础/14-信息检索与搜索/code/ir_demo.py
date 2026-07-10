# ir_demo.py — 信息检索：BM25 + 稠密检索 + RRF融合
# 依赖：无
# 对应课程：阶段 05 · 14（信息检索与搜索）

import math, re
from collections import Counter
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """中文: 逐字切分; 英文: 正则匹配"""
    # 检测是否包含中文
    if any('一' <= c <= '鿿' for c in text):
        # 逐字 + 保留连续英文/数字
        return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """BM25——稀疏检索的工作马。

    k1=1.5: 词频饱和参数——词出现第2次的权重增量小于第1次
    b=0.75: 长度归一化——b=1→完全归一化, b=0→不归一化
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = [tokenize(d) for d in corpus]
        self.k1, self.b = k1, b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        doc, dl = self.corpus[doc_idx], len(self.corpus[doc_idx])
        freq = Counter(doc)
        total = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            total += self.idf(term) * num / den
        return total

    def rank(self, query: str, top_k: int = 10):
        return sorted([(self.score(query, i), i) for i in range(self.n_docs)],
                      reverse=True)[:top_k]


def reciprocal_rank_fusion(rankings: List, k: int = 60):
    """RRF——融合多个排序列表。只依赖排名位置，不依赖原始分数。"""
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    corpus = [
        "苹果公司于2007年6月29日发布了第一代iPhone。",
        "Macworld 2007大会上Steve Jobs展示了iPhone。",
        "Android于2008年作为Google的移动操作系统推出。",
        "第一代iPod于2001年由苹果公司发布。",
        "印度刑法第420条涉及欺诈和不诚实地诱取财物。",
        "欺诈是指以欺骗手段获取经济利益的犯罪行为。",
        "通过欺骗手段获取他人钱财在大多数司法管辖区构成刑事犯罪。",
    ]

    bm25 = BM25(corpus)

    # 关键词搜索 vs 语义搜索
    query = "如果有人撒谎来骗取钱财会怎样"
    print(f"查询: {query}")
    print(f"\nBM25 稀疏检索（精确匹配关键词）:")
    for score, idx in bm25.rank(query, top_k=5):
        print(f"  [{score:.3f}] {corpus[idx]}")

    # RRF 融合演示
    sparse = bm25.rank(query, top_k=5)
    # 模拟稠密检索——用 Jaccard 作为教学替代
    q_tokens = set(tokenize(query))
    dense = sorted([(len(q_tokens & set(tokenize(d))) / max(1, len(q_tokens | set(tokenize(d)))), i)
                    for i, d in enumerate(corpus)], reverse=True)[:5]
    fused = reciprocal_rank_fusion([sparse, dense])[:5]

    print(f"\nRRF 融合后（稀疏+稠密）:")
    for doc_idx, score in fused:
        print(f"  [{score:.4f}] {corpus[int(doc_idx)]}")

    print(f"\n{'='*50}")
    print("混合检索四层架构（2026生产默认）")
    print("=" * 50)
    print("1. BM25 → 精确匹配, <10ms, 百万文档")
    print("2. 稠密检索 → 语义匹配, 50-200ms, FAISS/向量库")
    print("3. RRF融合 → 合并两路排序（只用排名，不管分数尺度）")
    print("4. Cross-encoder 重排 → top-30 精排→top-5 最终返回")


if __name__ == "__main__":
    main()
