# 高级 RAG：查询重写、RRF 融合、重排序

import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


class VectorStore:
    def __init__(self):
        self.docs, self.vectors = [], []
    def add(self, doc, vec):
        self.docs.append(doc); self.vectors.append(np.array(vec, dtype=np.float32))
    def search(self, query_vec, top_k=5):
        scores = [np.dot(np.array(query_vec), dv) / (np.linalg.norm(query_vec) * np.linalg.norm(dv) + 1e-10) for dv in self.vectors]
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return ranked[:top_k]


def reciprocal_rank_fusion(result_lists, k=60):
    """RRF 融合多路检索结果。"""
    from collections import defaultdict
    scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, 1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank(query, candidates, cross_encoder_fn, top_k=5):
    """用交叉编码器重排序。"""
    return sorted([(doc, cross_encoder_fn(query, doc)) for doc in candidates], key=lambda x: -x[1])[:top_k]


if __name__ == "__main__":
    np.random.seed(42)
    print("高级 RAG 策略演示\n")

    store = VectorStore()
    for doc in ["Python SQL 查询优化", "JavaScript 异步处理", "Python 网络爬虫", "SQL 索引优化"]:
        store.add(doc, np.random.randn(128))

    # 多路检索
    vec_results = store.search(np.random.randn(128), top_k=3)
    keyword_results = [("SQL 索引优化", 0.95), ("Python 异步处理", 0.8), ("Python 网络爬虫", 0.7)]
    fused = reciprocal_rank_fusion([vec_results, keyword_results])
    print("RRF 融合结果:")
    for doc, score in fused[:3]:
        print(f"  [{score:.4f}] {doc}")
