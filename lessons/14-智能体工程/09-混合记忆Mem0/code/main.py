# 混合记忆：向量+KV+图

import numpy as np


class HybridMemory:
    """混合记忆——三路并行存储。"""
    def __init__(self):
        self.vector_store = {}
        self.kv_store = {}
        self.graph_store = {}

    def store(self, key, value, embedding=None, relations=None):
        self.kv_store[key] = value
        if embedding is not None:
            self.vector_store[key] = embedding
        if relations is not None:
            self.graph_store[key] = relations

    def search_vector(self, query_embed, top_k=3):
        results = []
        for key, emb in self.vector_store.items():
            sim = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb) + 1e-10)
            results.append((key, sim))
        return sorted(results, key=lambda x: -x[1])[:top_k]

    def search_kv(self, query_key):
        return self.kv_store.get(query_key, None)

    def search_graph(self, entity):
        return self.graph_store.get(entity, [])

    def hybrid_search(self, query_embed, query_key, weights=(0.4, 0.4, 0.2)):
        v_results = self.search_vector(query_embed, top_k=3)
        combined = {}
        for key, score in v_results:
            combined[key] = weights[0] * score
        k_result = self.search_kv(query_key)
        if k_result:
            combined[k_result] = combined.get(k_result, 0) + weights[1]
        return sorted(combined.items(), key=lambda x: -x[1])


if __name__ == "__main__":
    print("混合记忆演示\n")
    mem = HybridMemory()
    mem.store("用户邮箱", "alice@example.com", embedding=np.random.randn(128), relations=["Alice→公司A"])
    mem.store("Python教程", "入门指南", embedding=np.random.randn(128))
    results = mem.hybrid_search(np.random.randn(128), "用户邮箱")
    print("混合检索结果:")
    for key, score in results[:3]:
        print(f"  {key}: {score:.4f}")
