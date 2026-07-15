# 嵌入与向量搜索

import numpy as np


def cosine_similarity(a, b):
    """余弦相似度。"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


class SimpleVectorStore:
    """简单向量存储。"""
    def __init__(self):
        self.docs = []
        self.vectors = []

    def add(self, doc, vec):
        self.docs.append(doc)
        self.vectors.append(np.array(vec, dtype=np.float32))

    def search(self, query_vec, top_k=5):
        query_vec = np.array(query_vec, dtype=np.float32)
        scores = [cosine_similarity(query_vec, dv) for dv in self.vectors]
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return ranked[:top_k]


def mock_embed(text):
    """模拟嵌入——使用词频向量。"""
    words = text.lower().split()
    vocab = list(set(w for doc in store.docs for w in doc.lower().split()))
    vec = np.array([words.count(w) for w in vocab], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-10) if norm > 0 else vec


if __name__ == "__main__":
    store = SimpleVectorStore()
    store.add("如何用 Python 连接 MySQL 数据库", mock_embed("python mysql 数据库"))
    store.add("Python SQLAlchemy ORM 教程", mock_embed("python sqlalchemy orm"))
    store.add("JavaScript 连接 MongoDB", mock_embed("javascript mongodb"))
    store.add("Python Django Web 开发", mock_embed("python django web"))
    store.add("SQL 查询优化技巧", mock_embed("sql 查询优化"))

    query = "用 Python 查询数据库"
    query_vec = mock_embed(query)
    results = store.search(query_vec, top_k=3)
    print("搜索结果:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")
