# ColPali 风格多向量检索

import numpy as np


class SimpleVisualIndex:
    """简化版多向量索引。"""
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, doc_id, embeddings):
        self.documents.append(doc_id)
        self.embeddings.append(embeddings)

    def search(self, query_vecs, top_k=5):
        scores = []
        for i, doc_vecs in enumerate(self.embeddings):
            score = self._maxsim(query_vecs, doc_vecs)
            scores.append((self.documents[i], score))
        return sorted(scores, key=lambda x: -x[1])[:top_k]

    def _maxsim(self, query_vecs, doc_vecs):
        total = 0
        for q in query_vecs:
            sims = [np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-10)
                    for d in doc_vecs]
            total += max(sims) if sims else 0
        return total / max(len(query_vecs), 1)


if __name__ == "__main__":
    print("ColPali 多向量检索演示\n")
    index = SimpleVisualIndex()
    index.index("doc1", np.random.randn(16, 128))
    index.index("doc2", np.random.randn(16, 128))
    index.index("doc3", np.random.randn(16, 128))

    query = np.random.randn(8, 128)
    results = index.search(query, top_k=3)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f}")
