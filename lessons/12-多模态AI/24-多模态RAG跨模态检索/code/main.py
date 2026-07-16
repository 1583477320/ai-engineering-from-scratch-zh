# 多模态 RAG 跨模态检索

import numpy as np


def cross_modal_similarity(query_embed, doc_embeds):
    """计算跨模态相似度。"""
    results = []
    for doc_id, doc_emb in doc_embeds.items():
        sim = np.dot(query_embed, doc_emb) / (np.linalg.norm(query_embed) * np.linalg.norm(doc_emb) + 1e-10)
        results.append((doc_id, sim))
    return sorted(results, key=lambda x: -x[1])


def rrf_fusion(result_lists, k=60):
    """RRF 融合多路检索结果。"""
    scores = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


if __name__ == "__main__":
    print("多模态 RAG 检索演示\n")
    query_text = np.random.randn(128)
    query_image = np.random.randn(128)

    text_docs = {f"doc{i}": np.random.randn(128) for i in range(5)}
    image_docs = {f"doc{i}": np.random.randn(128) for i in range(5)}

    text_results = cross_modal_similarity(query_text, text_docs)
    image_results = cross_modal_similarity(query_image, image_docs)

    fused = rrf_fusion([text_results[:3], image_results[:3]])
    print("RRF 融合结果:")
    for doc_id, score in fused[:3]:
        print(f"  {doc_id}: {score:.4f}")
