# RAG——检索增强生成：完整管道

import numpy as np


# ============================================================================
# 第 1 步：文档嵌入
# ============================================================================

def mock_embed(text):
    """模拟嵌入——实际中用 text-embedding-3-small。"""
    words = text.lower().split()
    return np.random.randn(384)  # 模拟 384 维向量


class VectorStore:
    """简单向量存储。"""
    def __init__(self):
        self.docs = []
        self.vectors = []

    def add(self, doc, vec):
        self.docs.append(doc)
        self.vectors.append(np.array(vec, dtype=np.float32))

    def search(self, query_vec, top_k=5):
        query_vec = np.array(query_vec, dtype=np.float32)
        scores = [np.dot(query_vec, dv) / (np.linalg.norm(query_vec) * np.linalg.norm(dv) + 1e-10)
                  for dv in self.vectors]
        ranked = sorted(zip(range(len(self.docs)), scores), key=lambda x: -x[1])
        return [(self.docs[i], score) for i, score in ranked[:top_k]]


# ============================================================================
# 第 2 步：RAG 管道
# ============================================================================

def rag_pipeline(query, store, embed_fn, llm_fn, top_k=3):
    """标准 RAG：检索 → 增强 → 生成。"""
    # 1. 查询嵌入
    query_vec = embed_fn(query)

    # 2. 检索相关文档
    retrieved = store.search(query_vec, top_k=top_k)
    context = "\n".join([doc for doc, _ in retrieved])

    # 3. 构建增强提示词
    prompt = f"""基于以下文档回答用户的问题。如果没有答案，说"我没有足够的信息"。

文档：
{context}

问题：{query}
回答："""

    answer = llm_fn(prompt)
    return answer, retrieved


# ============================================================================
# 第 3 步：演示
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("RAG 管道演示\n")

    # 知识库
    store = VectorStore()
    docs = [
        "退货政策：30天内无理由退货，需保持原包装，运费由买家承担。",
        "配送范围：全国范围内免费配送，偏远地区可能额外收费。",
        "API 配置：在设置页面获取 API Key，使用 Bearer Token 认证。",
        "会员权益：免费配送、优先客服、生日礼券。",
        "退换货时限：签收后7天内可申请换货。",
    ]
    for doc in docs:
        store.add(doc, mock_embed(doc))

    # 模拟 LLM
    def llm_fn(prompt):
        return f"[模拟回复] 根据文档内容回答：{prompt[-50:]}..."

    # RAG 查询
    queries = ["退货政策是什么？", "怎么配置 API？"]
    for query in queries:
        answer, retrieved = rag_pipeline(query, store, mock_embed, llm_fn, top_k=2)
        print(f"查询: {query}")
        print(f"  检索: {[doc[:30] + '...' for doc, _ in retrieved]}")
        print(f"  回答: {answer[:80]}")
        print()
