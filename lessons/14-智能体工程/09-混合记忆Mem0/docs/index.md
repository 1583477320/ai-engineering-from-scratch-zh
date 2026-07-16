# 混合记忆：向量+图+KV（Mem0）

> Mem0（Chhikara 等人，2025）将记忆视为三个并行的存储——向量用于语义相似度，KV 用于快速事实查找，图用于实体关系推理。一个评分层在检索时融合三个存储。这是 2026 年外部记忆的生产标准。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）、08（Letta Blocks）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么单一存储（仅向量、仅图、仅 KV）不足以支撑智能体记忆
- [ ] 实现混合记忆检索——融合三种存储的分数
- [ ] 理解 Mem0 的三存储架构及其各自的优劣
- [ ] 设计混合记忆的评分融合策略

---

## 1. 问题

MemGPT/Letta 用主上下文+归档存储的二元结构。但生产级智能体需要更复杂的记忆——语义搜索（"类似这个问题之前怎么回答的？"）、快速查找（"用户的邮箱是什么？"）、关系推理（"这个用户和那个项目有什么关联？"）。

**单一存储无法满足所有需求。** 向量搜索擅长语义但不擅长精确查找，KV 存储擅长精确查找但不擅长语义搜索，图数据库擅长关系推理但不擅长文本匹配。

---

## 2. 概念

### 2.1 Mem0 三存储架构

| 存储类型 | 用途 | 查询方式 | 优势 |
|---------|------|---------|------|
| **向量存储** | 语义相似度搜索 | 余弦相似度 | 找相似问题的答案 |
| **KV 存储** | 快速事实查找 | 精确键匹配 | 查"用户的邮箱是什么？" |
| **图存储** | 实体关系推理 | 图遍历/子图匹配 | "这个用户和那个项目的关系？" |

### 2.2 检索融合

```
查询 → 三路并行检索
  ├── 向量检索 → 分数 V
  ├── KV 查找 → 分数 K
  └── 图检索 → 分数 G
      ↓
融合层: 综合分数 = w_V * V + w_K * K + w_G * G
      ↓
返回 Top-K 结果
```

### 2.3 评分融合策略

| 策略 | 方法 |
|------|------|
| 加权平均 | 综合 = w1*V + w2*K + w3*G |
| 专家选择 | 根据查询类型选择最佳存储 |
| 竞赛排名 | 每个存储返回 Top-N，RRF 融合 |

---

## 3. 从零实现

### Step 1：混合记忆检索

```python
class HybridMemory:
    """混合记忆——向量+KV+图。"""
    def __init__(self):
        self.vector_store = {}  # 语义搜索
        self.kv_store = {}      # 精确查找
        self.graph_store = {}   # 关系图

    def store(self, key, value, embedding=None, relations=None):
        """存储到三种存储。"""
        self.kv_store[key] = value
        if embedding is not None:
            self.vector_store[key] = embedding
        if relations is not None:
            self.graph_store[key] = relations

    def search_vector(self, query_embed, top_k=3):
        """向量搜索。"""
        results = []
        for key, emb in self.vector_store.items():
            sim = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb) + 1e-10)
            results.append((key, sim))
        return sorted(results, key=lambda x: -x[1])[:top_k]

    def search_kv(self, query_key):
        """KV 精确查找。"""
        return self.kv_store.get(query_key, None)

    def search_graph(self, entity):
        """图关系搜索。"""
        if entity in self.graph_store:
            return self.graph_store[entity]
        return []

    def hybrid_search(self, query_embed, query_key, weights=(0.4, 0.4, 0.2)):
        """融合三路检索。"""
        v_results = self.search_vector(query_embed, top_k=3)
        k_result = self.search_kv(query_key)

        # 融合分数
        combined = {}
        for key, score in v_results:
            combined[key] = weights[0] * score
        if k_result:
            combined[k_result] = combined.get(k_result, 0) + weights[1]

        return sorted(combined.items(), key=lambda x: -x[1])


if __name__ == "__main__":
    print("混合记忆 Mem0 演示\n")
    mem = HybridMemory()
    mem.store("用户邮箱", "alice@example.com", embedding=np.random.randn(128), relations=["Alice→公司A"])
    mem.store("公司A地址", "北京朝阳区", embedding=np.random.randn(128), relations=["公司A→北京"])
    mem.store("Python教程", "Python入门指南", embedding=np.random.randn(128))

    query_embed = np.random.randn(128)
    results = mem.hybrid_search(query_embed, "用户邮箱")
    print("混合检索结果:")
    for key, score in results[:3]:
        print(f"  {key}: {score:.4f}")

---

## 4. 工具

### 4.1 Mem0 框架

```python
# Mem0: https://github.com/mem0ai/mem0
```

### 4.2 向量数据库

| 数据库 | 特点 |
|--------|------|
| Chroma | 轻量级、嵌入式 |
| Pinecone | 托管、可扩展 |
| Qdrant | 高性能、Rust |

---

## 5. 工程最佳实践

### 5.1 混合记忆设计

- **向量存储**：语义搜索——找到相似问题的答案
- **KV 存储**：精确查找——查"用户的邮箱是什么？"
- **图存储**：关系推理——"这个用户和那个项目的关系？"

### 5.2 踩坑经验

- **单一存储不足**：向量搜索不擅长精确查找，KV 不擅长语义搜索
- **融合权重不当**：需要根据查询类型动态调整权重
- **存储爆炸**：定期清理过期或低价值记忆

---

## 6. 常见错误

### 错误 1：只使用向量存储

**现象：** 精确查询返回不相关结果。

**修复：** 添加 KV 存储支持精确查找。

---

## 7. 面试考点

### Q1：为什么需要三种存储？（难度：⭐⭐）

**参考答案：** 向量擅长语义、KV 擅长精确查找、图擅长关系推理。三者互补。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 混合记忆 | "三路并行" | 向量+KV+图三种存储并行融合 |
| 评分融合 | "综合打分" | 多种检索结果按权重合并 |

---

## 📚 小结

Mem0 将记忆视为三种并行存储——向量（语义）、KV（精确）、图（关系）。评分层融合。

---

## ✏️ 练习

1. **【实现】** 构建混合记忆检索
2. **【设计】** 为客服智能体设计混合记忆

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 混合记忆检索 | `code/main.py` | 向量+KV+图三路融合 |

---

## 📖 参考资料

1. [论文] Chhikara et al. "Mem0". 2025.
2. [GitHub] Mem0: https://github.com/mem0ai/mem0

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
