# 多模态 RAG 与跨模态检索

> 视觉原生文档 RAG 只是一个切片。生产级多模态 RAG 覆盖更广——跨文本、图像、音频和视频检索，支持旅行规划（"找一家安静的素食早午餐店，自然光"）、医疗分诊（"这个伤情匹配这张照片+这些笔记"）、电商（"类似这张自拍的服装，我的尺码"）等工作流。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 23（ColPali）、阶段 11（RAG 基础）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计跨模态检索——文本→图像、图像→文本、音频→视频等
- [ ] 实现检索融合——多模态检索结果的排序和融合
- [ ] 理解生成接地——检索结果如何影响多模态生成
- [ ] 设计一个生产级多模态 RAG 管道

---

## 1. 问题

传统 RAG 只处理文本。但用户的需求往往是多模态的——"这张 X 光片有什么问题"需要同时理解图像和文本；"类似这张自拍的服装"需要图像到图像的检索；"类似这段音乐的视频"需要音频到视频的检索。

生产级多模态 RAG 需要解决三个子问题：**跨模态检索**（找到匹配的内容）、**检索融合**（排序多模态结果）、**生成接地**（检索结果如何指导生成）。

---

## 2. 概念

### 2.1 跨模态检索

| 查询类型 | 检索目标 | 方法 |
|---------|---------|------|
| 文本→图像 | 找到与文本描述匹配的图像 | 文本嵌入 vs 图像嵌入 |
| 图像→文本 | 找到描述图像的文本 | 图像嵌入 vs 文本嵌入 |
| 图像→图像 | 找到视觉上相似的图像 | 图像嵌入 vs 图像嵌入 |
| 音频→视频 | 找到包含特定声音的视频 | 音频嵌入 vs 视频帧嵌入 |

### 2.2 检索融合

多种模态的检索结果需要融合排序：

```python
def multi_modal_fusion(text_scores, image_scores, audio_scores, weights=(0.4, 0.4, 0.2)):
    """加权融合多模态检索结果。"""
    fused = {}
    for doc_id in set(list(text_scores) + list(image_scores) + list(audio_scores)):
        score = (text_scores.get(doc_id, 0) * weights[0] +
                 image_scores.get(doc_id, 0) * weights[1] +
                 audio_scores.get(doc_id, 0) * weights[2])
        fused[doc_id] = score
    return sorted(fused.items(), key=lambda x: -x[1])
```

### 2.3 生成接地

检索到的内容如何影响多模态生成：

```
多模态查询 → [跨模态检索] → 相关内容
                                ↓
                        [生成模型] → 基于检索内容生成
```

---

## 3. 从零实现

### Step 1：跨模态相似度

```python
def cross_modal_similarity(query_embed, doc_embeds, query_type="text", doc_type="image"):
    """计算跨模态相似度。"""
    similarities = []
    for doc_id, doc_emb in doc_embeds.items():
        sim = np.dot(query_embed, doc_emb) / (np.linalg.norm(query_embed) * np.linalg.norm(doc_emb) + 1e-10)
        similarities.append((doc_id, sim))
    return sorted(similarities, key=lambda x: -x[1])
```

### Step 2：检索融合

```python
def rrf_fusion(result_lists, k=60):
    """RRF 融合多路检索结果。"""
    scores = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

---

## 4. 工具

### 4.1 跨模态嵌入模型

| 模型 | 支持模态 | 用途 |
|------|---------|------|
| CLIP | 文本↔图像 | 文本-图像检索 |
| ImageBind | 文本↔图像↔音频↔视频 | 统一跨模态 |
| CLAP | 文本↔音频 | 音频检索 |
| BGE-M3 | 文本多语言 | 多语言文本检索 |

---

## 6. 工程最佳实践

### 6.1 跨模态检索管道

```
1. 查询理解：将用户查询分解为模态和意图
2. 模态路由：决定使用哪些检索通道
3. 并行检索：文本通道 + 图像通道 + 音频通道
4. RRF 融合：合并多路结果
5. 重排序：用交叉编码器精选
6. 生成接地：用检索结果增强生成
```

### 6.2 踩坑经验

- **模态权重不对**：文本权重太高导致忽略视觉信息——需要按任务调整
- **索引存储爆炸**：多模态索引存储量是纯文本的 5-10 倍
- **跨模态对齐困难**：不同模态的嵌入空间需要预训练对齐（如 CLIP）

---

## 7. 常见错误

### 错误 1：忽略模态权重

**现象：** 检索结果偏重文本——图像和音频信息被忽略。

**修复：** 根据查询类型动态调整模态权重——图像查询增大图像通道权重。

---

## 8. 面试考点

### Q1：跨模态检索和单模态检索有什么区别？（难度：⭐⭐）

**参考答案：**
单模态检索（如纯文本 RAG）只在一个嵌入空间中做向量匹配。跨模态检索需要在多个嵌入空间中做匹配——文本嵌入、图像嵌入、音频嵌入可能不在同一个空间中。解决方案：(1) 联合嵌入（如 CLIP 将文本和图像嵌入同一空间）；(2) 跨模态投影（将不同模态映射到共享空间）；(3) 跨模态 RRF 融合（多个单模态检索结果加权融合）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 跨模态检索 | "多模态搜索" | 在多个模态的嵌入空间中检索匹配内容 |
| 检索融合 | "合并多路结果" | 将多种模态的检索结果通过 RRF 或加权融合统一排序 |
| 生成接地 | "用检索指导生成" | 检索结果作为上下文增强多模态生成 |
| ImageBind | "统一嵌入空间" | Meta 的模型——将六种模态嵌入同一向量空间 |

---

## 📚 小结

多模态 RAG 跨文本、图像、音频、视频检索。核心：跨模态嵌入（CLIP/ImageBind）+ 检索融合（RRF）+ 生成接地（检索结果增强生成）。生产管道：查询理解→模态路由→并行检索→融合→重排序→生成。

---

## ✏️ 练习

1. **【设计】** 为医疗影像 RAG 设计管道——CT 图像 + 文本报告 + 检索融合
2. **【实现】** 实现 RRF 融合——对比纯文本检索和跨模态融合的准确率

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 跨模态检索 | `code/main.py` | 跨模态相似度 + RRF 融合 |

---

## 📖 参考资料

1. [论文] Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" (CLIP). ICML, 2021.
2. [论文] Girdhar et al. "ImageBind: One Embedding Space To Bind Them All". CVPR, 2023.
3. [论文] Abootorabi et al. "A Survey on Multimodal RAG". arXiv, 2025.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
