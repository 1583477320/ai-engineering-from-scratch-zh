# 综合项目64——分块策略对比（Chunking Strategies, Compared）

> 分块决定了检索器能返回什么。边界切错了，任何嵌入模型、重排序器、LLM 都无法修复下游的损坏。

**类型：** 构建
**语言：** Python
**前置知识：** 第11章第06节（RAG基础）；第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 实现五种分块策略：固定窗口、句子、递归分割、语义聚类、结构化 Markdown 标题
- 在带金标准答案的固定语料上测量 recall@k
- 识别每种策略注入的失败模式
- 无需跑基准测试即可为新语料选择默认策略

---

## 1. 问题

每个 RAG 流水线都从切割源文档开始。切得足够小以让嵌入模型适配，切得足够大以让每个片段包含独立想法。切的位置不是超参数——它是检索器能返回的上限。

2024 年的论文 "LongRAG" 测量到仅改变分块策略就有 35% 的绝对检索召回率波动。分块选择不是锦上添花，而是基础架构决策。

---

## 2. 核心概念

### 2.1 五种策略

| 策略 | 原理 | 适用 | 失败模式 |
|:----|:-----|:-----|:---------|
| 固定窗口 | 每 N 字符切 | 控制基线 | 切断句子、符号 |
| 句子 | 按句切分后打包 | 散文 | 切断段落 |
| 递归分割 | 按层次分隔符递归 | 混合文档 | 过度碎片化 |
| 语义聚类 | 嵌入+聚类中心漂移 | 话题切换多 | 慢，依赖嵌入模型 |
| Markdown 标题 | 按 H1/H2 切分 | 有结构文档 | 仅限结构化文档 |

### 2.2 recall@k 如何衡量边界

每个查询有金标准答案在源文档中的字符偏移。分块后，检查检索器返回的 top-k 块中是否有块与金标准偏移重叠。重叠则 recall@k=1，否则为 0。

### 2.3 选择策略的三属性

| 属性 | 默认策略 |
|:----|:---------|
| 无结构散文 | 递归分割，目标 800 字符 |
| Markdown/API 文档 | 结构化 Markdown 标题 |
| 短段落混合主题 | 语义聚类，阈值 0.6 |

---

## 3. 从零实现

```python
"""分块策略对比——五种策略+recall@k 评估。"""
import re, hashlib
from dataclasses import dataclass


def fixed_window(text, size=500, overlap=50):
    chunks = []; start = 0
    while start < len(text):
        chunks.append(text[start:start+size]); start += size - overlap
    return chunks


def sentence_chunks(text, target=500):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) > target and current:
            chunks.append(" ".join(current)); current, cur_len = [], 0
        current.append(s); cur_len += len(s)
    if current: chunks.append(" ".join(current))
    return chunks


def recursive_split(text, separators=None, target=500):
    if separators is None: separators = ["\n\n", "\n", " ", ""]
    if not text: return []
    if len(text) <= target: return [text]
    if not separators: return [text[i:i+target] for i in range(0, len(text), target)]
    sep = separators[0]; rest = separators[1:]
    if sep == "": return [text[i:i+target] for i in range(0, len(text), target)]
    parts = text.split(sep)
    chunks, current, cur_len = [], [], 0
    for p in parts:
        if cur_len + len(p) + len(sep) > target and current:
            chunks.append(sep.join(current)); current, cur_len = [], 0
        current.append(p); cur_len += len(p) + len(sep)
    if current: chunks.append(sep.join(current))
    if len(chunks) == 1 and len(chunks[0]) > target:
        return recursive_split(chunks[0], rest, target)
    return chunks


def mock_embed(text, dim=32):
    h = hashlib.md5(text.encode()).digest()
    vec = [0.0] * dim
    for i, b in enumerate(h):
        if i >= dim: break
        vec[i] = b / 255.0 - 0.5
    norm = sum(v**2 for v in vec) ** 0.5 or 1
    return [v/norm for v in vec]


def semantic_chunks(text, threshold=0.6, dim=32):
    sents = re.split(r'(?<=[.!?])\s+', text)
    if not sents: return []
    chunks, current = [], [sents[0]]
    centroid = mock_embed(sents[0], dim)
    for s in sents[1:]:
        vec = mock_embed(s, dim)
        cos = sum(a*b for a,b in zip(centroid, vec))
        if cos < threshold and current:
            chunks.append(" ".join(current)); current = []
        current.append(s)
        n = len(current)
        centroid = [c*(n-1)/n + v/n for c,v in zip(centroid, vec)]
    if current: chunks.append(" ".join(current))
    return chunks


def structural_markdown(text):
    chunks = []; current_title = ""; current_body = []
    for line in text.split("\n"):
        if line.startswith("#"):
            if current_title or current_body:
                chunks.append(current_title + "\n" + " ".join(current_body))
            current_title = line; current_body = []
        else:
            if line.strip(): current_body.append(line.strip())
    if current_title or current_body:
        chunks.append(current_title + "\n" + " ".join(current_body))
    return chunks if chunks else [text]


def eval_recall(strategy_fn, doc, query_span, k=5):
    """简单 recall：策略分块后，检查 top-k 块中是否有包含查询偏移的。"""
    chunks = strategy_fn(doc)
    start, end = query_span
    for i, chunk in enumerate(chunks[:k]):
        ci = doc.find(chunk)
        if ci != -1 and ci <= start < ci + len(chunk):
            return 1.0
    return 0.0


def main():
    doc = ("注意力稀疏性是提高 Transformer 效率的关键方向。"
           "Top-k 路由只保留最显著的注意力头。"
           "实验表明稀疏注意力在困惑度上匹配密集注意力。"
           "部署成本降低约 30%。")
    strategies = [("固定窗口", lambda t: fixed_window(t, 30)),
                  ("句子", lambda t: sentence_chunks(t, 30)),
                  ("递归", lambda t: recursive_split(t, ["\n", "。", " ", ""], 30)),
                  ("语义", lambda t: semantic_chunks(t, 0.3, 16))]

    print("分块数量:")
    for name, fn in strategies:
        print(f"  {name}: {len(fn(doc))} 块")

    print("\nrecall@5 对比:")
    for name, fn in strategies:
        r = eval_recall(fn, doc, (50, 70), 5)
        print(f"  {name}: {r:.1f}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 策略 | 特点 |
|:----|:-----|:-----|
| LangChain RecursiveTextSplitter | 递归分割 | 最通用 |
| LlamaIndex SentenceSplitter | 句子 | 快速 |
| Unstructured.io | 多种 | 文档解析 + 分块 |
| SemanticChunker (LlamaIndex) | 语义 | 基于嵌入 |

---

## 5. 工程最佳实践

- 默认选择：递归分割，目标 800 字符——最强单策略基线
- **中文场景建议**：中文句子用 `。！？` 分隔而非空格；递归分割的分隔符顺序需要调整

---

## 6. 常见错误

- **固定窗口作为默认**：它是最差的默认——会切断代码、符号、句子
- **语义聚类未设字符上限**：5000 字符的聚类嵌入太稀疏
- **仅用一种策略**：不同文档类型需要不同策略

---

## 7. 面试考点

**Q1：递归分割为什么通常优于固定窗口？**（难度：⭐⭐）

**参考答案：** 递归分割优先在层次分隔符处切分（段落→句子→字符），保留了文档结构。固定窗口无视内容边界，会切断代码标识符、句子中间、表格单元。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| Recall@K | top-k 块中包含金标准答案的比例 |
| 分块重叠 | 滑动窗口在相邻块间重叠 N 字符 |
| 语义漂移 | 聚类中心相似度下降表示话题切换 |

---

## 📚 小结

分块策略是 RAG 的天花板。你实现了五种策略并对比了 recall@k。下一节构建混合检索器。

---

## ✏️ 练习

1. 【实验】添加代码块到散文语料，重新跑评估表格
2. 【实现】添加 `summary` 字段：每块附加一句摘要描述

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 分块策略 | `code/main.py` |

---

## 📖 参考资料

1. [论文] LongRAG. arXiv 2406.15319.
2. [官方文档] LlamaIndex 分块策略. https://docs.llamaindex.ai/
