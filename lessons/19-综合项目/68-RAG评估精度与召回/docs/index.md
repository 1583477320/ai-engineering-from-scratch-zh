# 综合项目68——RAG 评估精度与召回（RAG Evaluation: Precision, Recall, MRR, nDCG）

> 如果你无法同时评估检索和答案质量，就无法发布系统。两者不是同一个指标。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第64-67节
**预计时间：** 90分钟

---

## 学习目标

- 从金标准 qrels 计算 precision@k、recall@k、MRR、nDCG@k
- 计算忠实度和答案相关性
- 读取指标值诊断管道故障阶段
- 使用确定性 mock 评审器离线评估

---

## 1. 问题

RAG 系统至少有四个活动部件：分块器、检索器、重排序器、生成器。任何一个都可能是错误答案的原因。没有逐阶段指标就是盲目飞行。

---

## 2. 核心概念

### 2.1 四个检索指标

| 指标 | 衡量 | 适用 |
|:----|:-----|:-----|
| precision@k | Top-k 中金标准的比例 | 减少无关块 |
| recall@k | 金标准在 Top-k 中的比例 | 不遗漏答案 |
| MRR | 第一个相关文档的位置 | 将最佳答案置顶 |
| nDCG@k | 分级相关性排序质量 | 多等级相关性 |

### 2.2 两个答案指标

- **忠实度**：答案中每个断言是否被检索到的上下文支持
- **答案相关性**：答案是否真正回答了问题

### 2.3 指标诊断表

| 症状 | 可能原因 | 修复 |
|:----|:--------|:-----|
| recall@k 低, precision@k 低 | 分块器切错或检索器找不到 | 分块策略或检索器 |
| recall@k 可以, MRR 低 | 正确块在 Top-k 但不在位置1 | 重排序器 |
| MRR 高, 低忠实度 | 生成器编造内容 | 生成提示词 |
| 全部高但用户抱怨 | 评估集不具代表性 | 扩展 qrels |

---

## 3. 从零实现

```python
"""RAG 评估——P@K + R@K + MRR + nDCG@K。"""
import math
from collections import defaultdict


def precision_at_k(retrieved, gold, k):
    top = retrieved[:k]
    return sum(1 for d in top if d in gold) / max(len(top), 1)

def recall_at_k(retrieved, gold, k):
    top = retrieved[:k]
    return sum(1 for d in top if d in gold) / max(len(gold), 1)

def mean_reciprocal_rank(results_list, gold_list):
    mrr = 0
    for retrieved, gold in zip(results_list, gold_list):
        for rank, d in enumerate(retrieved, 1):
            if d in gold:
                mrr += 1 / rank; break
    return mrr / max(len(results_list), 1)

def ndcg_at_k(retrieved, graded, k):
    dcg = sum(graded.get(d, 0) / math.log2(i + 2) for i, d in enumerate(retrieved[:k]))
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

def faithfulness(answer, context, threshold=0.3):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens: return 0.0
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return 1.0 if overlap >= threshold else 0.0

def answer_relevance(question, answer, threshold=0.3):
    q_tokens = set(question.lower().split())
    a_tokens = set(answer.lower().split())
    if not q_tokens: return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    return 1.0 if overlap >= threshold else 0.0


QRELS = [
    {"query": "abort threshold", "gold": ["d1", "d3"], "graded": {"d1": 3, "d3": 2}, "gold_answer": "three failed parts"},
    {"query": "retry policy", "gold": ["d3", "d4"], "graded": {"d3": 3, "d4": 1}, "gold_answer": "exponential backoff"},
    {"query": "upload chunking", "gold": ["d2"], "graded": {"d2": 3}, "gold_answer": "split uploads"},
]

def mock_retrieve(query, top_k=3):
    all_docs = ["d1", "d2", "d3", "d4"]
    if "abort" in query.lower(): return ["d1", "d3", "d2"][:top_k]
    if "retry" in query.lower(): return ["d3", "d4", "d1"][:top_k]
    return ["d2", "d1", "d3"][:top_k]

def main():
    results = [mock_retrieve(q["query"]) for q in QRELS]
    p5 = [precision_at_k(r, q["gold"], 5) for r, q in zip(results, QRELS)]
    r5 = [recall_at_k(r, q["gold"], 5) for r, q in zip(results, QRELS)]
    mrr = mean_reciprocal_rank(results, [q["gold"] for q in QRELS])
    ndcg5 = [ndcg_at_k(r, q["graded"], 5) for r, q in zip(results, QRELS)]

    print(f"P@5 = {sum(p5)/len(p5):.3f}")
    print(f"R@5 = {sum(r5)/len(r5):.3f}")
    print(f"MRR = {mrr:.3f}")
    print(f"nDCG@5 = {sum(ndcg5)/len(ndcg5):.3f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 指标 | 特点 |
|:----|:-----|:-----|
| Ragas | 全部 | 自动化 RAG 评估 |
| DeepEval | P/R/F1 | 企业级 |
| TREC Eval | P/R/nDCG | 检索标准 |
| `nltk` | BLEU | 标题生成 |

---

## 5. 工程最佳实践

- qrels 定期更新——答案随语料库变化而过时
- 按查询类型分片报告——平均 recall@k 可能掩盖特定类别失败
- **中文场景建议**：忠实度评估对中文需要先分词再做词元重叠计算

---

## 6. 常见错误

- **LLM-as-judge 自身偏差**：模型评判自己的输出更忠实——使用不同模型家族
- **忠实度微检查漏过宏观断言**：逐句检查通过但整体结构误导
- **Recall@k 掩盖查询级失败**：90% 平均 recall 可能隐藏某类查询总是失败

---

## 7. 面试考点

**Q1：precision@k 和 recall@k 何时偏好不同？**（难度：⭐⭐）

**参考答案：** 当无关块成本高（生成器浪费 token）时偏好 precision@k。当遗漏答案成本高（宁愿多见一个无关块也不愿漏掉答案）时偏好 recall@k。生产 RAG 通常报告 recall@k。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| precision@k | Top-k 中金标准的比例 |
| recall@k | 金标准在 Top-k 中的比例 |
| MRR | 1/第一个相关文档的排名 |
| nDCG@k | 分级相关性归一化排序增益 |
| 忠实度 | 答案断言被检索上下文支持的比例 |

---

## 📚 小结

RAG 评估覆盖检索、排序和生成三个阶段。你实现了六个指标，可以精确定位管道的故障环节。下一节将所有组件组合为端到端 RAG 系统。

---

## ✏️ 练习

1. 【实现】添加 hit-rate@k 并与 recall@k 对比
2. 【实现】分级忠实度：0（不支持）→ 1（部分支持）→ 2（完全支持）

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| RAG 评估 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Buckley & Voorhees. "Evaluating Evaluation Measure Stability". SIGIR 2000.
2. [论文] Järvelin & Kekäläinen. "Cumulated Gain-based Evaluation of IR Techniques". 2002.
3. [GitHub] Ragas. https://docs.ragas.io
