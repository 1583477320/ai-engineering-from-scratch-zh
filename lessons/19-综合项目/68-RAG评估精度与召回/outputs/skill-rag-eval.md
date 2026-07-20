# RAG 评估配方

## 六个指标

| 类型 | 指标 |
|:----|:-----|
| 检索 | precision@k, recall@k, MRR, nDCG@k |
| 生成 | 忠实度, 答案相关性 |

## 诊断表

- recall低+precision低 → 分块器
- recall可以+MRR低 → 重排序器
- MRR高+忠实度低 → 生成提示词
