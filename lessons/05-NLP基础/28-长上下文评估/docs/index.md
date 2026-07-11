# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 标称 10M token 上下文。在 1M token 处，8 针 MRCR 降到 26.3%。标称 ≠ 可用。长上下文评估告诉你实际可用的容量。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 05 · 13、05 · 23 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

- [ ] 区分四个长上下文基准——NIAH、RULER、LongBench、MRCR——各自衡量什么
- [ ] 理解"标称长度"与"实际有效容量"的差异——后者通常是前者的 60-70%
- [ ] 构建领域特定的针测试——因为通用基准的结论不一定迁移到你的分布

---

## 1. 四个基准——各自测不同的维度

| 基准 | 测什么 | 2026 前沿模型结果 |
|---|---|---|
| **NIAH**（大海捞针） | 128k 文档某处放一个事实。能找到吗？ | 近乎完美直到标称最大值 |
| **RULER** | 多针 + 多跳推理。NIAH 升级——不只找到针，还要综合多针信息 | 远低于标称最大值。多跳在 128k+ 急剧退化 |
| **LongBench** | 多任务长文理解——QA、摘要、少样本 | 各模型差异极大——不存在"最好" |
| **MRCR**（多针复述） | 8 根针分散在文档中——能全部复述吗？ | Gemini 3 Pro 10M → 1M 处仅 26.3% |

**关键发现：分散事实的推理是第一个失败的任务。**

---

## 2. 构建领域针测试

```python
def needle_test(model, doc, needle, position_ratio):
    insert_pos = int(len(doc) * position_ratio)
    modified = doc[:insert_pos] + f"\n{needle}\n" + doc[insert_pos:]
    answer = model.query(modified, "针的内容是什么？")
    return needle.lower() in answer.lower()
```

在 0%、25%、50%、75%、100% 五个位置各测。**真实验 = 你的领域文档 + 你的任务。** 中文注意：中文字符 vs 英文 token 约 1:1.5——标称 128k ≈ 85k 中文字。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
