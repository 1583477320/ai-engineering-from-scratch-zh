# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 标称 10M token 上下文。在 1M token 处，8 针 MRCR 降到 26.3%。标称 ≠ 可用。长上下文评估告诉你实际可用的容量。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 05 · 13、05 · 23 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

- [ ] 区分四个长上下文基准——NIAH、RULER、LongBench、MRCR
- [ ] 理解"标称"vs"实际"的差距（60-70%）——且取决于任务类型
- [ ] 构建领域特定针测试——通用基准结论不自动迁移到你的分布

---

## 1. 问题

200 页合同。模型标称 1M token 上下文。你贴进去问"终止条款是什么？"——模型从封面页回答。因为终止条款在 120k token 深处——超出了模型实际注意范围。

**2026 上下文容量鸿沟：** 规格表说 1M/10M。现实说 60-70% 可用——且"可用"依赖于任务：
- **检索（单针 NIAH）：** 前沿模型在标称最大值处近乎完美
- **多跳/聚合：** 大多数模型在 ~128k 后急剧退化
- **分散事实推理：** 第一个失败的任务

---

## 2. 四个基准

| 基准 | 测什么 | 2026 关键结果 |
|---|---|---|
| **NIAH**（大海捞针） | 单事实检索。128k 文档某处一针 | 近乎完美直到标称最大值——但只是最低门槛 |
| **RULER** | 多针+多跳。NIAH 升级——不只找还要综合 | 远低于标称。多跳在 128k+ 急剧退化 |
| **LongBench** | 多任务长文理解——QA/摘要/少样本 | 各模型差异极大——不存在"最好" |
| **MRCR**（多针复述） | 8 针分散——能全部复述吗？ | Gemini 3 Pro 10M→1M 处仅 26.3% |

**MRCR 揭示的问题比 NIAH 更接近真实使用场景。** 用户在一个文档中问不止一个问题——分散的多针检索是生产常态。

---

## 3. 从零实现——领域针测试

```python
def needle_test(model, doc, needle, position_ratio):
    """在文档 position_ratio 处插入针→问模型针的内容。"""
    insert_pos = int(len(doc) * position_ratio)
    modified = doc[:insert_pos] + f"\n{needle}\n" + doc[insert_pos:]
    answer = model.query(modified, "针的内容是什么？")
    return needle.lower() in answer.lower()

# 在 0/25/50/75/100% 五个位置各测
for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"位置 {ratio:.0%}: {needle_test(model, doc, needle, ratio)}")

# 多针 MRCR 测试
def mrcr_test(model, doc, needles_and_positions):
    modified = doc
    for pos, needle in needles_and_positions:
        insert_pos = int(len(doc) * pos)
        modified = modified[:insert_pos] + f"\n{needle}\n" + modified[insert_pos:]
    answer = model.query(modified, "列出文档中所有插入的事实。")
    return sum(1 for n in needles_and_positions if n[1].lower() in answer.lower()) / len(needles_and_positions)
```

**真实验 = 你的领域文档 + 你的任务。** 通用基准的结论在中文上需独立验证——中文 token vs 英文约 1:1.5，标称 128k ≈ 85k 中文字。

---

## 4. 陷阱

- **NIAH 通过 ≠ 长上下文"能用"。** NIAH 是最低门槛——检索。推理/聚合/比较是真正区分模型的任务
- **MRCR 在 1M 处的 26.3% 是 2026 生产环境的真实天花板。** 不要假设你的 RAG 管道在 128k+ 处会有高于这个数字的准确率
- **针测试对针的内容高度敏感。** 数字 vs 日期 vs 名字 → 不同针类型的结果不同。在你的领域数据类型上测试

---

## 5. 模型选择建议

| 上下文需求 | 模型建议 |
|---|---|
| < 8K token | 标准 chunk + 嵌入检索——不需要长上下文模型 |
| 8K-128K | 验证模型在你任务上的 RULER 多跳分数 |
| > 128K | 先跑 MRCR 多针测试——不要假设标称=可用 |

**中文模型：** Qwen2.5/3、DeepSeek-V3、Hunyuan 在中文长上下文上表现各异——在中文本地针测试上独立验证。

---

## 🔑 关键术语 | 📚 小结

四个基准测不同维度：NIAH（检索门槛）、RULER（多跳推理）、LongBench（多任务）、MRCR（最接近生产）。标称 ≠ 可用——实际可用 = 标称的 60-70%，且依赖于任务类型。始终在领域文档上构建针测试（包括多针 MRCR）。中文 token ≈ 1.5× 英文 token——跨语言基准结论不可自动迁移。

---

## ✏️ 练习

1. 【理解】在你使用的模型上跑 NIAH——128k 上下文 5 个位置各一根针。
2. 【实现】构建 8 针 MRCR 测试——衡量多针复述准确率 vs 单针。
3. 【思考】NIAH 完美但真实查询差——NIAH 漏掉了什么？

---

## 📖 参考资料

1. [论文] Hsieh et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?". 2024. https://arxiv.org/abs/2404.06654
2. [论文] Kamradt. "Needle In A Haystack — Pressure Testing LLMs". 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
