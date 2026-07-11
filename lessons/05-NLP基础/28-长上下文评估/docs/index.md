# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 标称 10M token 上下文。在 1M token 处，8 针 MRCR 降到 26.3%。标称 ≠ 可用。长上下文评估告诉你实际可用的容量。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 05 · 13（问答系统）、05 · 23（RAG 分块策略） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分六个长上下文基准——NIAH、RULER、LongBench v2、MRCR、NoLiMa、BABILong——各自衡量什么
- [ ] 构建领域定制的 NIAH 扫描（深度 × 长度热力图）和多针变体
- [ ] 报告两个数字而非一个——"有效检索长度"和"有效推理长度"——后者通常是前者的 25-50%

---

## 1. 问题

200 页合同。模型标称 1M token 上下文。你问"终止条款是什么？"——模型从封面页回答。终止条款在 120k token 深处——超出了模型实际注意范围。

**2026 年的上下文容量鸿沟：** 规格表说 1M 或 10M。现实说 60-70% 可用——且"可用"依赖于任务：
- **检索（单针 NIAH）：** 前沿模型在标称最大值处近乎完美
- **多跳/聚合：** 大多数模型在 ~128k 后急剧退化
- **分散事实推理：** 第一个失败的任务

长上下文评估衡量这些维度。本课命名基准、每个基准实际测什么、以及如何为你的领域构建定制针测试。

---

## 2. 概念——六个基准，六个维度

### 2.1 NIAH（大海捞针，2023）

在长上下文的受控深度处置入一个事实（"神奇词是 pineapple"）。让模型检索它。扫描深度 × 长度。**原始长上下文基准。** 前沿模型现在在此饱和；它是必要但不充分的基线。

### 2.2 RULER（Nvidia, 2024）

13 种任务类型，跨 4 个类别：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频率）、QA。可配置上下文长度（4k 到 128k+）。**暴露在 NIAH 上饱和但在多跳上失败的模型。** 在 2024 发布中，声称 32k+ 上下文的 17 个模型中只有一半在 32k 处维持质量。

### 2.3 LongBench v2（2024）

503 道多选题，8k-2M 词上下文，六个任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码仓库、长结构化数据。**真实世界长上下文行为的生产基准。**

### 2.4 MRCR（多轮指代消解）

规模化多轮指代消解。8 针、24 针、100 针变体。**暴露一个模型在注意力退化前能同时处理多少事实。**

### 2.5 NoLiMa

"非词汇针"。针和查询之间没有任何字面重叠——检索需要一步语义推理。比 NIAH 更难——NIAH 中针和查询共享关键词。

### 2.6 BABILong

将 bAbI 推理链嵌入无关的干草堆中。测试**推理在大海捞针中是否成立**——不仅仅是检索。

### 2.7 实际应该报告什么

两个数字——而非一个：

- **标称上下文窗口。** 规格表上的数字
- **有效检索长度。** NIAH 在某个阈值（如 90%）通过的最大长度
- **有效推理长度。** 多跳或聚合在同一阈值通过的最大长度
- **退化曲线。** 准确率 vs 上下文长度，按任务类型分别绘制

**两个数字写进你的规格表：** 有效检索长度和有效推理长度。通常有效推理长度是标称窗口的 25-50%。

---

## 3. 从零实现

### 第 1 步：为你的领域定制 NIAH

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    """在干草堆中 depth_ratio 处插入针。"""
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    body_len = max(total_tokens - len(needle_tokens), 0)

    # 重复 filler 直到足够长
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)

def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

**扫描 depth_ratio ∈ {0, 0.25, 0.5, 0.75, 1.0} × total_tokens ∈ {1k, 4k, 16k, 64k}。** 绘制热力图。这就是你目标模型的 NIAH 卡片。

### 第 2 步：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]  # 三根针分散在文档中
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        chunks.append(filler[int(total_tokens * depth):int(total_tokens * (depth + 0.3))])
    return " ".join(chunks)
```

问题如"What are the three magic words?"需要检索全部三根针。**单针成功不能预测多针成功。** 这是 MRCR 精神的简化版——测试注意力在多事实间的分配。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = "X1 = 42. ...（大量填充文本）... X2 = X1 + 10. ...（填充）... X3 = X2 * 2."
question = "What is X3?"
# 答案需要链接三次赋值。前沿模型在 128k 处常降到 50-70%。
```

### 第 4 步：LongBench v2 在你的技术栈上

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = sum(1 for x in tasks
                  if normalize(model.complete(x["context"] + "\nQ:" + x["question"]))
                  == normalize(x["answer"]))
    return correct / len(tasks)
```

**按类别分别报告准确率。** 聚合分数隐藏了任务级别上的巨大差异。

---

## 4. 陷阱

- **仅评估 NIAH。** 在 1M token 上通过 NIAH 对多跳毫无意义。始终运行 RULER 或定制多跳测试
- **均匀深度采样。** 许多实现只在 depth=0.5 测试。在 depth=0, 0.25, 0.5, 0.75, 1.0 测试——"中间丢失"效应是真实存在的
- **针与填充文本的词汇重叠。** 如果针与填充文本共享关键词——检索变得微不足道。使用 NoLiMa 风格的非重叠针
- **忽略延迟。** 1M token prompt 需要 30-120 秒来预填充。在衡量准确率的同时衡量首 token 时间
- **厂商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重跑

---

## 5. 工业工具——2026 技术栈

| 场景 | 基准 |
|---|---|
| 快速健全检查 | 定制 NIAH——3 深度 × 3 长度 |
| 生产模型选择 | RULER（13 任务）——你的目标长度 |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或定制变量追踪 |
| 对话 | MRCR 8 针——你的目标长度 |
| 模型升级回归 | 固定的 in-house NIAH + RULER 框架——每次新模型重跑 |

**生产经验法则：** 在没有 NIAH + 一个推理任务在你目标长度上验证通过之前——永远不要信任一个上下文窗口。

### 中文长上下文特别建议

- **中文 token ≈ 1.5× 英文 token。** 标称 128k 约等于 85k 个中文字。NIAH 测试中的针位置计算需用 token 数而非字符数
- **中文模型独立验证。** Qwen2.5/3、DeepSeek-V3、Hunyang 在中文长上下文上的表现各异——通用英文基准（RULER/LongBench）的结论不可自动迁移。使用中文填充文本 + 中文针 + 中文查询在中文本地数据上独立测试
- **中文"中间丢失"效应可能比英文更严重。** 中文的每个 token 携带更多信息（一个汉字 ≈ 1.5 个英文 token），上下文中部的信息密度更高——注意力的"中间塌陷"可能更早出现

---

## 6. 面试考点

### Q1：为什么 NIAH 完美但生产长上下文性能很差？（难度：⭐⭐）

**参考答案：**
因为 NIAH 只测检索——一个事实、一个查询、直接字面匹配。"神奇词是 pineapple"→"神奇词是什么？"→模型只需要在最简单的"找到这个字符串"的意义上 attend。生产场景要求的不只是检索——多跳推理（X1→X2→X3）、分散事实的聚合、或跨段落的信息比较。NIAH 是最低门槛——通过它意味着"模型至少能注意到远处的内容"。不通过是一个巨大的红旗。但通过它——并不意味着模型能对远处的内容进行推理。

### Q2："有效检索长度"和"有效推理长度"分别怎么定义和衡量？（难度：⭐⭐⭐）

**参考答案：**
有效检索长度 = NIAH 通过率首次降到 90% 以下的长度。有效推理长度 = RULER 多跳或聚合任务准确率首次降到 70% 以下的长度。两者的差距告诉你"模型能在多大程度上注意到远处的信息 vs 能在多大程度上对远处的信息进行推理"。有效推理长度通常是标称窗口的 25-50%——如果一个模型标称 128k，它的有效推理极限可能在 32-64k——你的分块策略和上下文预算应该基于推理长度而非检索长度来规划。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| NIAH | "大海捞针" | 在填充文本中置入一个事实，让模型检索它 |
| RULER | "NIAH 的升级版" | 13 种任务类型——检索/多跳/聚合/QA |
| 有效上下文 | "真实容量" | 准确率仍高于阈值的长度 |
| 中间丢失 | "深度偏差" | 模型对长输入中间部分的内容关注不足 |
| 多针 | "一次多个事实" | 多个置入点；测试注意力分配——而非单独检索 |
| MRCR | "多轮指代" | 8/24/100 针指代消解；暴露注意力饱和 |
| NoLiMa | "非词汇针" | 针和查询不共享字面 token；需要推理而非检索 |

---

## 📚 小结

六个基准测六个维度——NIAH（检索门槛）< RULER（多跳）< LongBench v2（真实世界）< MRCR（注意力容量）< NoLiMa（语义推理）< BABILong（推理在大海捞针中）。**报告两个数字：有效检索长度（NIAH @ 90%）和有效推理长度（RULER @ 70%）。后者通常是前者的 25-50%。**

生产铁律：永远不在没有 NIAH + 一个推理任务在你目标长度上验证通过之前信任一个上下文窗口。中文 token ≈ 1.5× 英文——所有基准结论需要中文本地针测试独立验证。"中间丢失"效应在中文上可能更早出现。

---

## ✏️ 练习

1. 【理解】构建 NIAH——3 深度（0.25, 0.5, 0.75）× 3 长度（1k, 4k, 16k）。在任何模型上运行。将通过率绘制为 3×3 热力图。

2. 【实现】加入 3 针变体。在每个长度上衡量全部 3 针的检索率。与同长度的单针通过率对比。

3. 【实验】构造一个变量追踪任务（X1→X2→X3，3 跳）嵌入 64k 填充文本。在 3 个前沿模型上衡量准确率。报告每个模型的有效推理长度。

4. 【思考】你的模型在 NIAH depth=0.5（正中间）处准确率骤降——但 depth=0 和 depth=1.0 都近乎完美。"中间丢失"在你的 RAG 分块策略中意味着什么？如何仅靠分块来缓解？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-long-context-eval.md` | 按模型和用例设计长上下文评估的系统化方案 |

---

## 📖 参考资料

1. [论文] Kamradt. "Needle in a Haystack — Pressure Testing LLMs". 2023. https://github.com/gkamradt/LLMTest_NeedleInAHaystack — 原始 NIAH 仓库
2. [论文] Hsieh et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?". 2024. https://arxiv.org/abs/2404.06654 — 多任务基准
3. [论文] Bai et al. "LongBench v2: Towards Deeper Understanding and Reasoning on Long Context". 2024. https://arxiv.org/abs/2412.15204 — 真实世界长上下文评估
4. [论文] Liu et al. "Lost in the Middle: How Language Models Use Long Contexts". TACL, 2024. https://arxiv.org/abs/2307.03172 — 深度偏差论文
5. [论文] Modarressi et al. "NoLiMa: Long-Context Evaluation Beyond Literal Matching". 2024. https://arxiv.org/abs/2404.06666 — 非词汇针
6. [论文] Kuratov et al. "BABILong: Testing the Limits of LLMs with Long Context Reasoning". 2024. https://arxiv.org/abs/2406.10149 — 大海捞针中的推理

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文长上下文建议（token 换算、"中间丢失"中文适配）、工程最佳实践、常见错误、面试考点等均为原创内容。
