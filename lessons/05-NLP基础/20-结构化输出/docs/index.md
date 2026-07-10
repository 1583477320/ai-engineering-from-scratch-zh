# 结构化输出与约束解码

> 向 LLM 要 JSON。大多数时候拿到的确实是 JSON。在生产环境中，"大多数"就是问题。约束解码在采样之前修改 logits，把"大多数"变成"永远"。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 17、05 · 19 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 理解约束解码的原理——在每一步生成时，logit processor 将无效 token 的 logit 设为 -∞
- [ ] 区分三层结构化输出方案——提示词工程 → 原生 API → 约束解码

---

## 1. 问题

分类器提示 LLM："返回 positive/negative/neutral 中的一个。"模型返回："这个评论的情感是 positive——因为顾客明确表达了……"你的解析器崩溃了。分类器的 F1 是 0.0。

自由形式的生成不是合约——它只是建议。**生产系统需要合约。**

2026 年有三层方案：

| 层 | 方法 | 可靠性 |
|---|---|---|
| 1 | 提示词工程——"只返回 JSON 对象" | 前沿模型 ~80%，小模型更低 |
| 2 | 原生结构化输出 API——OpenAI `response_format`、Anthropic tool use | 支持的 schema 上可靠。厂商锁定 |
| 3 | 约束解码——每步生成前修改 logits，模型**不可能**输出无效 token | 100% 有效——由构造保证。任何本地模型可用 |

---

## 2. 约束解码的原理

```
正常生成: LLM → logits(100k 词表) → softmax → 采样 → token
约束解码: LLM → logits(100k 词表) → [logit processor: 无效 token → -∞] → softmax → 采样 → token
```

**logit processor** 坐在模型和采样器之间。它在当前目标语法的位置（JSON Schema / 正则 / 上下文无关文法）上计算哪些 token 合法，将所有不合法 token 的 logit 设成负无穷。Softmax 只在合法 token 上分配概率——模型**不可能**输出不合法的东西。

### 2026 工具

```python
# lm-format-enforcer — JSON Schema → 约束解码
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_logits_processor

# outlines — 正则表达式 → 约束解码
import outlines
generator = outlines.generate.json(model, schema)

# guidance / lmql — 模板语言 + 约束解码
# instructor — 补丁 OpenAI 客户端，将 Pydantic 模型转为 tool calls
```

---

## 3. 三层方案的选择

| 场景 | 选择 |
|---|---|
| 原型、非生产代码 | 提示词工程——最快迭代 |
| OpenAI/Anthropic API、标准 schema | 原生结构化输出 API——最可靠 |
| 本地模型、复杂 schema、需要 100% 保证 | 约束解码——由构造保证 |
| 中文 JSON 键值对 | 约束解码 + JSON Schema 中指定中文键名 |

### 中文陷阱

中文 JSON 键名中的全角引号 vs 半角引号——`{"姓名"："张三"}` vs `{"姓名":"张三"}`。约束解码的 JSON Schema 需要显式处理全角/半角字符集。中文逗号（`，`vs`,`）同样——Schema 中明确只接受半角标点。

---

> 本课程参考了 AI Engineering From Scratch 的课程体系。
