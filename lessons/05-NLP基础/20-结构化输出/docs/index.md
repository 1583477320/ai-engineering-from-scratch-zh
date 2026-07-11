# 结构化输出与约束解码

> 向 LLM 要 JSON。大多数时候拿到的确实是 JSON。在生产环境中，"大多数"就是问题。约束解码在采样之前修改 logits，把"大多数"变成"永远"。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 17（聊天机器人）、阶段 05 · 19（子词分词）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 14（智能体工程）— 工具调用 = 结构化输出 + 函数签名

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解约束解码的原理——logit processor 每步将无效 token 的 logit 设为 -∞
- [ ] 使用 Outlines 从 Pydantic 模型生成 JSON——零验证错误，由构造保证
- [ ] 理解 Schema 字段顺序为什么重要——先写 reasoning 再写 answer，否则模型会在"想清楚之前就承诺答案"
- [ ] 区分三层方案——提示词工程 → 原生 API → 约束解码——各自的适用场景

---

## 1. 问题

分类器提示 LLM："返回 positive/negative/neutral 中的一个。"模型返回："这个评论的情感是 positive——因为顾客明确表达了……"你的解析器崩溃了。分类器的 F1 是 0.0。

**自由形式的生成不是合约——它只是建议。生产系统需要合约。**

2026 年有三层方案：

1. **提示词工程。** 好好请求。"只返回 JSON 对象。"前沿模型上 ~80% 成功，小模型更低
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。支持的 schema 上可靠。厂商锁定
3. **约束解码。** 每步生成前修改 logits——模型**不可能**输出无效 token。100% 有效——由构造保证。任何本地模型可用

---

## 2. 概念

### 2.1 约束解码的工作原理

```
正常生成: LLM → logits(100k 词表) → softmax → 采样 → token
约束解码: LLM → logits(100k 词表) → [logit processor: 无效token→-∞] → softmax → 采样 → token
```

**logit processor** 坐在模型和采样器之间。它在当前目标语法（JSON Schema / 正则 / 上下文无关文法）的位置上计算哪些 token 合法，将所有不合法 token 的 logit 设为 -∞。Softmax 只在合法 token 上分配概率——**模型不可能输出不合法的东西。**

### 2.2 2026 年实现

- **Outlines。** 将 JSON Schema 或正则编译为有限状态机（FSM）。每个 token 做 O(1) 的合法下一 token 查询。基于 FSM——递归 schema 需要展平
- **XGrammar / llguidance。** 上下文无关文法（CFG）引擎。处理递归 JSON Schema。几乎零解码开销。OpenAI 在 2025 年的结构化输出实现中归功于 llguidance
- **vLLM 引导解码。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`——通过 Outlines、XGrammar 或 lm-format-enforcer 后端
- **Instructor。** 基于 Pydantic 的跨 LLM wrapper。验证失败时重试。跨提供商——但不修改 logits——依赖重试 + 结构化输出感知的 prompts

### 2.3 反直觉的结果

**约束解码通常比无约束生成更快。** 两个原因。第一，收缩了下一 token 的搜索空间。第二，聪明实现对强制 token（脚手架如 `{"name": "`——每个字节都是确定的）完全跳过 token 生成。

### 2.4 让你付出代价的陷阱

**字段顺序很重要。** 把 `answer` 放在 `reasoning` 前面——模型在思考之前就承诺了答案。JSON 有效。答案是错的。没有任何验证能捕获这一点。

```json
// 错误：先答案后推理
{"answer": "yes", "reasoning": "because ..."}

// 正确：先推理后答案
{"reasoning": "... therefore ...", "answer": "yes"}
```

**Schema 字段顺序是逻辑，不是排版。**

---

## 3. 从零实现

### 第 1 步：正则约束生成——从零

```python
def mask_logits(logits, valid_token_ids):
    """将无效 token 的 logit 设为 -∞。"""
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask

def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)  # FSM 告诉我们哪些 token 合法
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 追踪"语法的哪些部分已经被满足"。`valid_tokens(state, tokenizer)` 计算哪些词表 token 可以推进 FSM 而不离开接受路径。

### 第 2 步：Outlines——JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str

model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: '服务员很周到，菜上来时还是热的。'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='周到 ... 热的')
```

**零验证错误。永远。** FSM 使无效输出不可到达。

### 第 3 步：Instructor——跨提供商的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]

client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7", max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "提取：'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不碰 logits——它把 schema 格式化成 prompt，解析输出，验证失败时重试（默认 3 次）。适用任何提供商。重试增加延迟和成本。跨提供商的移植性是卖点。

### 第 4 步：原生厂商 API

```python
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "分类：'菜是凉的。'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                      "enum": ["positive", "negative", "neutral"]}}}}},
)
```

服务端约束解码。在支持的 schema 上与 Outlines 可靠性对等。不需管理本地模型。锁定厂商。

---

## 4. 陷阱

- **递归 Schema。** Outlines 将递归展平为固定深度。树形结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（CFG 引擎）
- **巨大枚举。** 1 万选项的枚举编译极慢或超时。改为检索器：先预测 top-k 候选，在这些候选中约束
- **语法太严格。** 强制 `date: "YYYY-MM-DD"` 正则——模型无法输出缺失日期的 `"unknown"`。模型通过编造一个日期来补偿。允许 `null` 或一个哨兵值
- **过早承诺。** 见上面字段顺序陷阱。始终把 reasoning 放在第一位
- **厂商 JSON 模式无 Schema。** 纯 JSON 模式只保证语法合法的 JSON——不保证**你的场景**合法。始终提供完整 Schema

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| OpenAI/Anthropic/Google 模型、简单 Schema | 原生厂商结构化输出 |
| 任意提供商、Pydantic 工作流、可容忍重试 | Instructor |
| 本地模型、需要 100% 有效性、扁平 Schema | Outlines (FSM) |
| 本地模型、递归 Schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 可容忍重试的批处理 | Instructor + 最便宜模型 |

---

## 6. 知识连线

- **阶段 05 · 17（聊天机器人）→** LLM 智能体的工具调用 = 结构化输出 + 函数签名——本课的约束解码保证了工具调用的参数在语法上是合法的 JSON
- **阶段 05 · 29（对话状态跟踪）→** 槽位值对 = 结构化输出 + 槽位 Schema——约束解码确保了"价格"字段的值来自 `{cheap, moderate, expensive}` 枚举

---

## 7. 面试考点

### Q1：Schema 字段顺序为什么重要？（难度：⭐⭐）

**参考答案：**
LLM 是自回归的——从左到右逐 token 生成。如果 `answer` 字段在 `reasoning` 之前，模型在生成答案时还没有"看到"推理过程——它在思考之前就承诺了一个结论。一旦答案 token 被采样，它无法反悔——后续的推理被迫为已输出的答案辩护。Schema = 思考顺序。先推理，后答案。

### Q2：Instructor 和 Outlines 的本质差异是什么？（难度：⭐⭐⭐）

**参考答案：**
Outlines 修改 logits——在采样层面保证输出合法。Instructor 不碰 logits——它依赖 prompt 格式 + 输出解析 + 重试。Outlines = 由构造保证（100% 有效但不跨提供商）。Instructor = 由重试保证（可能失败但跨所有 API）。选择取决于你是否需要跨提供商的移植性——或是否能接受重试的延迟成本。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 约束解码 | "强制有效输出" | 每步生成时掩码无效 token 的 logit |
| Logit processor | "做约束的东西" | 函数：`(logits, state) → masked_logits` |
| FSM | "有限状态机" | 编译后的语法表示；O(1) 合法下一 token 查找 |
| CFG | "上下文无关文法" | 处理递归的语法；比 FSM 慢但更具表达力 |
| Schema 字段顺序 | "有影响吗？" | 有——第一个字段先提交；始终把 reasoning 放在 answer 前面 |
| JSON 模式 | "OpenAI 早期版本" | 保证 JSON 语法；**不**保证符合你的 Schema |

---

## 📚 小结 | ✏️ 练习

三层结构化输出——提示词工程（-80%）→ 原生 API（-95%）→ 约束解码（100%）——可靠性递增。Outlines FSM 提供零验证错误，XGrammar CFG 处理递归 Schema，Instructor 提供跨提供商移植。**永远把 reasoning 字段放在 answer 前面。——Schema 顺序 = 思考顺序。**

练习：在小开源模型上不加约束解码地 prompt `Review(sentiment, confidence, evidence_span)`。衡量 100 条评论的合法 JSON 比例。用 Outlines JSON 模式重复——对比合规率。

---

## 📖 参考资料

1. [论文] Willard, Louf. "Efficient Guided Generation for LLMs". 2023. https://arxiv.org/abs/2307.09702 — Outlines 论文
2. [论文] XGrammar. 2024. https://arxiv.org/abs/2411.15100 — 快速 CFG 约束解码
3. [官方文档] vLLM — Structured Outputs. https://docs.vllm.ai/en/latest/features/structured_outputs.html
4. [官方文档] Instructor. https://python.useinstructor.com/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
