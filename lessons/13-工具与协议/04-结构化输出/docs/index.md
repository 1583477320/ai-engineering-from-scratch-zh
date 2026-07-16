# 结构化输出——JSON Schema、Pydantic、约束解码

> "让模型返回 JSON" 有 5-15% 的失败率。约束解码从根源解决——模型被字面阻止生成违反 schema 的 token。OpenAI 的 strict 模式、Anthropic 的 schema 类型化工具使用、Gemini 的 responseSchema、Pydantic AI 的 output_type、Zod 的 `.parse` 是同一个思想的五种表面形式。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用深入）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 编写 JSON Schema 2020-12 子集——使用正确的约束（enum、min/max、required、pattern）
- [ ] 使用三种提供商的严格模式（strict mode）保证输出格式
- [ ] 实现 Pydantic 验证层——拒绝格式错误的 LLM 输出并重试
- [ ] 理解约束解码如何在词元级强制有效 JSON

---

## 1. 问题

"让模型返回 JSON" 在前沿模型上仍有 5-15% 的失败率。模型可能在 JSON 中间关闭括号、添加多余文本、使用非标准格式。

约束解码从根源解决：**模型被字面阻止生成违反 schema 的 token**——不是后处理检查，而是在生成过程中强制格式。

---

## 2. 概念

### 2.1 三种严格模式

| 提供商 | 方法 | 保证程度 |
|--------|------|---------|
| OpenAI | `response_format: {"type": "json_schema"}` | 100% |
| Anthropic | `tool_use` + `input_schema` | 接近 100% |
| Gemini | `responseSchema` | 接近 100% |

### 2.2 约束解码原理

```
标准解码: 词元 A → 词元 B → 词元 C → ...
约束解码: 词元 A → [只允许符合 schema 的词元] → 词元 B → ...

使用有限状态机 (FSM) 或上下文无关语法 (CFG) 在每步限制可选词元
```

### 2.3 JSON Schema 关键约束

| 约束 | 用途 | 示例 |
|------|------|------|
| `type` | 值类型 | `"string"`, `"number"`, `"boolean"` |
| `required` | 必填字段 | `["name", "age"]` |
| `enum` | 枚举值 | `["red", "green", "blue"]` |
| `minimum/maximum` | 数值范围 | `"minimum": 0, "maximum": 100` |
| `pattern` | 正则匹配 | `"^\\d{4}-\\d{2}-\\d{2}$"` |

---

## 3. 从零实现

### Step 1：JSON Schema 验证器

```python
def validate_json_schema(data, schema):
    """简化版 JSON Schema 验证。"""
    errors = []

    # 检查必填字段
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"缺少必填字段: {field}")

    # 检查类型
    for field, field_schema in schema.get("properties", {}).items():
        if field in data:
            expected = field_schema.get("type")
            actual = type(data[field]).__name__
            type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
            if expected and type_map.get(actual) != expected:
                errors.append(f"字段 {field}: 期望 {expected}，实际 {actual}")

    return errors


def strict_format(schema):
    """生成 OpenAI strict 格式的 response_format。"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "output_schema",
            "strict": True,
            "schema": schema,
        }
    }
```

### Step 2：重试包装器

```python
def safe_json_extract(schema, llm_fn, max_retries=2):
    """带验证和重试的 JSON 提取。"""
    for attempt in range(max_retries):
        response = llm_fn(schema=schema)
        errors = validate_json_schema(response, schema)
        if not errors:
            return response, None
    return None, "重试次数用尽"
```

---

## 4. 工具

### 4.1 OpenAI Structured Output

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "提取产品信息"}],
    response_format=ProductSchema,
)
```

### 4.2 Anthropic Tool Use（严格模式）

```python
response = client.messages.create(
    model="claude-sonnet-5",
    tools=[{"name": "extract", "input_schema": schema}],
    messages=[...],
)
```

### 4.3 Pydantic AI

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Pydantic 自动验证 LLM 输出
```

---

## 5. 工程最佳实践

### 5.1 Schema 设计原则

- 字段名用英文——减少模型猜测
- 类型精确——用 `integer` 而非 `number`
- 提供 `description`——帮助模型理解每个字段
- 必填字段明确列出——减少验证失败

### 5.2 踩坑经验

- **schema 太复杂**：>20 个字段时模型可能遗漏某些字段——简化或分步提取
- **严格模式开销**：约束解码比自由生成慢 10-30%——在延迟敏感场景权衡

---

## 7. 常见错误

### 错误 1：不使用严格模式

**现象：** 模型返回的 JSON 有语法错误——括号不匹配。

**修复：** 启用严格模式——OpenAI 的 `json_schema`、Anthropic 的 `tool_use`。

### 错误 2：schema 中缺少枚举约束

**现象：** 模型返回不在允许范围内的值。

**修复：** 为所有离散选项添加 `enum` 约束。

---

## 8. 面试考点

### Q1：约束解码和后处理验证有什么区别？（难度：⭐⭐）

**参考答案：**
后处理验证在模型生成完成后检查——如果格式错误需要重试（浪费算力）。约束解码在生成过程中用 FSM/CFG 限制可选 token——确保每步都符合 schema，100% 格式保证。约束解码更可靠但有 10-30% 的速度开销（因为每步需要计算允许的 token 集合）。

### Q2：JSON Schema 的 strict 模式如何保证 100% 有效？（难度：⭐⭐⭐）

**参考答案：**
strict 模式使用约束解码——在每个解码步骤中，根据当前 FSM 状态计算允许的 token 集合。例如，如果当前需要输出数字值，则只有数字 token 被允许；如果需要闭合括号，则只有 `}` 和 `,` 被允许。这确保了输出序列在词元级就符合 JSON 语法——不可能生成违反 schema 的 token。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 严格模式 | "强制 JSON" | 使用约束解码确保输出 100% 匹配 JSON Schema |
| 约束解码 | "限制生成" | 用 FSM/CFG 在每步限制可选 token——保证格式正确 |
| JSON Schema | "输出格式说明" | 用 JSON 定义输出的类型、约束、必填字段 |

---

## 📚 小结

约束解码从根源解决格式问题——不是后处理检查，而是在生成过程中强制格式。OpenAI/Anthropic/Gemini 都支持严格模式。JSON Schema 的关键约束：type、required、enum、minimum/maximum。生产系统应始终使用严格模式。

---

## ✏️ 练习

1. **【实现】** 用三种提供商格式分别提取结构化信息——对比格式差异
2. **【实验】** 对比有/无严格模式时的 JSON 有效性率

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 结构化提取器 | `code/main.py` | JSON Schema 验证 + 严格模式 + 重试 |

---

## 📖 参考资料

1. [文档] OpenAI Structured Output: https://platform.openai.com/docs/guides/structured-outputs
2. [文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. [文档] JSON Schema 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
