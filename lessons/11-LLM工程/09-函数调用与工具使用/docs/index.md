# 函数调用与工具使用

> LLM 不会计算 123 × 456。它不会查最新天气。它不会查数据库。但如果你告诉它有哪些工具可用，它会决定什么时候用哪个工具，生成正确的参数，你执行后把结果喂回去——它就能完成任何需要外部工具的任务。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-03（提示词工程）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解函数调用的机制——LLM 如何选择工具和生成参数
- [ ] 使用 OpenAI 和 Anthropic API 实现工具调用
- [ ] 设计健壮的工具定义——参数 schema 和错误处理
- [ ] 构建多轮工具调用循环——工具结果反馈给 LLM 继续推理

---

## 1. 问题

你让 ChatGPT："帮我查一下北京今天的天气。" 它不会自己上网。但如果它调用了 `get_weather(city="北京")` 函数，你执行这个函数，把结果返回给它——它就能完成这个任务。

**函数调用 = LLM 决定调用什么工具 + 生成正确参数 + 等待结果 + 基于结果继续推理。**

这是 LLM 从"文本生成器"变成"智能代理"的关键桥梁。

---

## 2. 概念

### 2.1 函数调用机制

```
1. 开发者：定义可用工具列表（函数 schema）
2. LLM：根据用户请求决定是否调用工具
3. LLM：生成调用参数（JSON 格式）
4. 开发者：执行函数，返回结果
5. LLM：基于结果生成最终回答
```

### 2.2 工具定义（JSON Schema）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    }
}]
```

### 2.3 多工具选择

LLM 根据用户意图从多个工具中选择最合适的。提示词中定义工具描述帮助 LLM 做出正确选择。

---

## 3. 从零实现

### Step 1：OpenAI 函数调用

```python
from openai import OpenAI

client = OpenAI()
tools = [{"type": "function", "function": {"name": "get_weather", "description": "获取天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

# LLM 决定调用工具
response = client.chat.completions.create(
    model="gpt-4o", tools=tools,
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}]
)
tool_call = response.choices[0].message.tool_calls[0]
# 执行工具
result = {"temperature": 22, "condition": "晴", "humidity": 45}
# 把结果返回给 LLM
response2 = client.chat.completions.create(
    model="gpt-4o", tools=tools,
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
    ]
)
```

### Step 2：Anthropic 工具使用

```python
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-5", max_tokens=1024,
    tools=[{"name": "get_weather", "description": "获取天气",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}]
)
# 如果 response.content[0].type == "tool_use" → 执行工具 → 把结果喂回
```

---

## 4. 工具

### 4.1 LangChain Tool

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}：晴天，22°C"
```

### 4.2 工具对比

| 方法 | 说明 |
|------|------|
| OpenAI Functions | 原生支持，JSON Schema 定义 |
| Anthropic Tool Use | 原生支持，input_schema 定义 |
| LangChain Tools | 多框架兼容，抽象层 |

---

## 6. 工程最佳实践

### 6.1 工具描述原则

- **名称清晰**：`get_weather` > `weather`
- **描述准确**：说明输入/输出和限制
- **参数 schema**：提供类型、描述、是否必填
- **错误处理**：工具返回错误信息而非崩溃

### 6.2 中文场景

- 工具描述用中英文双语提高 LLM 理解
- 中文查询 → 英文工具名 → 执行 → 中文结果

### 6.3 踩坑经验

- **工具太多**：>20 个工具时 LLM 选择困难——分组或动态加载
- **参数缺失**：LLM 有时省略必填参数——用重试循环
- **工具执行超时**：设置超时并返回错误信息

---

## 7. 面试考点

### Q1：函数调用和 RAG 有什么本质区别？（难度：⭐⭐）

**参考答案：**
RAG 是检索已有知识——从文档库中找到相关片段，让 LLM 基于这些片段生成回答。函数调用是执行外部操作——让 LLM 决定调用哪个工具，生成参数，等待执行结果，再基于结果继续推理。RAG 扩展了 LLM 的知识，函数调用扩展了 LLM 的能力（计算、查询、操作）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 函数调用 (Function Calling) | "让 LLM 用工具" | LLM 识别用户意图，选择工具，生成参数，等待结果，继续推理 |
| 工具 (Tools) | "LLM 的外部能力" | 函数定义 + JSON Schema，让 LLM 知道可以调用什么 |
| 参数 schema | "工具的说明书" | JSON 格式定义函数签名——参数名、类型、描述、必填性 |

---

## 📚 小结

函数调用让 LLM 从文本生成器变成智能代理。机制：定义工具 → LLM 选择工具 → 生成参数 → 开发者执行 → LLM 基于结果继续推理。OpenAI 和 Anthropic 都原生支持。工具描述要清晰，参数 schema 要完整，错误处理要健壮。

---

## ✏️ 练习

1. **【实现】** 用 OpenAI 函数调用实现一个"计算器"——LLM 决定何时调用 `add(a, b)` 函数
2. **【实验】** 测试 5 个不同查询——LLM 正确选择工具并生成参数的成功率

---

## 📖 参考资料

1. [文档] OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
2. [文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. [文档] LangChain Tools: https://python.langchain.com/docs/concepts/tools/
