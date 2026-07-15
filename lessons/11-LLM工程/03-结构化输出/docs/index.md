# 结构化输出：JSON、Schema 验证与约束解码

> 你的 LLM 返回字符串。你的应用需要 JSON。这个鸿沟比任何模型幻觉都导致了更多生产系统崩溃。结构化输出是自然语言和类型化数据之间的桥梁。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01（提示词工程）| **时间：** ~90 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式和 Schema 约束输出
- [ ] 构建 Pydantic 验证层——拒绝格式错误的 LLM 输出并带错误反馈重试
- [ ] 解释约束解码如何在词元层级强制有效 JSON——无需后处理
- [ ] 设计稳健的提取提示词——将非结构化文本可靠地转换为类型化数据结构

---

## 1. 问题

你让 LLM："从这个文本中提取产品名、价格和库存状态。" 它回答：

```
产品是 Sony WH-1000XM5 耳机，售价 $348.00，当前有库存。
```

完全正确的回答。对你的应用也完全无用。你的库存系统需要：
`{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。需要特定的键、特定的类型、特定的值约束。

最简单的"用 JSON 回答"——90% 的时候有效。另外 10% 模型把 JSON 包在 markdown 代码块里、加了前缀"这是 JSON"，或者语法错误少了一个括号。JSON 解析器崩溃。管道中断。你加 try/except，加重试。重试有时产生不同数据——现在又多了一个一致性问题。

这不是提示词工程问题。这是解码问题。模型从左到右生成词元，每一步从 10 万+ 的词表中选最有可能的下一个词元。大部分词元在当前位置会产生无效的 JSON。如果模型刚刚发出了 `{"price":`，下一个词元必须是数字、引号、null、true、false 或负号。没有约束，模型可能会选一个看起来合理的英文单词——语法上灾难性的错误。

---

## 2. 概念

### 2.1 三种结构化输出方法

| 方法 | 保证程度 | 实现难度 | 适用场景 |
|------|---------|---------|---------|
| **提示词约束** | 低（~90%） | 低 | 快速原型 |
| **JSON Mode / API 参数** | 中（~99%） | 低 | 生产环境 |
| **Schema 约束** | 高（接近 100%） | 中 | 关键业务 |
| **约束解码** | 最高（100%） | 高 | 最严格场景 |

### 2.2 JSON Mode vs Function Calling vs Structured Output

| 方法 | OpenAI | Anthropic | 说明 |
|------|--------|-----------|------|
| **JSON Mode** | `response_format={"type":"json_object"}` | 无 | 强制顶层是 JSON |
| **Function Calling** | `tools` 参数 | `tools` 参数 | 用函数定义做结构化 |
| **Structured Outputs** | `response_format` + JSON Schema | 无 | 2024 新版——强制整个输出匹配 schema |

---

## 3. 从零实现

### Step 1：带 Schema 的结构化输出

```python
import json
from pydantic import BaseModel, Field

class Product(BaseModel):
    """产品的 JSON Schema。"""
    product: str = Field(description="产品名称")
    price: float = Field(description="产品价格")
    in_stock: bool = Field(description="库存状态")

def extract_product(text, client):
    """结构化提取——Pydantic 对象直接返回。"""
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"提取产品信息:\n{text}"}],
        response_format=Product,
    )
    return response.choices[0].message.parsed
```

### Step 2：验证和重试

```python
from pydantic import ValidationError

def safe_extract(text, client, max_retries=2):
    """带验证和重试的结构化提取。"""
    for attempt in range(max_retries):
        try:
            result = extract_product(text, client)
            return result
        except ValidationError as e:
            if attempt < max_retries - 1:
                # 带错误反馈重试
                error_msg = f"提取无效 {e.errors()}，请确保格式匹配"
                continue
            return None
    return None
```

### Step 3：约束解码（伪代码）

```python
def constrained_decode(model, schema, prompt):
    """
    约束解码——只生成符合 schema 的 token。
    实际的约束解码使用 Outlines 或 XGrammar 在词元级做约束。
    """
    tokens = []
    while True:
        allowed_tokens = compute_allowed_tokens(schema, tokens)
        next_token = sample_from_allowed(model, tokens, allowed_tokens)
        if next_token is EOS:
            break
        tokens.append(next_token)
    return parse_json(tokens)
```

---

## 4. 工具

### 4.1 OpenAI Structured Outputs

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "创建一个日历事件：明天下午3点的团队会议"}],
    response_format=CalendarEvent,
)
print(completion.choices[0].message.parsed)
```

### 4.2 Anthropic Tool Use

```python
from anthropic import Anthropic

anthropic = Anthropic()
response = anthropic.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    tools=[{
        "name": "extract_product",
        "description": "提取产品信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["product", "price", "in_stock"],
        },
    }],
    messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
)
tool_use = response.content[0]
```

### 4.3 Instructor（Pydantic 自动验证）

```python
import instructor
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    messages=[{"role": "user", "content": "John Doe is 30 years old"}],
)
print(f"{user.name} is {user.age} years old")
```

### 4.4 工具对比

| 工具 | 方法 | 验证 | 适用场景 |
|------|------|------|---------|
| OpenAI Structured Outputs | API 原生 | 服务器端验证 | GPT-4o 系列 |
| Anthropic Tool Use | API 原生 | 客户端验证 | Claude 系列 |
| Instructor | Pydantic 包装 | 客户端验证 | 多模型兼容 |
| Outlines | 约束解码 | 词元级保证 | 严格格式要求 |
| LangChain | 提取链 | 客户端验证 | 现有 LangChain 项目 |

---

## 6. 工程最佳实践

### 6.1 Schema 设计原则

- **字段名明确**：`product_name` > `item`，减少模型猜测
- **类型精确**：明确用 `float` 而非 `number` 避免歧义
- **提供描述**：每个字段的 `description` 字段帮助模型理解预期值
- **可选字段**：非关键字段设为 Optional，减少验证失败概率

### 6.2 中文场景特别建议

- 提取中文文本时确保字段描述用中文或双语
- 注意中文标点符号和全角字符的处理

### 6.3 踩坑经验

- **验证失败不要直接重试**：将错误信息放入提示词让模型自己修复
- **不做后处理解析**：把 LLM 输出当字符串解析 JSON 迟早会失败
- **约束解码 > 验证**：验证只检查格式，约束解码在词元级保证格式

---

## 7. 常见错误

### 错误 1：提示词说 JSON 然后 parse

```python
# ❌ 错误：提示词 JSON + 后处理解析
prompt = "用 JSON 格式输出"
text = model.generate(prompt)
data = json.loads(text)  # 可能失败！

# ✓ 正确：使用 API 的结构化输出
response = client.beta.chat.completions.parse(
    response_format=MySchema,
)
```

### 错误 2：忽略嵌套字段的类型

**现象：** 父层正确但嵌套字段类型错误。

**修复：** 用 Pydantic 的递归验证检查整个结构。

---

## 8. 面试考点

### Q1：为什么约束解码比后处理 JSON 更可靠？（难度：⭐⭐）

**参考答案：**
后处理 JSON 在模型生成后才检查格式——如果格式错误，需要重试整个生成。约束解码在生成过程中用有限状态机（FSM）或上下文无关语法（CFG）约束词元的选择——确保每一步生成的词元都符合 JSON 语法。这不仅消除了格式错误，还避免了重试导致的输出不一致。

### Q2：OpenAI Structured Outputs 和 JSON Mode 有什么区别？（难度：⭐⭐）

**参考答案：**
JSON Mode（`response_format: {"type": "json_object"}`）只确保模型输出是一个 JSON 对象——不保证字段名和类型的正确性。Structured Outputs（`response_format: CalendarEvent`）通过一个包含 JSON Schema 的模型在服务器端做约束解码——确保输出完全匹配指定的 Schema。Structured Outputs 是 JSON Mode 的严格超集。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| JSON Mode | "强制 JSON" | API 参数——确保模型输出是有效的 JSON 对象 |
| Structured Outputs | "Schema 强制" | 用 JSON Schema 约束模型输出——字段名和类型均保证 |
| 约束解码 | "词元级控制" | 在解码阶段用 FSM/CFG 限制可选词元——100% 格式保证 |
| Pydantic | "Python 验证" | 数据验证库——定义 schema 并验证数据完整性 |

---

## 📚 小结

结构化输出是将 LLM 从文本生成器转换为可靠 API 的关键。提示词约束（90% 有效）不如 API 参数（99%）可靠，而 API 参数不如约束解码（100%）可靠。使用 OpenAI Structured Outputs 或 Anthropic Tool Use 做生产级的结构化输出。Pydantic 提供验证层。Instructor 自动化 Pydantic + API 集成。

---

## ✏️ 练习

1. **【实现】** 用 OpenAI Structured Outputs 从新闻文章提取结构化信息——标题、日期、作者、摘要。
2. **【实验】** 对比 JSON Mode vs Structured Outputs vs 提示词约束的 100 次调用的成功率。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 结构化提取 | `code/main.py` | Schema 定义 + 验证 + 重试的完整提取管道 |

---

## 📖 参考资料

1. [官方文档] OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
2. [官方文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. [GitHub] Instructor: https://github.com/jxnl/instructor
4. [论文] Outlines: Structure: https://github.com/outlines-dev/outlines

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
