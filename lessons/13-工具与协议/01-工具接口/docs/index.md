# 工具接口——为什么智能体需要结构化 I/O

> 语言模型生成词元。程序执行动作。两者之间的鸿沟就是工具接口——一个让模型请求动作、宿主执行它的契约。2026 年的每个技术栈——OpenAI、Anthropic、Gemini 的函数调用；MCP 的 tools/call；A2A 的任务部分——都是同一个四步循环的不同编码。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 11（LLM 完成 API）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么只生成文本的 LLM 无法自行对真实世界执行动作
- [ ] 画出四步工具调用循环——描述→决策→执行→观察——并命名每步的拥有者
- [ ] 将工具描述写为三部分——名称、JSON Schema 输入、确定性执行函数
- [ ] 区分纯函数和有副作用的工具，说明这个区分对安全的重要性

---

## 1. 问题

LLM 产生下一个词元的概率分布。这是它的整个输出空间。如果你问聊天模型"班加罗尔现在的天气是什么"，它可以写一个看起来合理的句子，但它无法拨入天气 API。这个句子可能碰巧对了，也可能是三天前的数据。

LLM 不能执行动作——它只能生成文本。要让 LLM "做事"，需要一个桥梁：**工具接口**。

---

## 2. 概念

### 2.1 四步工具调用循环

```
用户提问
    ↓
LLM 生成工具调用描述（名称 + 参数 JSON）
    ↓
宿主执行工具函数
    ↓
结果返回给 LLM
    ↓
LLM 基于结果生成最终回答
```

### 2.2 工具描述的三部分

| 部分 | 示例 |
|------|------|
| **名称** | `get_weather` |
| **JSON Schema** | `{"city": {"type": "string"}, "unit": {"type": "string"}}` |
| **执行函数** | `def get_weather(city, unit): ...` |

### 2.3 纯函数 vs 有副作用的工具

| 类型 | 示例 | 安全考虑 |
|------|------|---------|
| **纯函数** | 计算、查询 | 无风险——只返回结果 |
| **有副作用** | 删除文件、发送邮件 | 需要权限控制和确认 |

---

## 3. 从零实现

### Step 1：最小工具接口

```python
class ToolRegistry:
    """工具注册表——管理可用工具。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, description, schema, executor):
        self.tools[name] = {
            "description": description,
            "schema": schema,
            "executor": executor,
        }

    def execute(self, name, arguments):
        if name not in self.tools:
            return {"error": f"工具 {name} 不存在"}
        try:
            return self.tools[name]["executor"](**arguments)
        except Exception as e:
            return {"error": str(e)}

    def get_tools_for_llm(self):
        """生成 LLM 可用的工具描述。"""
        return [
            {"type": "function", "function": {
                "name": name,
                "description": info["description"],
                "parameters": info["schema"],
            }}
            for name, info in self.tools.items()
        ]
```

### Step 2：四步循环实现

```python
def tool_call_loop(user_query, llm_fn, registry):
    """四步工具调用循环。"""
    # 1. LLM 决定是否调用工具
    response = llm_fn(user_query, tools=registry.get_tools_for_llm())

    if response.get("tool_calls"):
        # 2. 执行工具
        for tool_call in response["tool_calls"]:
            result = registry.execute(
                tool_call["function"]["name"],
                tool_call["function"]["arguments"]
            )
            # 3. 结果返回给 LLM
            response = llm_fn(
                f"工具 {tool_call['function']['name']} 返回: {result}",
                tools=registry.get_tools_for_llm()
            )

    # 4. 生成最终回答
    return response["content"]
```

---

## 4. 工具

### 4.1 OpenAI 函数调用

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
}]
```

### 4.2 Anthropic 工具使用

```python
tools = [{
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]
```

---

## 6. 工程最佳实践

### 6.1 工具设计原则

- **确定性**：相同输入应产生相同输出（纯函数优先）
- **幂等性**：重复执行不产生额外副作用
- **可观测**：工具执行应产生日志和指标
- **错误处理**：返回结构化错误而非抛出异常

### 6.2 中文场景

- 工具描述用中英文双语——提高 LLM 理解
- JSON Schema 中的描述用中文——帮助中文用户

---

## 7. 常见错误

### 错误 1：工具描述不精确

**现象：** LLM 选错了工具或传了错误参数。

**修复：** 在描述中明确说明输入格式、限制条件和使用场景。

### 错误 2：有副作用的工具没有确认机制

**现象：** LLM 自动删除了文件——用户没有机会确认。

**修复：** 高危操作需要用户确认——不要自动执行有副作用的工具。

---

## 8. 面试考点

### Q1：LLM 的工具调用循环是什么？（难度：⭐⭐）

**参考答案：**
四步循环：(1) 描述——LLM 生成工具调用描述（名称+参数）；(2) 决策——宿主决定是否执行；(3) 执行——运行工具函数；(4) 观察——结果返回给 LLM 生成最终回答。关键：LLM 只负责生成调用描述和最终回答，实际执行由宿主完成。

### Q2：为什么工具需要区分纯函数和有副作用？（难度：⭐⭐⭐）

**参考答案：**
纯函数（查询、计算）可以安全地自动执行——不会改变系统状态。有副作用的工具（删除文件、发送邮件）可能造成不可逆的损害。区分它们让系统设计者可以：(1) 对纯函数自动执行，对有副作用的工具要求用户确认；(2) 对有副作用的工具添加权限控制和日志审计；(3) 在沙箱环境中限制有副作用工具的权限。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 工具接口 | "LLM 和代码之间的桥梁" | 让 LLM 请求动作、宿主执行的标准协议 |
| 四步循环 | "描述→执行→观察" | LLM 生成调用、宿主执行、结果返回、LLM 生成回答 |
| JSON Schema | "参数格式说明" | 用 JSON 定义工具输入的类型、描述、约束 |
| 有副作用工具 | "会改变状态的操作" | 执行后会修改系统状态的工具——需要安全控制 |

---

## 📚 小结

工具接口是 LLM 从"文本生成器"到"智能体"的关键桥梁。四步循环：描述→决策→执行→观察。工具描述需要名称、JSON Schema、执行函数三部分。纯函数可自动执行，有副作用的工具需要安全控制。

---

## ✏️ 练习

1. **【实现】** 构建一个工具注册表——支持注册、查询、执行三个操作
2. **【设计】** 设计一个带确认机制的有副作用工具——"删除文件"需要用户确认

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 工具注册表 | `code/main.py` | 四步循环 + 工具注册/执行 |

---

## 📖 参考资料

1. [文档] OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
2. [文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. [文档] MCP 规范: https://spec.modelcontextprotocol.io

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
