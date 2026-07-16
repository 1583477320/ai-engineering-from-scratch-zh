# 函数调用深入——OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛到了同一个工具调用循环，然后在其他所有方面分化。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课并排对比三种格式，确保在一个提供商上发布的代码在移植时不会崩溃。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 01（工具接口）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 OpenAI、Anthropic、Gemini 函数调用载荷的三个形状差异
- [ ] 将一个工具声明在三种提供商格式之间转换
- [ ] 使用各提供商的 `tool_choice` 强制、禁止或自动选择工具调用
- [ ] 了解各提供商的硬限制和违反限制时的错误签名

---

## 1. 问题

函数调用请求的形状因提供商而异。三个具体例子来自 2026 年的生产技术栈：

- **OpenAI**：`tools` 数组 + `tool_calls` 响应 + `tool_choice`
- **Anthropic**：`tools` 数组 + `tool_use` 内容块 + `tool_result` 用户消息
- **Gemini**：`functionDeclarations` + 唯一 ID 关联 + `functionCall`/`functionResponse`

如果你为 OpenAI 写了工具调用代码——移植到 Anthropic 时会崩溃。本课让你理解这些差异。

---

## 2. 概念

### 2.1 三种格式对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 工具声明 | `tools` 数组 | `tools` 数组 | `functionDeclarations` |
| 模型输出 | `tool_calls` 数组 | `tool_use` 内容块 | `functionCall` |
| 结果传回 | `tool` 角色消息 | `tool_result` 用户消息 | `functionResponse` |
| 强制调用 | `tool_choice: "any"` | `tool_choice` 指定名称 | `tool_config: {"mode": "ANY"}` |
| 禁止调用 | `tool_choice: "none"` | `tool_choice: "none"` | 不提供工具 |

### 2.2 工具数量限制

| 提供商 | 最大工具数 | 超限错误 |
|--------|-----------|---------|
| OpenAI GPT-4 | 128 | 截断或错误 |
| Anthropic Claude | 128 | 截断或错误 |
| Gemini | 无硬限制 | 性能下降 |

### 2.3 Schema 深度限制

| 提供商 | 最大深度 | 说明 |
|--------|---------|------|
| OpenAI | 5 层嵌套 | 超出截断 |
| Anthropic | 无硬限制 | 建议不超过 3 层 |
| Gemini | 无硬限制 | 过深影响性能 |

---

## 3. 从零实现

### Step 1：统一工具声明转换器

```python
def convert_tool_to_all_formats(name, description, schema):
    """将工具声明转换为三种提供商格式。"""
    openai_tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        }
    }
    anthropic_tool = {
        "name": name,
        "description": description,
        "input_schema": schema,
    }
    gemini_tool = {
        "name": name,
        "description": description,
        "parameters": schema,
    }
    return {
        "openai": openai_tool,
        "anthropic": anthropic_tool,
        "gemini": gemini_tool,
    }
```

### Step 2：统一工具执行

```python
class UnifiedToolExecutor:
    """统一工具执行器——适配三种提供商格式。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, executor, description="", schema=None):
        self.tools[name] = {"executor": executor, "description": description, "schema": schema}

    def execute_from_openai(self, tool_calls):
        """处理 OpenAI 格式的工具调用。"""
        results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            result = self.tools[name]["executor"](**args)
            results.append({"tool_call_id": tc["id"], "result": str(result)})
        return results

    def execute_from_anthropic(self, tool_use_blocks):
        """处理 Anthropic 格式的工具调用。"""
        results = []
        for block in tool_use_blocks:
            if block["type"] == "tool_use":
                name = block["name"]
                args = block["input"]
                result = self.tools[name]["executor"](**args)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": str(result),
                })
        return results
```

---

## 4. 工具

### 4.1 统一接口模式

```python
# 不管底层是什么提供商，调用方应该看到统一的接口
class LLMToolClient:
    def __init__(self, provider="openai"):
        self.provider = provider

    def call_with_tools(self, prompt, tools):
        if self.provider == "openai":
            return self._call_openai(prompt, tools)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, tools)

    def _call_openai(self, prompt, tools):
        # OpenAI 特定实现
        pass

    def _call_anthropic(self, prompt, tools):
        # Anthropic 特定实现
        pass
```

---

## 6. 工程最佳实践

### 6.1 多提供商策略

- **抽象层**：不要直接调用特定提供商的 API——用统一接口
- **提供商切换**：一个提供商故障时自动切换到另一个
- **测试覆盖**：确保工具在所有提供商上都能正常工作

### 6.2 踩坑经验

- **Schema 格式不兼容**：OpenAI 用 `properties`，Anthropic 用 `input_schema`，Gemini 用 `parameters`
- **错误处理不统一**：不同提供商对超限、无效参数的错误返回不同
- **工具数量超限**：超过 128 个工具时某些提供商可能截断或报错

---

## 7. 常见错误

### 错误 1：为一个提供商写的工具调用代码直接移植到另一个

**现象：** 运行时报类型错误或字段缺失。

**修复：** 使用抽象层封装三种格式的差异——只暴露统一接口。

### 错误 2：忽略 tool_choice 的差异

**现象：** 想强制使用某个工具但参数格式不匹配。

**修复：** OpenAI 用 `tool_choice: {"type": "function", "function": {"name": "..."}}`，Anthropic 用 `tool_choice: {"type": "tool", "name": "..."}`，Gemini 用 `tool_config: {"mode": "ANY"}`。

---

## 8. 面试考点

### Q1：OpenAI 和 Anthropic 的函数调用有什么本质区别？（难度：⭐⭐）

**参考答案：**
两者都实现了相同的四步循环，但消息格式不同。OpenAI 将工具调用放在 `tool_calls` 数组中，结果通过 `tool` 角色消息传回。Anthropic 将工具调用放在 `tool_use` 内容块中，结果通过 `tool_result` 用户消息传回。关键差异：(1) Anthropic 的结果必须在下一条用户消息中传回；(2) Anthropic 支持并行工具调用（多个 tool_use 块）；(3) OpenAI 有更丰富的 `tool_choice` 选项。

### Q2：如何设计跨提供商的统一工具接口？（难度：⭐⭐⭐）

**参考答案：**
定义一个抽象接口，每个工具只实现一次，然后在三个提供商上适配：(1) 工具注册表存储统一格式（名称、描述、JSON Schema、执行函数）；(2) 格式转换器将统一格式转换为每个提供商的格式；(3) 结果解析器处理不同提供商的工具调用响应格式。这样工具只需要写一次，通过适配层支持所有提供商。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| tool_choice | "强制/禁止工具调用" | 控制 LLM 是否必须调用某个工具——"any"强制、"none"禁止 |
| tool_calls | "工具调用列表" | OpenAI 格式的模型输出——包含名称和参数 |
| tool_use | "工具使用块" | Anthropic 格式的模型输出——内容块而非 JSON 字段 |
| functionDeclarations | "函数声明" | Gemini 格式的工具定义——与 OpenAI 的 tools 类似但语法不同 |

---

## 📚 小结

三大提供商的函数调用在四步循环上收敛但在格式上分化。OpenAI/Anthropic/Gemini 的工具声明、调用、结果格式各不相同。生产环境需要抽象层适配差异。关键差异：OpenAI 用 tool_calls 数组，Anthropic 用 tool_use 内容块，Gemini 用 functionDeclarations。

---

## ✏️ 练习

1. **【实现】** 将一个"获取天气"工具声明在三种格式之间转换——验证格式兼容
2. **【实验】** 在 OpenAI 和 Anthropic 上分别调用同一个工具——对比响应格式

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 统一工具执行器 | `code/main.py` | 三种提供商格式转换 + 统一执行 |

---

## 📖 参考资料

1. [文档] OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
2. [文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
3. [文档] Gemini Function Calling: https://ai.google.dev/docs/function-calling

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
