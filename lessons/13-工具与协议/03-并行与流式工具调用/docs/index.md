# 并行工具调用与流式处理

> 三个独立的天气查询串行化就是三次往返。并行执行总时间缩减为最慢单次调用的时间。每个前沿提供商现在在单轮中发出多个工具调用。收益是真实的；管道是微妙的。本课讲解两半：并行扇出和流式参数重组，重点是 ID 关联陷阱。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用深入）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么 `parallel_tool_calls: true` 存在以及何时禁用它
- [ ] 实现并行工具调用的扇出和结果聚合
- [ ] 处理流式响应中的工具调用——如何在 token 流中检测和重组工具调用
- [ ] 理解 ID 关联陷阱——为什么工具调用顺序可能与响应顺序不一致

---

## 1. 问题

如果你的 LLM 需要查三个城市的天气，串行化意味着三次往返——每次等待上一次完成。并行化意味着同时发出三个调用——总时间等于最慢的一个。

但并行化有管道问题：工具调用可能乱序返回，结果需要正确关联，流式响应中工具调用的边界需要正确检测。

---

## 2. 概念

### 2.1 并行工具调用

```
LLM 单轮输出:
  tool_call_1: get_weather(city="北京")
  tool_call_2: get_weather(city="上海")
  tool_call_3: get_weather(city="广州")

宿主并行执行:
  thread_1: get_weather("北京") → 22°C
  thread_2: get_weather("上海") → 25°C
  thread_3: get_weather("广州") → 28°C

结果按 ID 关联后返回 LLM
```

### 2.2 流式响应中的工具调用

流式响应逐 token 输出——工具调用 JSON 在 token 流中逐字符构建。需要正确检测工具调用的开始和结束边界。

### 2.3 ID 关联陷阱

LLM 的工具调用输出可能不按顺序。`tool_call_1` 的结果可能比 `tool_call_3` 的结果先返回。必须用 `tool_call_id` 正确关联。

### 2.4 parallel_tool_calls 参数

| 提供商 | 参数 | 说明 |
|--------|------|------|
| OpenAI | `parallel_tool_calls: true/false` | 控制是否并行 |
| Anthropic | 自动并行 | 无显式参数 |
| Gemini | 自动并行 | 无显式参数 |

---

## 3. 从零实现

### Step 1：并行工具执行器

```python
import concurrent.futures

class ParallelToolExecutor:
    """并行工具执行器。"""
    def __init__(self, registry):
        self.registry = registry

    def execute_parallel(self, tool_calls):
        """并行执行多个工具调用。"""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                future = executor.submit(self.registry.execute, name, args)
                futures[future] = tc["id"]

            for future in concurrent.futures.as_completed(futures):
                tc_id = futures[future]
                try:
                    result = future.result()
                    results[tc_id] = {"result": result, "success": True}
                except Exception as e:
                    results[tc_id] = {"error": str(e), "success": False}

        return results
```

### Step 2：流式工具调用检测

```python
def detect_tool_calls_from_stream(token_stream):
    """从 token 流中检测工具调用。"""
    buffer = ""
    in_tool_call = False
    tool_calls = []

    for token in token_stream:
        buffer += token
        if '"function":' in buffer and '{' in buffer:
            in_tool_call = True
        if in_tool_call and '}' in buffer and 'name' in buffer:
            # 简化：提取工具调用
            tool_calls.append({"id": str(len(tool_calls)), "buffer": buffer})
            in_tool_call = False
            buffer = ""

    return tool_calls
```

---

## 4. 工具

### 4.1 OpenAI 并行调用

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "查北京、上海、广州的天气"}],
    tools=tools,
    parallel_tool_calls=True,  # 允许并行
)
# 返回多个 tool_calls
```

### 4.2 Anthropic 并行调用

```python
response = client.messages.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "查北京、上海、广州的天气"}],
    tools=tools,
    # Anthropic 自动并行，无需显式参数
)
```

---

## 5. 工程最佳实践

### 5.1 并行 vs 串行选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 多个独立查询 | 并行 | 总时间 = 最慢单次 |
| 查询有依赖 | 串行 | 前一个结果影响后一个 |
| 工具有副作用 | 串行 | 避免并发冲突 |

### 5.2 踩坑经验

- **ID 关联错误**：结果必须用 `tool_call_id` 关联——不能按顺序假设
- **流式边界检测**：工具调用 JSON 在 token 流中逐字符构建——需要正确检测边界

---

## 7. 常见错误

### 错误 1：并行执行有副作用的工具

**现象：** 两个工具同时修改同一文件——数据损坏。

**修复：** 有副作用的工具应串行执行——或加锁机制。

### 错误 2：流式响应中丢弃工具调用

**现象：** 工具调用 JSON 在流式 token 中不完整——解析失败。

**修复：** 缓冲 token 直到完整 JSON 构建完成——使用状态机检测边界。

---

## 8. 面试考点

### Q1：并行工具调用的 ID 关联陷阱是什么？（难度：⭐⭐）

**参考答案：**
LLM 的工具调用输出可能不按顺序——`tool_call_1` 的响应可能比 `tool_call_3` 的响应先返回。如果按顺序假设关联，结果会错配。必须用 `tool_call_id` 精确关联。在异步执行环境中（如多线程），这个问题更加突出——不同线程的完成时间不可预测。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 并行工具调用 | "同时查多个东西" | LLM 在单轮中发出多个工具调用，宿主并行执行 |
| ID 关联 | "结果对应哪个调用" | 用 tool_call_id 精确关联工具调用和响应 |
| 流式工具调用 | "边生成边执行" | 在 token 流中逐字符构建工具调用 JSON |

---

## 📚 小结

并行工具调用减少延迟——多个独立查询同时执行。流式响应需要正确检测工具调用边界。ID 关联是并行执行的关键——必须用 `tool_call_id` 精确匹配调用和结果。有副作用的工具应串行执行。

---

## ✏️ 练习

1. **【实现】** 构建并行工具执行器——支持多线程执行和结果关联
2. **【实验】** 对比串行和并行执行三个 API 调用的总时间

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 并行执行器 | `code/main.py` | 多线程并行工具执行 |

---

## 📖 参考资料

1. [文档] OpenAI Parallel Tool Calls: https://platform.openai.com/docs/guides/function-calling
2. [文档] Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
