# 智能体循环——感知-推理-行动的核心

> 一个智能体不是一个函数调用。它是一个循环：观察环境、推理下一步、执行动作、接收反馈。这个循环是所有智能体系统的基础——从简单的 ReAct 到复杂的多智能体协作。理解这个循环，你就理解了智能体工程的全部。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 09（函数调用）、阶段 13（MCP）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义智能体循环的四个阶段——感知、推理、行动、反馈
- [ ] 实现 ReAct 模式——交替推理和行动
- [ ] 理解智能体与普通 LLM 应用的区别——状态、记忆、自主性
- [ ] 设计一个简单的智能体系统——选择工具、执行动作、处理结果

---

## 1. 问题

普通 LLM 应用是"一问一答"——用户输入，模型输出，结束。智能体不同——它**自主决策**：观察环境，推理下一步做什么，执行动作，观察结果，再决定下一步。

关键区别：普通 LLM 没有状态、没有记忆、没有自主性。智能体有这三个——它能记住之前做了什么、判断当前状况、自主决定下一步。

---

## 2. 概念

### 2.1 智能体循环

```
观察 (Observation) → 推理 (Reasoning) → 行动 (Action) → 反馈 (Feedback)
      ↑                                                        ↓
      └────────────────────────────────────────────────────────┘
```

### 2.2 智能体 vs 普通 LLM

| 方面 | 普通 LLM | 智能体 |
|------|---------|--------|
| 输入 | 单次查询 | 持续观察 |
| 状态 | 无 | 有（记忆） |
| 决策 | 被动响应 | 主动规划 |
| 工具 | 可选 | 必需 |
| 循环 | 无 | 有 |

### 2.3 ReAct 模式

Reasoning + Acting 交替进行：

```
观察：今天北京天气22°C
推理：用户需要明天的天气预报，我需要调用天气API
行动：get_weather(city="北京", date="tomorrow")
反馈：明天北京晴天，最高温度25°C
推理：天气信息已获取，现在可以回答用户了
行动：生成回答
```

### 2.4 关键组件

| 组件 | 功能 | 工具 |
|------|------|------|
| **感知** | 获取环境信息 | VLM、ASR、传感器 |
| **推理** | 决定下一步行动 | LLM、思维链 |
| **行动** | 执行动作 | MCP 工具、API 调用 |
| **反馈** | 接收执行结果 | 工具返回值、环境状态 |
| **记忆** | 存储历史信息 | 对话历史、向量数据库 |

---

## 3. 从零实现

### Step 1：智能体循环核心

```python
class Agent:
    """简化版智能体——感知-推理-行动循环。"""
    def __init__(self, llm_fn, tools):
        self.llm_fn = llm_fn
        self.tools = tools
        self.memory = []

    def run(self, user_query, max_steps=5):
        """运行智能体循环。"""
        observation = user_query
        self.memory.append({"role": "user", "content": observation})

        for step in range(max_steps):
            # 1. 推理：LLM 决定下一步
            action = self.llm_fn(observation, tools=self.tools)

            # 2. 执行动作
            if action["type"] == "respond":
                self.memory.append({"role": "assistant", "content": action["content"]})
                return action["content"]

            # 3. 调用工具
            tool_result = self.execute_tool(action)

            # 4. 反馈：将结果加入上下文
            self.memory.append({"role": "tool", "content": str(tool_result)})
            observation = f"工具 {action['tool']} 返回: {tool_result}"

        return "超过最大步数限制"

    def execute_tool(self, action):
        tool_name = action.get("tool", "")
        args = action.get("args", {})
        if tool_name in self.tools:
            return self.tools[tool_name](**args)
        return f"工具 {tool_name} 不存在"
```

### Step 2：ReAct 提示词模板

```python
REACT_PROMPT = """你是一个有帮助的智能体。你可以使用以下工具：

可用工具：
- get_weather(city): 获取天气
- search(query): 搜索信息
- calculator(expression): 计算数学表达式

请按照以下格式回答：
思考: [你的推理过程]
行动: [工具名称(参数)]
观察: [工具返回的结果]
回答: [最终回答]"""
```

### Step 3：简单工具库

```python
TOOLS = {
    "get_weather": lambda city=f"北京": f"{city}: 晴天, 22°C, 湿度 45%",
    "search": lambda query="": f"搜索结果: 关于'{query}'的3个相关网页",
    "calculator": lambda expression="1+1": f"计算结果: {eval(expression)}",
}
```

---

## 4. 工具

### 4.1 框架对比

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| ReAct | 推理+行动交替 | 简单智能体 |
| Plan-and-Execute | 先计划后执行 | 复杂任务 |
| Reflexion | 自我反思改进 | 需要迭代优化 |
| ToT | 多路径探索 | 复杂推理任务 |

### 4.2 工具库

| 工具 | 功能 |
|------|------|
| LangChain | 智能体编排 |
| LangGraph | 有状态图执行 |
| LlamaIndex | RAG + 工具 |

---

## 6. 工程最佳实践

### 6.1 智能体设计原则

- **最小权限**：智能体只应访问必要的工具
- **最大透明度**：每个决策步骤都应可追踪
- **优雅降级**：工具失败时有回退策略
- **循环限制**：设置最大步数防止无限循环

### 6.2 踩坑经验

- **无限循环**：智能体反复执行相同动作——设置 max_steps 和重复检测
- **工具选择错误**：LLM 选错了工具——优化工具描述
- **上下文溢出**：长对话历史消耗太多 token——定期摘要

---

## 7. 常见错误

### 错误 1：没有循环限制

**现象：** 智能体陷入无限循环——不断调用工具但从不结束。

**修复：** 设置 max_steps（通常 5-10）和重复动作检测。

### 错误 2：不记录历史

**现象：** 智能体重复做相同的事情——没有记住之前的结果。

**修复：** 在每步将结果加入记忆——LLM 可以看到之前的行动和反馈。

---

## 8. 面试考点

### Q1：智能体循环的四个阶段是什么？（难度：⭐⭐）

**参考答案：**
(1) **观察**：获取环境信息——用户查询、工具返回值、系统状态；(2) **推理**：LLM 分析当前状况，决定下一步——可能需要调用工具或直接回答；(3) **行动**：执行决定——调用工具、生成文本、发送 API；(4) **反馈**：接收执行结果——工具返回值、错误信息、新状态。循环直到任务完成或达到步数上限。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 智能体循环 | "感知-推理-行动" | 观察→推理→行动→反馈的持续循环 |
| ReAct | "推理+行动" | Reasoning + Acting——交替推理和执行 |
| 最大步数 | "循环上限" | 防止无限循环的步数限制 |
| 上下文管理 | "记忆管理" | 智能体如何维护和截断历史信息 |

---

## 📚 小结

智能体循环是所有智能体系统的基础：观察→推理→行动→反馈。ReAct 是最常见的实现模式——交替推理和执行。关键组件：LLM 推理、工具调用、记忆管理、循环限制。理解这个循环，就理解了智能体工程的核心。

---

## ✏️ 练习

1. **【实现】** 构建一个简单的智能体——能调用天气/搜索/计算器三个工具完成用户查询
2. **【实验】** 对比有/无 max_steps 限制时智能体的行为差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 智能体循环 | `code/main.py` | 感知-推理-行动循环 + ReAct 模式 |

---

## 📖 参考资料

1. [论文] Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models". ICLR, 2023. https://arxiv.org/abs/2210.03629
2. [论文] Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning". NeurIPS, 2023.
3. [论文] Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models". ICLR, 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
