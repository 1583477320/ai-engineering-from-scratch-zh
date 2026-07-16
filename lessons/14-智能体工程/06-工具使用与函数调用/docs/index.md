# 工具使用与函数调用

> Toolformer（Schick 等人，2023）开创了自监督工具注释。Berkeley 函数调用排行榜 V4（Patil 等人，2025）设立了 2026 年的标准：40% 智能体、30% 多轮、10% 实时、10% 非实时、10% 幻觉。单轮已解决。记忆、动态决策和长期工具链尚未解决。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、阶段 13 · 01（函数调用深入）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 Toolformer 的自监督训练信号——只在执行减少下一个 token 损失时保留工具注释
- [ ] 实现工具选择和参数生成——从工具库中选择最合适的工具
- [ ] 理解多轮工具调用的上下文管理
- [ ] 设计工具调用的错误处理和重试策略

---

## 1. 问题

LLM 不能计算、不能查数据库、不能访问实时信息。但它可以通过**工具调用**间接做到这些——只要它知道什么工具可用、何时使用、如何调用。

Toolformer 的核心洞察：**工具使用可以通过自监督学习**——让模型学习何时和如何调用工具，只在工具调用确实帮助预测下一个 token 时才保留。

---

## 2. 概念

### 2.1 工具使用管道

```
用户输入 → [LLM 生成工具调用] → [执行工具] → [结果注入] → [LLM 生成回答]
```

### 2.2 Toolformer 训练信号

```
输入: "巴黎的天气是" + [天气工具调用]
模型预测下一个词元

如果工具调用后：
  无工具 → P("晴天") = 0.3
  有工具 → P("晴天") = 0.9

差异 = 0.6 → 工具调用有价值 → 保留这个训练样本
```

### 2.3 工具选择策略

| 策略 | 描述 |
|------|------|
| **工具名匹配** | LLM 根据工具描述选择 |
| **参数生成** | LLM 生成工具调用的参数 JSON |
| **结果验证** | 检查工具返回值是否有效 |
| **多工具协调** | 一次调用多个工具 |

### 2.4 错误处理

```
工具调用失败 → 检查错误类型
  ├── 参数错误 → 修正参数重试
  ├── 超时 → 等待后重试
  ├── 权限不足 → 请求权限或跳过
  └── 未知错误 → 记录并继续
```

---

## 3. 从零实现

### Step 1：工具管理器

```python
class ToolManager:
    """工具管理器——注册、选择、执行。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, description, executor):
        self.tools[name] = {"description": description, "executor": executor}

    def select_tool(self, query):
        """LLM 选择最合适的工具（简化：关键词匹配）。"""
        for name, info in self.tools.items():
            if name in query.lower() or any(k in query for k in name.split("_")):
                return name
        return None

    def execute(self, tool_name, args):
        if tool_name in self.tools:
            return self.tools[tool_name]["executor"](**args)
        return f"工具 {tool_name} 不存在"
```

### Step 2：带工具的 LLM 调用

```python
def llm_with_tools(llm_fn, query, tools):
    """带工具的 LLM 调用。"""
    # 检查是否需要工具
    tool_name = tools.select_tool(query)
    if tool_name:
        result = tools.execute(tool_name, {"query": query})
        return f"工具结果: {result}"
    return llm_fn(query)
```

---

## 4. 工具

### 4.1 函数调用基准

| 基准 | 内容 |
|------|------|
| Berkeley Function Calling Leaderboard | 单轮+多轮+实时+幻觉 |
| ToolBench | 大规模工具调用评估 |
| API-Bank | API 调用能力评估 |

### 4.2 框架对比

| 框架 | 工具支持 |
|------|---------|
| OpenAI Function Calling | 原生支持 |
| Anthropic Tool Use | 原生支持 |
| LangChain | 统一工具接口 |

---

## 5. 工程最佳实践

### 5.1 工具选择设计

- **工具描述要精确**：帮助 LLM 选择正确工具
- **参数 Schema 完整**：类型、描述、必填/可选
- **错误处理健壮**：超时、权限、格式错误都要处理

### 5.2 踩坑经验

- **工具选择错误**：描述不精确→LLM 选错工具
- **参数格式错误**：LLM 生成的 JSON 格式不符合 Schema
- **工具执行超时**：长运行工具需要异步执行

---

## 6. 常见错误

### 错误 1：工具描述不够精确

**现象：** LLM 选错工具。

**修复：** 在描述中添加排除条件——"当 X 时使用此工具，不要用于 Y"。

### 错误 2：没有错误处理

**现象：** 工具失败时智能体崩溃。

**修复：** 每次工具调用都需要 try/except，失败时重试或跳过。

---

## 7. 面试考点

### Q1：Toolformer 的自监督训练信号是什么？（难度：⭐⭐）

**参考答案：**
Toolformer 用下一个 token 预测的损失变化作为训练信号。它在文本中注入工具调用，如果注入后降低了下一个词元的预测损失，说明工具调用有帮助——保留这个训练样本。否则丢弃。这意味着模型自动学会了何时以及如何调用工具。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Toolformer | "自监督工具学习" | LLM 自动学习何时调用工具——通过降低下一个词元损失来判断 |
| 函数调用 | "LLM 调用工具" | LLM 生成结构化的工具调用请求（名称+参数 JSON） |
| 工具选择 | "选哪个工具" | LLM 根据查询意图从可用工具中选择最合适的工具 |

---

## 📚 小结

Toolformer 证明了 LLM 可以通过自监督学习使用工具——只在调用帮助预测时才保留。Berkeley 排行榜设立了 2026 年的标准。单轮已解决，多轮和长期工具链是下一个前沿。

---

## ✏️ 练习

1. **【实现】** 构建工具管理器——支持注册、选择、执行和错误处理
2. **【实验】** 对比有/无工具调用时 LLM 在数学问题上的表现

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 工具管理器 | `code/main.py` | 工具注册/选择/执行/错误处理 |

---

## 📖 参考资料

1. [论文] Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools". NeurIPS, 2023.
2. [论文] Patil et al. "Berkeley Function Calling Leaderboard". 2025.
3. [论文] Qin et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs". ICLR, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
