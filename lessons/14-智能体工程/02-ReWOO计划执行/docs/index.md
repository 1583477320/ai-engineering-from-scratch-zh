# ReWOO：计划与执行的分离

> ReAct 交替推理和行动——每一步都调用 LLM。但对于复杂任务，LLM 调用是瓶颈。ReWOO 将计划和执行分离：一个 LLM 生成完整的行动计划，然后执行器逐步执行——LLM 只调用一次。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 ReWOO 的"计划-执行"分离原理
- [ ] 对比 ReWOO 和 ReAct 的成本和延迟
- [ ] 实现一个简单的计划-执行智能体
- [ ] 理解何时应该选择 ReWOO 而非 ReAct

---

## 1. 问题

ReAct 的每个推理步骤都需要一次 LLM 调用。对于需要 5-10 步的任务，这意味着 5-10 次 LLM 调用——每次都有延迟和成本。

**ReWOO 的核心洞察**：计划和执行可以分离。LLM 只需要在开始时生成完整的行动计划（一次调用），然后执行器逐步执行（无需 LLM）。

```
ReAct:  LLM推理 → 工具 → LLM推理 → 工具 → LLM推理 → 工具（3次LLM）
ReWOO:  LLM生成完整计划 → 工具 → 工具 → 工具（1次LLM）
```

---

## 2. 概念

### 2.1 ReWOO 架构

```
用户查询 → [规划器: LLM] → 执行计划 (E1, E2, E3, ...)
                        ↓
[执行器] → 执行 E1 → 执行 E2 → 执行 E3 → ...
                        ↓
结果汇总 → [总结器: LLM] → 最终回答
```

### 2.2 计划格式

```yaml
E1: get_weather(city="北京")
E2: search(query="北京明天天气预报")
E3: generate_answer(context=[E1结果, E2结果])
```

### 2.3 ReWOO vs ReAct

| 方面 | ReAct | ReWOO |
|------|-------|-------|
| LLM 调用次数 | N 次（每步一次） | 1-2 次（规划+总结） |
| 延迟 | N × LLM延迟 | 1 × LLM延迟 + N × 工具延迟 |
| 成本 | N × token 消耗 | 1 × token 消耗 |
| 适应性 | 每步根据反馈调整 | 计划固定，可能过时 |
| 适用场景 | 需要灵活调整的动态任务 | 计划明确的静态任务 |

### 2.4 何时用 ReWOO vs ReAct

| 场景 | 推荐 | 原因 |
|------|------|------|
| 计划明确（查询+检索+回答） | ReWOO | 无需中间推理 |
| 需要灵活调整（探索性任务） | ReAct | 每步根据反馈决策 |
| LLM 成本敏感 | ReWOO | 减少 LLM 调用次数 |
| 工具延迟低 | ReAct | LLM 调用不是瓶颈 |

---

## 3. 从零实现

### Step 1：ReWOO 规划器

```python
class ReWOOPlanner:
    """ReWOO 规划器——生成完整的行动计划。"""
    def __init__(self, llm_fn, tools):
        self.llm_fn = llm_fn
        self.tools = list(tools.keys())

    def plan(self, query):
        """生成执行计划。"""
        tools_desc = ", ".join(self.tools)
        prompt = f"""根据用户查询，生成一个执行计划。

可用工具：{tools_desc}

用户查询：{query}

请按顺序列出需要执行的步骤（每行一个）：
E1: [工具名(参数)]
E2: [工具名(参数)]
...
"""

        # 简化：模拟 LLM 生成计划
        plan = [
            {"step": 1, "tool": "get_weather", "args": {"city": "北京"}},
            {"step": 2, "tool": "search", "args": {"query": "北京明天天气"}},
        ]
        return plan
```

### Step 2：执行器

```python
class ReWOOExecutor:
    """ReWOO 执行器——逐步执行计划。"""
    def __init__(self, tools):
        self.tools = tools
        self.results = {}

    def execute_plan(self, plan):
        """执行完整计划。"""
        for step in plan:
            tool_name = step["tool"]
            args = step["args"]

            # 替换前一步的结果
            for k, v in list(args.items()):
                if isinstance(v, str) and v.startswith("$E"):
                    ref_step = int(v[2:])
                    args[k] = self.results.get(ref_step, v)

            # 执行工具
            if tool_name in self.tools:
                result = self.tools[tool_name](**args)
            else:
                result = f"工具 {tool_name} 不存在"

            self.results[step["step"]] = result
            print(f"  E{step['step']}: {tool_name}({args}) → {str(result)[:50]}")

        return self.results
```

### Step 3：完整 ReWOO 管道

```python
def rewoo_pipeline(query, planner, executor):
    """ReWOO 完整管道：规划 → 执行 → 总结。"""
    # 1. 规划（一次 LLM 调用）
    plan = planner.plan(query)
    print(f"执行计划: {[s['tool'] for s in plan]}")

    # 2. 执行（无 LLM 调用）
    results = executor.execute_plan(plan)

    # 3. 总结结果
    summary = " | ".join(f"E{s['step']}={results[s['step']]}" for s in plan)
    return summary
```

---

## 4. 工具

### 4.1 框架支持

| 框架 | ReAct | ReWOO | 说明 |
|------|-------|-------|------|
| LangChain | ✅ | 部分支持 | ReAct 原生 |
| LangGraph | ✅ | ✅ | 灵活的状态图 |
| LlamaIndex | ✅ | 部分支持 | 主要 RAG |

### 4.2 工具库

| 工具 | 功能 |
|------|------|
| LangGraph | 有状态图执行 |
| ReWOO 论文实现 | 学术参考 |

---

## 5. 工程最佳实践

### 5.1 ReWOO 设计原则

- **计划要完整**：规划阶段应输出所有需要的工具调用
- **结果可引用**：用 `$E1` 引用前一步结果
- **错误处理**：某步失败时的回退策略

### 5.2 踩坑经验

- **计划不准确**：LLM 规划的参数可能有误——需要验证
- **计划过时**：执行中环境变化——需要重新规划
- **LLM 规划成本**：长计划需要大上下文——token 限制

---

## 7. 常见错误

### 错误 1：用 ReAct 实现固定流程任务

**现象：** LLM 每步都调用，延迟和成本不必要地高。

**修复：** 固定流程任务用 ReWOO——一次 LLM 调用生成计划。

### 错误 2：ReWOO 计划中引用错误

**现象：** 计划中的 `$E1` 引用了不存在的步骤。

**修复：** 在规划时确保步骤编号连续且引用正确。

---

## 8. 面试考点

### Q1：ReWOO 和 ReAct 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
ReAct 交替推理和行动——每步都调用 LLM，灵活但成本高。ReWOO 将计划和执行分离——LLM 只在开始时生成完整计划（一次调用），执行器逐步执行（无 LLM）。ReWOO 更适合固定流程任务，ReAct 更适合需要灵活调整的动态任务。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| ReWOO | "规划-执行分离" | 先让 LLM 生成完整计划，再由执行器逐步执行——减少 LLM 调用次数 |
| ReAct | "推理+行动" | 交替推理和行动——每步都调用 LLM |
| 计划格式 | "步骤列表" | YAML/文本格式的执行计划——E1: tool(args) |
| 执行器 | "计划执行者" | 按计划逐步执行工具调用——无需 LLM |

---

## 📚 小结

ReWOO 将计划和执行分离——LLM 只在规划时调用一次，执行器按计划逐步执行。ReWOO 适合固定流程任务（延迟/成本低），ReAct 适合动态任务（灵活但成本高）。选择取决于任务是否需要中间推理。

---

## ✏️ 练习

1. **【对比】** 对比 ReAct 和 ReWOO 在"查天气+生成回答"任务上的 LLM 调用次数
2. **【实现】** 构建一个 ReWOO 管道——规划器生成计划，执行器执行

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| ReWOO 实现 | `code/main.py` | 规划器 + 执行器 + 完整管道 |

---

## 📖 参考资料

1. [论文] Xu et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought by Large Language Models". ACL, 2023.
2. [论文] Wu et al. "ReWOO: Unpacking Reasoning in LLMs". 2023.
3. [论文] Wang et al. "Self-Consistency Improves Chain of Thought Reasoning". ICLR, 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
