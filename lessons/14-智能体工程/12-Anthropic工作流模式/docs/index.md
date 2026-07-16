# Anthropic 工作流模式：简单胜过复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）和智能体（动态工具使用）。五种工作流模式覆盖了大多数情况。从直接 API 调用开始。只有当步骤无法预测时才添加智能体。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 Anthropic 的五种工作流模式——提示链、路由、并行化、编排器-工作者、评估器-优化器
- [ ] 根据任务特征选择合适的工作流模式
- [ ] 实现每个工作流模式的简化版本
- [ ] 理解何时应该从工作流切换到智能体

---

## 1. 问题

不是所有 LLM 应用都需要"智能体"——很多任务用预定义的工作流就足够了。Anthropic 的经验：**从最简单的方案开始，只在必要时增加复杂性。** 五种工作流模式覆盖了 90% 的场景。

---

## 2. 概念

### 2.1 五种工作流模式

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| **提示链** | 顺序调用，每步的输出作为下一步的输入 | 多步骤任务（先分析后生成） |
| **路由** | 根据输入选择不同的处理分支 | 分类/分发任务 |
| **并行化** | 多个步骤同时执行 | 多个独立任务（如多语言翻译） |
| **编排器-工作者** | 一个编排器调度多个工作者 | 复杂任务分解 |
| **评估器-优化器** | 生成→评估→改进循环 | 需要迭代优化的任务 |

### 2.2 工作流 vs 智能体

| 方面 | 工作流 | 智能体 |
|------|--------|--------|
| 路径 | 预定义、静态 | 动态、自主决策 |
| 工具调用 | 确定（流程固定） | 不确定（按需选择） |
| 可预测性 | 高 | 低 |
| 复杂度 | 低 | 高 |
| 适用 | 步骤可预测的任务 | 需要灵活调整的任务 |

### 2.3 选择决策树

```
任务的步骤是否可预测？
├── 是 → 使用工作流
│   ├── 只有一个步骤 → 直接 API 调用
│   ├── 有分支 → 路由
│   ├── 有并行步骤 → 并行化
│   ├── 需要多个工作者 → 编排器-工作者
│   └── 需要迭代优化 → 评估器-优化器
└── 否 → 使用智能体
```

---

## 3. 从零实现

### Step 1：提示链

```python
def prompt_chaining(input_text, steps):
    """提示链——顺序执行多个步骤。"""
    result = input_text
    for step_fn in steps:
        result = step_fn(result)
    return result

# 示例：先翻译再总结
translated = lambda x: f"翻译结果: {x}"
summary = lambda x: f"摘要: {x[:50]}..."
final = prompt_chain input_text, [translated, summary])
```

### Step 2：路由

```python
def routing(query, routes):
    """路由——根据查询选择处理分支。"""
    for route_name, matcher, handler in routes:
        if matcher(query):
            return handler(query)
    return "默认回复"
```

### Step 3：并行化

```python
from concurrent.futures import ThreadPoolExecutor

def parallelize(input_text, processors):
    """并行化——多个处理器同时处理。"""
    with ThreadPoolExecutor(max_workers=len(processors)) as executor:
        futures = [executor.submit(p, input_text) for p in processors]
    return {f"result_{i}": f.result() for i, f in enumerate(futures)}
```

### Step 4：评估器-优化器

```python
def evaluator_optimizer(input_text, generate_fn, eval_fn, max_iterations=3):
    """评估器-优化器模式。"""
    best_output = generate_fn(input_text)
    best_score = eval_fn(input_text, best_output)

    for _ in range(max_iterations):
        candidate = generate_fn(input_text)
        score = eval_fn(input_text, candidate)
        if score > best_score:
            best_output = candidate
            best_score = score

    return best_output
```

---

## 4. 工具

### 4.1 LangGraph

```python
from langgraph.graph import StateGraph, END
# 每种工作流模式都可以用 LangGraph 的状态图实现
```

### 4.2 框架对比

| 框架 | 工作流支持 | 智能体支持 |
|------|-----------|-----------|
| LangGraph | ✅ 原生 | ✅ |
| LangChain | ✅ 链式 | ✅ |
| Dify | ✅ 可视化 | ✅ |

---

## 5. 工程最佳实践

### 5.1 从简单开始

```
直接 API 调用 → 失败率 > 10%？
  → 加路由（分类请求）
    → 仍然不足？
      → 加并行化（多处理分支）
        → 还需要更多？
          → 加编排器（多工作者）
            → 还需要迭代？
              → 加评估器-优化器
                → 步骤仍不可预测？
                  → 切换到智能体
```

### 5.2 踩坑经验

- **过度工程化**：简单任务不需要工作流——直接 API 调用
- **选择错误模式**：并行化任务用提示链——浪费了并行能力
- **智能体过早引入**：步骤可预测的任务用智能体——增加了不必要的复杂性

---

## 6. 常见错误

### 错误 1：所有任务都用智能体

**现象：** 简单任务被过度工程化——增加了延迟和成本。

**修复：** 从直接 API 调用开始——只有步骤不可预测时才升级到智能体。

### 错误 2：忽略工作流模式的前置条件

**现象：** 评估器-优化器模式需要可评估的输出——但任务输出不可评估。

**修复：** 评估需要明确的评估标准（如代码测试通过、用户偏好）。

---

## 7. 面试考点

### Q1：Anthropic 的五种工作流模式是什么？什么时候应该用智能体？（难度：⭐⭐）

**参考答案：**
五种：提示链（顺序步骤）、路由（选择分支）、并行化（同时执行）、编排器-工作者（多任务调度）、评估器-优化器（迭代改进）。当步骤可预测时用工作流，当步骤不可预测、需要动态决策时用智能体。工作流更简单、更快、更便宜。智能体更灵活但更复杂。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 工作流 | "固定流程" | 预定义的 LLM 调用顺序——步骤可预测 |
| 智能体 | "自主决策" | LLM 动态选择工具和下一步行动 |
| 提示链 | "多步推理" | 顺序调用多个 LLM——前一步输出是下一步输入 |
| 路由 | "分流" | 根据查询内容选择不同的处理分支 |

---

## 📚 小结

Anthropic 的五种工作流模式覆盖 90% 的场景。从最简单的直接 API 调用开始，只在步骤不可预测时才升级到智能体。简单胜过复杂。

---

## ✏️ 练习

1. **【分析】** 分析一个客服系统——哪些步骤用工作流、哪些用智能体
2. **【实现】** 用 LangGraph 实现提示链模式

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 五种工作流模式 | `code/main.py` | 提示链+路由+并行化+评估器-优化器 |

---

## 📖 参考资料

1. [博客] Anthropic. "Building Effective Agents". 2024.
2. [GitHub] LangGraph: https://github.com/langchain-ai/langgraph
3. [文档] Anthropic 工作流指南

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
