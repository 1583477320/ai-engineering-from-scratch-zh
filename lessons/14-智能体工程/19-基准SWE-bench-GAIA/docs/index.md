# 基准测试：SWE-bench、GAIA、AgentBench

> 2026 年有三个基准锚定了智能体评估。SWE-bench 测试代码补丁。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、污染故事和它们不衡量什么。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 06（工具使用）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 SWE-bench 的测试工具——FAIL_TO_PASS 并解释为什么以单元测试为准
- [ ] 对比 SWE-bench、GAIA、AgentBench 的评估维度
- [ ] 理解基准测试的局限性——它们不衡量什么
- [ ] 设计一个适合自己项目的评估方案

---

## 1. 问题

"你的智能体有多好？" —— 需要量化评估。但智能体评估比传统 ML 复杂——它涉及代码执行、多步推理、工具使用。三个基准在 2026 年成为标准。

---

## 2. 概念

### 2.1 三大基准对比

| 基准 | 评估什么 | 难度 | 关键机制 |
|------|---------|------|---------|
| **SWE-bench** | 代码补丁 | 高 | 用单元测试验证补丁 |
| **GAIA** | 通用工具使用 | 高 | 多工具、多步推理 |
| **AgentBench** | 多环境推理 | 中 | 跨环境任务 |

### 2.2 SWE-bench

```
输入：GitHub issue + 代码库
    ↓
智能体分析 issue、阅读代码、生成补丁
    ↓
验证：运行单元测试
FAIL_TO_PASS：之前失败的测试现在通过
```

### 2.3 GAIA

测试 3 级难度：
- Level 1：简单检索（1 步）
- Level 2：需要 2-3 步推理
- Level 3：需要多工具协作

### 2.4 基准测试的局限

| 基准 | 不衡量的 |
|------|---------|
| SWE-bench | 文档质量、可维护性、安全性 |
| GAIA | 用户交互、多轮对话 |
| AgentBench | 实时性、延迟、成本 |

---

## 3. 从零实现

### Step 1：简单评估框架

```python
def evaluate_agent(agent_fn, test_cases, metric_fn):
    """评估智能体。"""
    correct = 0
    for test in test_cases:
        result = agent_fn(test["input"])
        if metric_fn(result, test["expected"]):
            correct += 1
    return correct / len(test_cases)
```

### Step 2：准确率计算

```python
def accuracy_metric(result, expected):
    """简单匹配度量。"""
    return str(expected).lower() in str(result).lower()
```

---

## 4. 工具

### 4.1 基准评估平台

| 平台 | 内容 |
|------|------|
| SWE-bench | 代码补丁评估 |
| GAIA | 通用工具使用 |
| AgentBench | 多环境推理 |
| Chatbot Arena | LLM 对话质量 |
| BigBench | 综合能力 |

---

## 7. 面试考点

### Q1：SWE-bench 为什么用单元测试验证补丁？（难度：⭐⭐）

**参考答案：**
单元测试提供了确定性的、可重复的正确性验证——不需要人类判断。FAIL_TO_PASS 指标只计算之前失败但现在通过的测试——确保补丁确实修复了 issue，而没有引入新问题。这比人工审查更客观、更可扩展。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| SWE-bench | "代码智能体基准" | 从 GitHub issue 到代码补丁的端到端评估 |
| GAIA | "通用智能体基准" | 多工具、多步推理的通用智能体评估 |
| AgentBench | "多环境基准" | 跨多个环境（CLI、Web、数据库）的推理评估 |
| FAIL_TO_PASS | "修复成功率" | 之前失败的单元测试现在通过的比例 |

---

## 📚 小结

三大基准覆盖不同维度：SWE-bench 代码补丁、GAIA 通用工具、AgentBench 多环境推理。它们定义了 2026 年智能体能力的上限。但它们不衡量用户体验、延迟和成本。

---

## ✏️ 练习

1. **【分析】** 对比 SWE-bench 和 GAIA 的评估侧重点——为什么需要两个基准
2. **【设计】** 为一个客服智能体设计评估方案——包括哪些维度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 评估框架 | `code/main.py` | 智能体评估 + 准确率计算 |

---

## 📖 参考资料

1. [论文] Jimenez et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" 2024.
2. [论文] Mialon et al. "GAIA: A Benchmark for General AI Assistants". 2023.
3. [论文] Liu et al. "AgentBench: Evaluating LLMs as Agents". 2023.
