# 基准测试：WebArena 和 OSWorld

> WebArena 测试 Web 智能体在四个自托管应用上的能力。OSWorld 测试桌面智能体在 Ubuntu、Windows、macOS 上的能力。发布时（2023-2024）两者都显示最佳智能体与人类之间存在巨大差距。差距在缩小；失败模式没有改变。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 19（SWE-bench、GAIA）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 WebArena 的四个自托管应用以及为什么基于执行的评估很重要
- [ ] 对比 WebArena 和 OSWorld 的评估方法和失败模式
- [ ] 理解为什么 GUI 操作比 API 调用更难——需要视觉定位
- [ ] 设计一个 Web 智能体的评估方案

---

## 1. 问题

WebArena 测试智能体操作 Web 应用的能力。OSWorld 测试智能体操作桌面应用的能力。两者都需要：屏幕理解、GUI 元素定位、多步操作。2024 年发布时，最强模型只有 ~30% 准确率。

---

## 2. 概念

### 2.1 WebArena

| 应用 | 评估能力 |
|------|---------|
| Reddit 论坛 | 帖子创建、评论、搜索 |
| GitLab | 代码仓库管理、Issue 操作 |
| 在线商店 | 商品浏览、购物车、结账 |
| CMS | 内容创建、编辑、发布 |

### 2.2 OSWorld

| 环境 | 评估能力 |
|------|---------|
| Ubuntu | 终端命令、文件操作 |
| Windows | GUI 操作、应用程序使用 |
| macOS | Spotlight 搜索、Finder 操作 |

### 2.3 两者的共同挑战

| 挑战 | 描述 |
|------|------|
| **视觉定位** | 在截图中找到正确的按钮/链接 |
| **状态跟踪** | 网页状态变化后的感知 |
| **错误恢复** | 操作失败后的回退策略 |
| **多步骤规划** | 完成复杂任务的步骤分解 |

---

## 3. 从零实现

### Step 1：简单 Web 智能体评估

```python
def evaluate_web_agent(agent_fn, test_cases, metric_fn):
    """评估 Web 智能体。"""
    results = []
    for test in test_cases:
        try:
            result = agent_fn(test["task"], test["initial_state"])
            passed = metric_fn(result, test["expected_state"])
            results.append({"task": test["task"], "passed": passed})
        except Exception as e:
            results.append({"task": test["task"], "passed": False, "error": str(e)})

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    return pass_rate, results
```

---

## 4. 工具

### 4.1 Web 智能体框架

| 框架 | 特点 |
|------|------|
| Browser Use | AI 驱动的浏览器操作 |
| Playwright | 程序化浏览器控制 |
| Selenium | 传统浏览器自动化 |

### 4.2 评估基准

| 基准 | 内容 |
|------|------|
| WebArena | Web 应用操作 |
| OSWorld | 桌面应用操作 |
| VisualWebArena | 视觉理解 + Web 操作 |

---

## 7. 面试考点

### Q1：为什么基于执行的评估比基于文本的评估更可靠？（难度：⭐⭐）

**参考答案：**
基于文本的评估（如问答）无法检测"幻觉"——模型可能生成看似合理但错误的回答。基于执行的评估通过实际执行操作来验证——如果 Web 智能体点击了错误的按钮，执行结果会显示错误。这种客观验证避免了 LLM 作为 Judge 的主观偏差。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| WebArena | "Web 操作基准" | 在四个自托管 Web 应用上测试智能体的 GUI 操作能力 |
| OSWorld | "桌面操作基准" | 在 Ubuntu/Windows/macOS 上测试智能体的桌面操作能力 |
| 视觉定位 | "找到正确元素" | 在屏幕截图中精确定位按钮、输入框等 GUI 元素 |

---

## 📚 小结

WebArena 和 OSWorld 是 2024 年发布的 GUI 操作基准。两者都显示智能体与人类有巨大差距——2024 年最强模型约 30%。基于执行的评估比文本评估更可靠——操作结果是客观的。

---

## ✏️ 练习

1. **【分析】** 对比 WebArena 和 GAIA——为什么需要两个不同的基准？
2. **【设计】** 为一个电商 Web 智能体设计评估方案——包括哪些任务

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Web 智能体评估 | `code/main.py` | Web 智能体评估框架 |

---

## 📖 参考资料

1. [论文] Zhou et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents". ICLR, 2024.
2. [论文] Xie et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments". NeurIPS, 2024.
3. [论文] Zheng et al. "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks". ICLR, 2024.
