# 多模态智能体与计算机使用（综合实践）

> 2026 年的前沿产品是能看屏幕截图、点击按钮、导航 Web UI、填写表单、端到端完成工作流的多模态智能体。SeeClick 和 CogAgent（2024）证明了 GUI 定位原语。Ferret-UI 增加了移动端支持。VisualWebArena 和 AgentVista（2026）是前沿追赶的基准。本综合实践汇聚了第 12 章的所有线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、定位（坐标输出）、长期记忆和评估。

**类型：** 综合实践 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、09（Qwen-VL JSON）、阶段 14（智能体工程）| **时间：** ~240 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计多模态智能体循环——感知→推理→行动→观察→重复
- [ ] 实现 GUI 定位——将自然语言指令映射到屏幕坐标
- [ ] 构建带长期记忆的多模态智能体
- [ ] 在基准上评估多模态智能体——VisualWebArena、AgentVista

---

## 1. 问题

传统 LLM 智能体只能处理文本。但现实工作流涉及 GUI 操作——打开浏览器、搜索、填写表单、提交报告。需要一个能"看"屏幕并"点击"按钮的智能体。

**多模态智能体 = VLM 看屏幕 + LLM 做决策 + 动作执行器操作 GUI。**

---

## 2. 概念

### 2.1 多模态智能体循环

```
观察（屏幕截图）→ [VLM 理解] → 推理（LLM 决策）→ 行动（GUI 操作）→ 新观察
      ↑                                                              ↓
      └──────────────────────────────────────────────────────────────┘
```

### 2.2 GUI 定位

将自然语言指令映射到屏幕坐标：

```
指令: "点击登录按钮"
屏幕截图 → [VLM 识别] → 坐标 (x=300, y=150, w=80, h=30)
```

### 2.3 关键组件

| 组件 | 功能 | 代表模型 |
|------|------|---------|
| 视觉感知 | 理解屏幕内容 | SeeClick, CogAgent |
| 推理决策 | 决定下一步行动 | GPT-4V, Claude |
| 动作执行 | 点击、输入、滚动 | Playwright, Selenium |
| 长期记忆 | 记住之前的操作 | MemGPT |

### 2.4 基准

| 基准 | 衡量什么 | 2026 前沿分数 |
|------|---------|-------------|
| VisualWebArena | Web 浏览器操作 | ~30% |
| AgentVista | 多模态代理评估 | ~30% |
| OSWorld | 操作系统任务 | ~25% |

---

## 3. 从零实现

### Step 1：感知-推理-行动循环

```python
class MultimodalAgent:
    """简化版多模态智能体。"""
    def __init__(self, vlm, llm, executor):
        self.vlm = vlm      # 视觉感知
        self.llm = llm      # 推理决策
        self.executor = executor  # 动作执行
        self.memory = []     # 长期记忆

    def step(self, observation):
        # 1. 感知
        understanding = self.vlm.understand(observation)

        # 2. 推理
        action_plan = self.llm.reason(understanding, self.memory)

        # 3. 行动
        result = self.executor.execute(action_plan)

        # 4. 记忆
        self.memory.append({
            "observation": understanding,
            "action": action_plan,
            "result": result,
        })

        return result
```

### Step 2：GUI 定位

```python
def gui_grounding(vlm, screenshot, instruction):
    """将自然语言指令映射到屏幕坐标。"""
    prompt = f"在以下屏幕截图中，'{instruction}' 对应的元素位于什么位置？返回边界框坐标。"
    coordinates = vlm.generate(screenshot, prompt)
    return {"instruction": instruction, "coordinates": coordinates}
```

### Step 3：工作流执行

```python
def execute_workflow(agent, task_description, max_steps=10):
    """执行一个多步骤工作流。"""
    agent.reset()
    for step in range(max_steps):
        observation = agent.get_observation()
        result = agent.step(observation)
        if result.get("task_complete", False):
            return {"success": True, "steps": step + 1}
    return {"success": False, "steps": max_steps}
```

---

## 4. 工具

### 4.1 视觉智能体框架

| 工具 | 功能 | 特点 |
|------|------|------|
| SeeClick | GUI 定位 | 像素级坐标输出 |
| CogAgent | 多模态推理 | 图像+文本理解 |
| Ferret-UI | 移动端 GUI | 适配手机界面 |

### 4.2 浏览器自动化

| 工具 | 功能 |
|------|------|
| Playwright | 现代浏览器自动化 |
| Selenium | 传统浏览器自动化 |
| Browser Use | AI 驱动的浏览器操作 |

---

## 6. 工程最佳实践

### 6.1 多模态智能体架构

```
1. 视觉感知（VLM）: 理解屏幕内容 → 结构化表示
2. 推理决策（LLM）: 根据目标和记忆决定行动
3. 动作执行（API）: 调用 Playwright/Selenium 操作 GUI
4. 长期记忆（RAG）: 保存历史操作和观察
5. 评估验证（Verifier）: 检查行动是否符合预期
```

### 6.2 踩坑经验

- **坐标偏移**：截图和实际页面的坐标可能不一致——需要坐标归一化
- **多标签页混淆**：智能体操作了错误的标签页——需要在状态中跟踪当前标签
- **长期记忆膨胀**：存储所有操作历史——需要定期摘要

---

## 7. 常见错误

### 错误 1：VLM 的 GUI 定位不精确

**现象：** 点击位置偏差——点击了按钮旁边的空白。

**原因：** VLM 的坐标输出精度有限——需要后处理校准。

**修复：** 使用 OCR 辅助定位——先识别按钮文字，再在截图中搜索文字位置。

### 错误 2：智能体陷入循环

**现象：** 智能体在同一个操作上反复尝试——无法完成任务。

**原因：** 缺乏长期记忆——不记得之前尝试过什么。

**修复：** 在记忆中存储历史操作——避免重复尝试相同动作。

---

## 8. 面试考点

### Q1：多模态智能体和纯文本智能体的核心区别是什么？（难度：⭐⭐）

**参考答案：**
纯文本智能体只能处理文本输入——无法理解屏幕、图像或 GUI。多模态智能体通过 VLM "看"屏幕，通过 LLM "思考"下一步操作，通过动作执行器"点击"按钮。关键差异：(1) 感知维度不同——文本 vs 图像+文本；(2) 动作空间不同——文本输出 vs GUI 操作（点击、输入、滚动）；(3) 状态表示不同——纯文本上下文 vs 屏幕截图+UI 元素。

### Q2：GUI 定位为什么比文本问答更难？（难度：⭐⭐⭐）

**参考答案：**
(1) 坐标精度要求高——按钮可能只有 80×30 像素，偏差 10 像素就会点击错误位置；(2) 视觉干扰多——网页上有大量无关元素（广告、导航栏）需要过滤；(3) 动态变化——页面加载中状态会变化，时间窗口有限；(4) 跨应用差异——不同浏览器/应用的 UI 风格不同。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 多模态智能体 | "能看屏幕的 AI" | 同时处理视觉（截图）和语言（指令）并执行 GUI 操作的智能体 |
| GUI 定位 | "点击准确位置" | 将自然语言指令映射到屏幕上的具体坐标 |
| VisualWebArena | "浏览器 AI 基准" | 测试 AI 操作 Web 浏览器完成任务的标准基准 |
| SeeClick | "视觉点击" | 能够精确定位 GUI 元素并生成坐标点击的模型 |

---

## 📚 小结

多模态智能体 = VLM 感知 + LLM 推理 + 动作执行器操作 GUI。关键组件：视觉感知（SeeClick/CogAgent）、推理决策（GPT-4V/Claude）、动作执行（Playwright）、长期记忆。2026 年前沿基准：VisualWebArena、AgentVista——即使最强模型也只有 ~30% 准确率。这是多模态 AI 的最后一块拼图。

---

## ✏️ 练习

1. **【实现】** 构建一个简单的多模态智能体——能读取截图并输出下一步操作
2. **【实验】** 用 SeeClick 或类似工具在网页上完成"搜索天气"任务——记录成功/失败次数

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 多模态智能体 | `code/main.py` | 感知→推理→行动循环 + GUI 定位 |

---

## 📖 参考资料

1. [论文] SeeClick. "SeeClick: Harnessing GUI Grounding for Advanced Visual Agent". arXiv, 2024.
2. [论文] CogAgent. "CogAgent: A Visual Language Model for GUI Agents". CVPR, 2024.
3. [论文] Zhou et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents". ICLR, 2024.
4. [GitHub] Browser Use: https://github.com/browser-use/browser-use

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
