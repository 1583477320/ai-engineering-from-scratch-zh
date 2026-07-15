# 智能体框架选型

> 2026 年的 AI Agent 框架多如牛毛——LangGraph、CrewAI、AutoGen、Dify、Coze。选错框架，三个月后被迫重写。选对框架，事半功倍。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 11 · 01-16 | **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 对比主流 Agent 框架的架构差异和适用场景
- [ ] 根据项目需求选择最合适的 Agent 框架
- [ ] 理解自托管 vs 云托管方案的权衡

---

## 1. 问题

你决定构建一个 AI Agent。打开 GitHub 搜索 "AI agent framework"——LangGraph、CrewAI、AutoGen、Dify、Coze、MetaGPT、OpenHands... 每个都声称"最好"。但没有一个框架适合所有场景。

---

## 2. 概念

### 2.1 主流框架对比

| 框架 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| **LangGraph** | 编排层 | 有状态图、可调试 | 复杂多步流水线 |
| **CrewAI** | 多智能体 | 角色扮演、团队协作 | 多 Agent 协作任务 |
| **Dify** | 低代码平台 | 可视化编排、无需代码 | 快速原型、非工程师 |
| **Coze** | 扣子/字节 | 插件生态、中文支持 | 中文 Agent |
| **AutoGen** | 多智能体 | 对话式协作 | 研究探索 |
| **MetaGPT** | 多智能体 | SOP 驱动 | 软件工程任务 |

### 2.2 选型决策矩阵

| 需求 | 推荐 | 原因 |
|------|------|------|
| 复杂多步流水线 | LangGraph | 有状态图最灵活 |
| 多 Agent 协作 | CrewAI / AutoGen | 原生支持多 Agent |
| 快速原型 | Dify | 拖拽式，无代码 |
| 中文优先 | Coze / Dify | 中文生态好 |
| 生产部署 | LangGraph + LangServe | 可观测性强 |
| 研究探索 | AutoGen | 对话式 Agent |

### 2.3 架构选择原则

- **简单任务**：直接用 API + 函数调用，不需要框架
- **多步任务**：LangGraph（有状态图最清晰）
- **多 Agent**：CrewAI（角色分工简单）或 AutoGen（更灵活）
- **快速原型**：Dify（可视化，无代码）
- **企业级**：LangGraph + LangSmith（监控+调试）

---

## 3. 实现建议

### 3.1 评估清单

选择框架前问自己：
1. **任务复杂度**：单步？多步？多 Agent？
2. **开发团队**：工程师？非技术？
3. **部署环境**：本地？云？混合？
4. **可控性要求**：需要调试每个步骤？还是黑盒可接受？
5. **预算**：开源免费？付费平台？

### 3.2 中文生态

| 平台 | 中文支持 | 特点 |
|------|---------|------|
| Coze（扣子） | 原生中文 | 字节跳动，中文 Agent 平台 |
| Dify | 中文界面 | 开源低代码，社区活跃 |
| 百川 | 中文优先 | 模型+Agent 一体化 |

---

## 7. 面试考点

### Q1：LangGraph 和 CrewAI 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
LangGraph 是一个**状态图编排层**——将 LLM 应用建模为有向图，节点是 LLM 调用/工具，边是有条件转移。适合需要精确控制执行顺序和状态管理的复杂流水线。CrewAI 是一个**多 Agent 协作框架**——预定义了"角色"（研究员、编写者、评论者），适合需要多个 Agent 协作的任务。LangGraph 更灵活但需要更多设计工作；CrewAI 开箱即用但灵活性较低。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agent 框架 | "AI 智能体工具" | 提供多步推理、工具调用、状态管理能力的软件框架 |
| LangGraph | "有状态 Agent" | 将 LLM 应用建模为状态图的编排框架 |
| CrewAI | "多 Agent 团队" | 多个 Agent 分工协作的框架——适合团队任务 |
| 低代码 Agent | "拖拽 Agent" | 可视化编排界面——非工程师也可以搭建 Agent |

---

## 📚 小结

2026 年主流 Agent 框架：LangGraph（复杂流水线）、CrewAI（多 Agent 协作）、Dify（低代码）、Coze（中文）、AutoGen（研究）。选型看任务复杂度、团队能力、可控性要求。简单任务不需要框架——直接用 API + 函数调用。

---

## ✏️ 练习

1. **【实验】** 在 Dify 上搭建一个简单的客服 Agent——上传文档、配置对话流程
2. **【思考】** 如果你需要构建一个支持 100 万用户的 AI 客服系统，你会选什么架构？为什么？

---

## 📖 参考资料

1. [GitHub] LangGraph: https://github.com/langchain-ai/langgraph
2. [GitHub] CrewAI: https://github.com/joaomdmoura/crewAI
3. [GitHub] AutoGen: https://github.com/microsoft/autogen
4. [平台] Dify: https://dify.ai/
5. [平台] Coze: https://www.coze.com/
