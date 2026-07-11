# 对话状态跟踪

> "我想要北区便宜点的餐馆…算了改成中等价位…再加意大利菜。"三轮对话、三次状态更新。DST 保持槽位-值字典的同步——让预订操作在执行时拿到正确的参数。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 17（聊天机器人）、05 · 20（结构化输出） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 理解 DST 如何从多轮对话中提取和更新 slot-value 对——ADD/UPDATE/DELETE 三种操作
- [ ] 解释为什么 DST 在 LLM 时代仍然重要——合规域需要确定性槽位值，工具调用需要槽位解析

---

## 1. 问题

在任务导向对话系统中，用户目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每一轮对话都可能 ADD/UPDATE/DELETE 一个槽位。

```
Turn 1: "北区便宜点的餐馆" → {area: north, price: cheap}
Turn 2: "改成中等价位"      → {area: north, price: moderate}  ← UPDATE
Turn 3: "再加意大利菜"       → {area: north, price: moderate, cuisine: italian}  ← ADD
```

搞错一个槽位 → 订错餐厅、安排错航班、扣错款。DST 是用户说了什么和后端执行了什么之间的铰链。

**2026 年 LLM 时代为什么仍然重要：** 合规敏感域要求确定性槽位值——不是自由形式生成。工具调用智能体在调用 API 之前仍需要槽位解析。多轮修正在"算了，改成周四吧"上仍然比看起来更难。

---

## 2. 现代流水线

经典 DST 概念 + LLM 提取器 + 结构化输出 guardrails。

```python
# LLM-based DST：给对话历史 + 当前状态 → 返回更新后的状态
prompt = f"""Current state: {state}
Latest turn: "{user_message}"
Return the updated state as JSON. For any modified slot, include the source phrase."""
```

**中文特别挑战：** 区分"确认"和"不关心"——"中等价位行吗？"答"行"=确认 update。答"都可以"=不指定。"便宜点的"/"差不多就行"/"贵一点的"需要口语化范围映射。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。中文 DST 挑战为原创内容。
