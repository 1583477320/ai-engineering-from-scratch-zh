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

**2026 年 LLM 时代为什么仍然重要：** 合规敏感域（银行、医疗、航空）要求确定性槽位值——不是自由形式生成。工具调用智能体在调用 API 前仍需槽位解析。多轮修正"算了，改成周四吧"——比看起来更难。

---

## 2. 现代流水线：经典 DST 概念 + LLM 提取 + Guardrails

```python
# LLM-based DST：给定对话历史 + 当前状态 → 返回更新后的 JSON
prompt = f"""Current state: {state}
Latest turn: "{user_message}"
Return updated state as JSON. For modified slots, include source phrase."""

# 输出示例：
# {"area": "north", "price": "moderate"}  ← price 被 UPDATE
```

**三种槽位操作：** ADD（新增槽位）、UPDATE（覆盖已有值）、DELETE（"不需要了"——清除）。

---

## 3. 中文 DST 特别挑战

- **口语化价格/时间表达。** "便宜点的"→ cheap、"差不多就行"→ moderate、"贵一点的"→ expensive——需要宽泛的语义映射而非精确关键词匹配
- **区分"确认"和"不关心"。** "中等价位行吗？"→"行"=确认 UPDATE。"意大利菜可以吗？"→"都可以"=不指定（不 UPDATE）。这是中文对话中 DST 最大的难点
- **LLM-based DST 的 guardrails。** 破坏性槽位（金额/日期/证件号）的 UPDATE 必须经过二次确认——即使 LLM 声称用户确认了

---

## 4. 陷阱

- **LLM 过度自信更新。** "我想取消订单"——LLM 直接 UPDATE 所有槽位为空。实际上用户可能只想取消而非清空全部状态。**破坏性操作必须经过结构化确认**
- **槽位值漂移。** 第 3 轮对话 LLM 回读的 price 值与第 1 轮不同——"中等"被误记为"中档"。**在关键槽位上做值匹配验证**

---

## 🔑 关键术语 | 📚 小结

DST = ADD/UPDATE/DELETE 三种操作在对话轮次间维护槽位-值字典。现代流水线 = 经典 DST 概念 + LLM 提取 + 结构化 guardrails。中文 DST 的额外挑战：口语化值映射和"确认"vs"不关心"的区分。**破坏性槽位永远不单靠 LLM——结构化确认是硬性要求。**

---

## ✏️ 练习

1. 【理解】手写 5 轮订餐对话。追踪每轮的槽位状态变化。标注 ADD/UPDATE/DELETE。
2. 【实现】用 LLM prompt 从对话中提取当前状态。对比 20 轮对话上 LLM 输出和人工标注的槽位一致性。
3. 【实验】加入破坏性操作 guardrail——当金额/日期被 UPDATE 时强制二次确认——衡量假正面（误确认）和假负面（阻断了合理的修改）率。

---

## 📖 参考资料

1. [论文] Mrkšić et al. "Neural Belief Tracker". ACL, 2017.
2. [调研] Balaraman et al. "Recent Advances in DST". 2021.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。中文 DST 挑战为原创内容。
