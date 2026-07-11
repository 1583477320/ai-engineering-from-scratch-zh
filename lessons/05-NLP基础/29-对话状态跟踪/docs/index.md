# 对话状态跟踪

> "我想要北区便宜点的餐馆…算了改成中等价位…再加意大利菜。"三轮对话、三次状态更新。DST 保持槽位-值字典的同步——让预订操作在执行时拿到正确的参数。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 17（聊天机器人）、05 · 20（结构化输出） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 理解 DST 的三种槽位操作——ADD/UPDATE/DELETE——从多轮对话中维护状态字典
- [ ] 实现 LLM-based DST：对话历史 + 当前状态 → 更新后的 JSON
- [ ] 解释为什么 DST 在 2026 LLM 时代仍然重要——合规域需要确定性槽位值

---

## 1. 问题

任务导向对话中，用户目标被编码为槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每轮对话可能 ADD/UPDATE/DELETE 一个槽位：

```
Turn 1: "北区便宜点的餐馆" → {area: north, price: cheap}
Turn 2: "改成中等价位"      → {area: north, price: moderate}  ← UPDATE
Turn 3: "再加意大利菜"       → {area: north, price: moderate, cuisine: italian}  ← ADD
Turn 4: "算了不要意大利菜了"  → {area: north, price: moderate}  ← DELETE
```

搞错一个槽位 → 订错餐厅、安排错航班、扣错款。DST 是用户说了什么和后端执行了什么之间的铰链。

**2026 年 LLM 时代为什么仍然重要：** 合规敏感域（银行/医疗/航空）要求确定性槽位值——不是自由形式生成。工具调用智能体在调用 API 前仍需槽位解析。多轮修正"算了，改成周四吧"——比看起来更难。

---

## 2. 概念

### 2.1 槽位-值对 + 三种操作

| 操作 | 含义 | 示例 |
|---|---|---|
| ADD | 新增未设置过的槽位 | "加意大利菜" → `cuisine: italian`（之前为空） |
| UPDATE | 覆盖已有槽位 | "改成中等价位" → `price: cheap` 变为 `price: moderate` |
| DELETE | 清除槽位 | "不要意大利菜了" → 移除 `cuisine` |

### 2.2 现代流水线：经典 DST + LLM + Guardrails

```
对话历史 + 当前状态 → [LLM 提取] → 更新后的状态 JSON → [Guardrails 验证]
```

**三个组件各司其职。** LLM 负责理解自然语言中的状态变更意图。Guardrails 负责验证破坏性槽位（金额/日期/证件号）的变更是否经过了确认流程。经典 DST 概念（槽位 Schema、本体约束）确保输出的 JSON 是合法的。

---

## 3. 从零实现

### LLM-based DST

```python
def update_state(conversation_history, current_state, llm):
    prompt = f"""Current state: {json.dumps(current_state)}
Latest user turn: "{user_message}"

Return the updated state as JSON. For any modified slot, include:
- the new value
- the source phrase from the user's message
- the operation: "ADD", "UPDATE", or "DELETE"
"""
    response = llm(prompt, response_format={"type": "json_object"})
    new_state = json.loads(response)
    # Guardrail: 破坏性槽位必须经过二次确认
    if destructive_slot_modified(current_state, new_state):
        return require_confirmation(new_state)
    return new_state
```

**破坏性槽位（金额/日期/证件号/航班号）的 UPDATE 必须经过二次确认。** 即使 LLM 声称用户确认了——在结构化确认流程中再问一遍。

---

## 4. 中文 DST 特别挑战

- **口语化值映射。** "便宜点的"→ cheap、"差不多就行"→ moderate、"贵一点的"→ expensive——需要宽泛语义映射而非精确关键词
- **区分"确认"和"不关心"。** "中等价位行吗？"→"行"=确认 UPDATE。"意大利菜可以吗？"→"都可以"=不指定（不 UPDATE）。这是中文对话 DST 最大的难点
- **LLM-based DST 的 Guardrails。** 破坏性槽位（金额/日期/证件号）的变更必须经二次确认——即使 LLM 声称用户确认了

---

## 5. 陷阱

- **LLM 过度自信更新。** "我想取消订单"→ LLM 直接 UPDATE 所有槽位为空。实际上用户可能只想取消而非清空全部状态
- **槽位值漂移。** 第 3 轮 LLM 回读的 price 值与第 1 轮不同——"中等"被误记为"中档"
- **历史窗口截断。** 长对话中最早的槽位设置被推出上下文窗口——LLM 丢失了原始值

---

## 🔑 关键术语 | 📚 小结

DST = ADD/UPDATE/DELETE 三种操作在对话轮次间维护槽位-值字典。现代流水线 = LLM 提取 + Guardrails 验证。中文 DST 两大额外挑战：口语化值映射、"确认"vs"不关心"的区分。**破坏性槽位永远不单靠 LLM——二次确认是硬性要求。**

---

## ✏️ 练习

1. 【理解】手写 5 轮订餐对话。追踪每轮槽位状态变化。标注 ADD/UPDATE/DELETE。
2. 【实现】用 LLM prompt 从对话中提取状态。在 20 轮对话上对比 LLM 输出和人工标注的一致性。
3. 【实验】加入破坏性操作 guardrail——金额/日期被 UPDATE 时强制二次确认。衡量假正面/假负面率。

---

## 📖 参考资料

1. [论文] Mrkšić et al. "Neural Belief Tracker: Data-Driven Dialogue State Tracking". ACL, 2017.
2. [调研] Balaraman et al. "Recent Advances in DST". 2021.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。中文 DST 挑战分析为原创内容。
