# 群聊与说话者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体间共享一个对话；选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中保留；AutoGen v0.4 重写为事件驱动 actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 放入维护模式并与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都存活——学一次，到处用。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 04（原语模型）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 GroupChat 的形状——共享池 + 选择器函数 + 智能体轮次
- [ ] 实现三种说话者选择变体——轮询、LLM 选择、自定义——并对比它们的权衡
- [ ] 理解 AutoGen → AG2 分裂和 Microsoft Agent Framework 合并的技术背景
- [ ] 识别 GroupChat 的失败模式：谄媚级联、热说话者、上下文膨胀
- [ ] 对比 GroupChat 选择器和 Swarm 移交——谁决定下一个发言

---

## 1. 问题

静态图（LangGraph）在工作流已知时很好。真实对话不是静态的：有时编码者问审查者，有时研究员问写作者。硬编码每个可能的移交会产生边爆炸。你需要**智能体对共享池做出反应**，某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 做的——但它的 API 表面和行为在不同版本之间变化巨大，需要理解它在设计空间中的位置才能做出明智的选择。

---

## 2. 概念

### 2.1 GroupChat 的形状

```
              ┌─── 共享池 ───┐
              │   m1  m2  m3  │
              └───────┬──────┘
                      │ (每个人都读所有消息)
      ┌───────┬───────┼───────┬───────┐
      ▼       ▼       ▼       ▼       ▼
    Agent A  Agent B  Agent C  ...  选择器
                                     │
                                     ▼
                              "下一个发言者 = C"
```

每个智能体看到每条消息。选择器函数在每轮被调用来选择谁下一个发言。

### 2.2 三种选择器变体

**轮询 (Round-robin)：** 固定循环。确定性。按 N 线性扩展但忽略上下文——编码者在法律审查主题上也会发言。

**LLM 选择：** 一个 LLM 读取池并返回最佳下一个发言者。上下文感知但慢：每轮增加一次 LLM 调用。AutoGen 的默认。

**自定义：** 任意 Python 函数。典型：LLM 选择 + 回退规则（如"编码者发言后总是给审查者"）。

### 2.3 ConversableAgent 和 GroupChatManager API

```python
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成一轮时，管理者调用选择器，选择器返回下一个智能体。循环继续直到终止条件。

### 2.4 终止条件

三种常见模式：

| 模式 | 说明 | 优劣 |
|------|------|------|
| 最大轮数 | 硬上限（如 10-20 轮） | 简单但可能提前终止或浪费 |
| TERMINATE 令牌 | 智能体发出哨兵消息 | 灵活但依赖智能体判断 |
| 目标达成检查 | 轻量验证器每轮运行 | 可靠但需要实现验证器 |

### 2.5 AutoGen → AG2 分裂和 Microsoft Agent Framework 合并

2025 年初 Microsoft 开始重写 AutoGen（v0.4）为事件驱动 actor 模型。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2。2026 年 2 月 Microsoft 宣布 AutoGen 进入维护模式，事件驱动模型合并到 Microsoft Agent Framework（RC 2026 年 2 月）。

GroupChat 概念在两个轨道中都存活；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### 2.6 GroupChat 何时合适/失败

| 场景 | 合适性 | 原因 |
|------|--------|------|
| 涌现式对话 | ✅ | 不需要预连线每个下一个发言者 |
| 角色混合任务 | ✅ | 编码者问审查者，审查者问档案员 |
| 探索性问题求解 | ✅ | "头脑风暴会议"，不是"装配线" |
| 严格确定性 | ❌ | LLM 选择器不一致 |
| 谄媚级联 | ❌ | 智能体顺从最自信的发言者 |
| 上下文膨胀 | ❌ | 10 轮后每个智能体看到每条消息 |

---

## 3. 从零实现

### 第 1 步：定义智能体和策略

```python
def coder_policy(pool):
    recent = [m for m in pool[-3:] if m.speaker != "coder"]
    last = recent[-1].content if recent else ""
    if "review" in last.lower() or "fix" in last.lower():
        return "revised code: return a + b"
    if not any(m.speaker == "coder" for m in pool):
        return "initial code: return a - b  (buggy)"
    return "TERMINATE"

def reviewer_policy(pool):
    last_coder = next((m for m in reversed(pool) if m.speaker == "coder"), None)
    if last_coder is None:
        return "waiting for code"
    if "a - b" in last_coder.content:
        return "review: bug detected -- sum must be a+b, please fix"
    if "a + b" in last_coder.content:
        return "review: approved"
    return "review: unclear"

def manager_policy(pool):
    approvals = [m for m in pool if m.speaker == "reviewer" and "approved" in m.content]
    return "TERMINATE" if approvals else "manager: continue working"
```

### 第 2 步：实现三种选择器

```python
def round_robin_selector(pool, team):
    """轮询选择器——固定循环，确定性。"""
    names = list(team.keys())
    if not pool:
        return names[0]
    idx = (names.index(pool[-1].speaker) + 1) % len(names)
    return names[idx]

def llm_style_selector(pool, team):
    """LLM 选择器——基于上下文。"""
    if not pool:
        return "manager"
    last = pool[-1]
    if last.speaker == "coder":
        return "reviewer"
    if last.speaker == "reviewer":
        return "manager" if "approved" in last.content else "coder"
    if last.speaker == "manager":
        return "coder"
    return None
```

### 第 3 步：实现 GroupChat 运行循环

```python
def run_groupchat(team, selector, max_rounds, label):
    """运行群聊：共享池 + 选择器 + 智能体轮次。"""
    pool = []
    trace = []
    for _ in range(max_rounds):
        nxt = selector(pool, team)
        if nxt is None:
            break
        trace.append(nxt)
        agent = team[nxt]
        content = agent.policy(pool)
        pool.append(Msg(speaker=nxt, content=content))
        if content.strip().endswith("TERMINATE"):
            break
    print(f"  选择器追踪: {trace}")
    print(f"  使用轮数: {len(pool)}")
    return pool
```

### 第 4 步：运行对比

```python
def main():
    p_rr = run_groupchat(AGENTS, round_robin_selector, max_rounds=8, label="轮询")
    print(f"  发言次数: {speaker_counts(p_rr)}")

    p_llm = run_groupchat(AGENTS, llm_style_selector, max_rounds=8, label="LLM 选择")
    print(f"  发言次数: {speaker_counts(p_llm)}")

    print("\n观察:")
    print("  - 轮询给每个智能体平等轮次，不管上下文。")
    print("  - LLM 选择按上下文路由；审查者只在编码者之后发言。")
    print("  - 两者在 TERMINATE 令牌或最大轮数时终止。")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 选择器对比

| 选择器 | 机制 | 上下文感知 | 确定性 | 成本 |
|--------|------|----------|--------|------|
| 轮询 | 固定循环 | 否 | 是 | 零 |
| LLM 选择 | LLM 读取池 | 是 | 否 | 每轮 LLM 调用 |
| 自定义 | 任意逻辑 | 取决于实现 | 取决于实现 | 取决于实现 |

### 4.2 GroupChat vs Supervisor

| 维度 | GroupChat | Supervisor |
|------|-----------|-----------|
| 谁决定发言 | 选择器（从外部） | 领智能体（从内部） |
| 智能体角色 | 都是对等的 | 有明确的领-工层次 |
| 共享状态 | 全池（所有消息） | 投影或领持有 |
| 适用场景 | 涌现式对话、头脑风暴 | 研究、分析、需要全局规划的任务 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 最大轮数上限 | 必须。10-20 轮对典型任务足够 |
| 说话者平衡度量 | 追踪每个智能体的轮次；超过阈值时告警 |
| TERMINATE 令牌 | 清晰的终止条件 |
| 投影或限定内存 | ~10 条消息后，给每个智能体限定视图防止上下文膨胀 |
| 选择器日志 | 记录 LLM 选择器的输入和选择——用于调试和审计 |

### 5.1 中文场景特别建议

- **LLM 选择器在中文中可能更不一致**——中文的多义词和语境依赖更重
- **轮询选择器在中文团队中更可预测**——确定性对中文审查流程很重要
- **说话者平衡在中文对话中更重要**——中文礼貌文化可能加剧谄媚级联

---

## 6. 常见错误

### 错误 1：谄媚级联

**现象：** LLM 选择器总是选择最自信的智能体。对话退化为最大声者独白。

**修复：** 在选择器中添加说话者平衡规则——"限制每个智能体最多 N 次发言"。

### 错误 2：上下文膨胀

**现象：** 10 轮后每个智能体看到每条消息，上下文巨大，智能体开始重复或迷失。

**修复：** 使用投影——每个智能体只看到角色相关的消息子集。

### 错误 3：选择器不一致

**现象：** LLM 选择器同一输入不同运行选择不同下一个发言者。调试变得困难。

**修复：** 选择器日志——记录选择器的输入和选择。用于事后分析。

---

## 7. 面试考点

### Q1：GroupChat 的形状是什么？（难度：⭐）

**参考答案：**
共享消息池 + 选择器函数 + 智能体轮次。每个智能体看到每条消息。选择器在每轮被调用来选择谁下一个发言。

### Q2：三种选择器的权衡是什么？（难度：⭐⭐）

**参考答案：**
轮询：确定性、简单、零成本，但忽略上下文。
LLM 选择：上下文感知，但每轮增加一次 LLM 调用且不确定。
自定义：完全灵活，但需要自己实现。

### Q3：GroupChat 何时失败？（难度：⭐⭐）

**参考答案：**
严格确定性（LLM 选择器不一致）、谄媚级联（选择器总是选择最自信的）、上下文膨胀（10 轮后上下文巨大）、热说话者（一个智能体主导对话）。

### Q4：AutoGen → AG2 → Microsoft Agent Framework 的关系？（难度：⭐⭐⭐）

**参考答案：**
2025 年 Microsoft 重写 AutoGen v0.4 为事件驱动 actor 模型。社区分叉 v0.2 的 GroupChat 语义为 AG2。2026 年 2 月 Microsoft 将 AutoGen 放入维护模式，合并到 Agent Framework。GroupChat 概念在两个轨道中都存活——实现细节不同。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GroupChat | "智能体群聊" | 共享消息池 + 选择器函数 |
| 说话者选择 | "谁下一个发言" | 选择下一个发言者的函数 |
| GroupChatManager | "会议主持人" | 持有选择器并循环轮次的 AutoGen 组件 |
| ConversableAgent | "基础智能体" | AutoGen 基类；可发送和接收消息的智能体 |
| TERMINATE | "停止词" | 结束聊天的哨兵字符串 |
| 热说话者 | "一个智能体主导" | 选择器总是选择同一个智能体的失败模式 |
| 上下文膨胀 | "池无界增长" | 每个智能体读每条先前消息 |

---

## 📚 小结

GroupChat：共享消息池 + 选择器函数 + 智能体轮次。三种选择器：轮询（确定性）、LLM 选择（上下文感知）、自定义。AutoGen → AG2 分裂 + Microsoft Agent Framework 合并，GroupChat 原语在两个轨道中存活。失败模式：谄媚级联、热说话者、上下文膨胀。选择器日志对调试至关重要。

下一课：移交与例程——无状态编排。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。对比轮询和 LLM 选择下的对话。哪个智能体在每种模式下主导？

2. **【实现】** 在选择器中添加"每个智能体最多发言 N 次"规则。观察对话如何变化。

3. **【实现】** 实现目标达成终止：审查者返回"approved"时停止。它在轮数上限前多久触发？

4. **【阅读】** 阅读 AutoGen 稳定文档中的 GroupChat。识别 `GroupChatManager` 的默认选择器。

5. **【阅读】** 阅读 AG2 仓库。对比其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 增加了什么具体属性？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| GroupChat 演示 | `code/main.py` | 3 智能体 + 轮询/LLM 选择器 + TERMINATE |
| 技能提示词 | `outputs/skill-groupchat-selector.md` | 为新任务配置 GroupChat 选择器 |

---

## 📖 参考资料

1. [文档] AutoGen GroupChat. https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html
2. [GitHub] AG2. https://github.com/ag2ai/ag2 — 社区 AutoGen v0.2 延续
3. [文档] Microsoft Agent Framework. https://microsoft.github.io/agent-framework/ — 合并后继，RC 2026 年 2 月
4. [文档] AutoGen v0.4 Release Notes. https://microsoft.github.io/autogen/stable/ — 事件驱动 actor 模型重写详情

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
