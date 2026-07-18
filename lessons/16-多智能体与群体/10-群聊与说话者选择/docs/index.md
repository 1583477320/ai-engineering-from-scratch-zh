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

---

## 1. 问题

静态图（LangGraph）在工作流已知时很好。真实对话不是静态的：有时编码者问审查者，有时研究员问写作者。硬编码每个可能的移交会产生边爆炸。你需要**智能体对共享池做出反应**，某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 做的。

---

## 2. 概念

### 2.1 形状

```
              ┌─── 共享池 ───┐
              │   m1  m2  m3  │
              └───────┬──────┘
                      │ (每个人都读所有)
      ┌───────┬───────┼───────┬───────┐
      ▼       ▼       ▼       ▼       ▼
    Agent A  Agent B  Agent C  ...  选择器
                                     │
                                     ▼
                              "下一个发言者 = C"
```

每个智能体看到每条消息。选择器函数在每轮被调用来选择谁下一个发言。

### 2.2 三种选择器变体

| 变体 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| 轮询 (Round-robin) | 固定循环 | 确定性、简单 | 忽略上下文 |
| LLM 选择 | LLM 读取池并选择 | 上下文感知 | 每轮增加 LLM 调用 |
| 自定义 | 任意逻辑 | 完全灵活 | 需要自己实现 |

### 2.3 ConversableAgent API

```python
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

### 2.4 终止条件

三种常见模式：
- **最大轮数**——硬上限
- **TERMINATE 令牌**——智能体发出哨兵消息
- **目标达成检查**——轻量验证器每轮运行

### 2.5 AutoGen → AG2 分裂和 Microsoft Agent Framework 合并

2025 年初 Microsoft 开始重写 AutoGen（v0.4）为事件驱动 actor 模型。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2。2026 年 2 月 Microsoft 宣布 AutoGen 进入维护模式，事件驱动模型合并到 Microsoft Agent Framework（RC 2026 年 2 月）。

GroupChat 概念在两个轨道中都存活；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### 2.6 GroupChat 何时合适/失败

**合适：** 涌现式对话、角色混合任务、探索性问题求解

**失败：** 严格确定性（LLM 选择器不一致）、谄媚级联、上下文膨胀（每个智能体读每条消息）、热说话者

---

## 3. 从零实现

### 第 1 步：定义智能体和选择器

```python
def round_robin_selector(pool, team):
    """轮询选择器——固定循环。"""
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
        if "approved" in last.content:
            return "manager"
        return "coder"
    if last.speaker == "manager":
        return "coder"
    return None
```

### 第 2 步：实现 GroupChat 运行循环

```python
def run_groupchat(team, selector, max_rounds, label):
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
        print(f"  [{nxt:8s}]: {content}")
        if content.strip().endswith("TERMINATE"):
            break
    print(f"  选择器追踪: {trace}")
    print(f"  使用轮数: {len(pool)}")
    return pool
```

### 第 3 步：运行对比

```python
def main():
    # 轮询
    p_rr = run_groupchat(AGENTS, round_robin_selector, max_rounds=8, label="轮询")
    print(f"  发言次数: {speaker_counts(p_rr)}")

    # LLM 选择
    p_llm = run_groupchat(AGENTS, llm_style_selector, max_rounds=8, label="LLM 选择")
    print(f"  发言次数: {speaker_counts(p_llm)}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 选择器对比

| 选择器 | 机制 | 上下文感知 | 成本 |
|--------|------|----------|------|
| 轮询 | 固定循环 | 否 | 零 |
| LLM 选择 | LLM 读取池 | 是 | 每轮一次 LLM 调用 |
| 自定义 | 任意逻辑 | 取决于实现 | 取决于实现 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 最大轮数上限 | 必须。10-20 轮对典型任务足够 |
| 说话者平衡度量 | 追踪每个智能体的轮次；超过阈值时告警 |
| TERMINATE 令牌 | 终止条件 |
| 投影或限定内存 | ~10 条消息后，给每个智能体限定视图 |
| 选择器日志 | 记录选择器的输入和选择 |

---

## 6. 常见错误

### 错误 1：谄媚级联

**现象：** 选择器总是选择最自信的智能体。

**修复：** 在选择器中添加说话者平衡规则——"限制每个智能体最多 N 次发言"。

### 错误 2：上下文膨胀

**现象：** 10 轮后每个智能体看到每条消息，上下文巨大。

**修复：** 使用投影——每个智能体只看到角色相关的消息子集。

### 错误 3：选择器不一致

**现象：** LLM 选择器同一输入不同运行选择不同下一个发言者。

**修复：** 选择器日志——记录输入和选择。用于调试和审计。

---

## 7. 面试考点

### Q1：GroupChat 的形状是什么？（难度：⭐）

**参考答案：**
共享消息池 + 选择器函数 + 智能体轮次。每个智能体看到每条消息。选择器在每轮被调用来选择谁下一个发言。

### Q2：三种选择器的权衡是什么？（难度：⭐⭐）

**参考答案：**
轮询：确定性、简单、零成本，但忽略上下文。
LLM 选择：上下文感知，但每轮增加一次 LLM 调用。
自定义：完全灵活，但需要自己实现。

### Q3：AutoGen → AG2 分裂和 Microsoft Agent Framework 合并是什么？（难度：⭐⭐⭐）

**参考答案：**
2025 年 Microsoft 重写 AutoGen v0.4 为事件驱动 actor 模型。社区分叉 v0.2 的 GroupChat 语义为 AG2。2026 年 2 月 Microsoft 将 AutoGen 放入维护模式，合并到 Agent Framework。GroupChat 概念在两个轨道中都存活。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GroupChat | "智能体群聊" | 共享消息池 + 选择器函数 |
| 说话者选择 | "谁下一个发言" | 选择下一个发言者的函数 |
| GroupChatManager | "会议主持人" | 持有选择器并循环轮次的 AutoGen 组件 |
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
3. [文档] Microsoft Agent Framework. https://microsoft.github.io/agent-framework/ — 合并后继

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
