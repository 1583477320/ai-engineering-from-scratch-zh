# 长期后台智能体：持久执行

> 生产中的长期智能体不会在 `while True` 中运行。每个 LLM 调用都变成了一个带检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月 GA。Claude Code Routines（Anthropic）运行定时调度的 Claude Code 调用，无需持久本地进程。会话在人类输入时暂停，在部署后存活，并从由 `thread_id` 键控的最新检查点恢复。在新的 UI 背后是一个老模式——工作流编排——只有一个新输入：LLM 调用作为非确定性活动，必须在恢复时确定性地重放。

**类型：** 实现课
**语言：** Python（标准库，最小持久执行状态机）
**前置知识：** 阶段 15 · 10（权限模式）、阶段 15 · 01（长期智能体）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 13（成本门控）— 持久性 + 预算终止开关配合使用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分工作流（确定性编排代码）和活动（非确定性工作单元）
- [ ] 理解事件日志 + 重放模式使长期运行在崩溃时幸存
- [ ] 从零实现 `@activity` 装饰器、事件日志和 `run_or_replay` 重放机制
- [ ] 解释为什么 LLM 调用适合作为活动——非确定性、昂贵、有副作用、可能失败
- [ ] 理解"35 分钟退化"——METR 观察到的可靠性大致随视界二次方下降
- [ ] 识别何时持久执行是错误的选择——短运行、严格只读、单上下文窗口任务

---

## 1. 问题

考虑一个运行四小时的智能体。它调用三个工具，提示用户两次，进行四十次 LLM 调用。中途，宿主机重启。会发生什么？

| 场景 | 朴素 `while True` | 持久执行 |
|------|------------------|---------|
| 已有结果 | 一切丢失 | 从最近检查点恢复 |
| 已完成的活动 | 重新执行（包括副作用） | 从日志重放，不重新执行 |
| 人类批准 | 重新提示已经批准的事 | 不重新提示 |
| LLM 调用 | 重新计费四十次 | 不重新计费 |

这是工作流引擎已经提供了十年的模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于 LLM 调用现在是一种活动——非确定性、昂贵、有副作用——它们干净地适合这个模式。

---

## 2. 概念

### 2.1 工作流、活动、重放

```
工作流（确定性编排代码）
    │
    ├── 活动 1: LLM 调用（非确定性）
    │       ├── 输入记录到事件日志
    │       ├── 执行 LLM 调用
    │       └── 输出记录到事件日志
    │
    ├── 活动 2: 工具调用（有副作用）
    │       ├── 输入记录到事件日志
    │       ├── 执行工具
    │       └── 输出记录到事件日志
    │
    └── 活动 3: 人类输入（一等状态）
            └── 等待 → 暂停 → 恢复
```

| 术语 | 定义 | 特性 |
|------|------|------|
| **工作流** | 确定性编排代码 | 定义活动序列、分支、等待。必须确定性以便重放 |
| **活动** | 非确定性的、可能失败的工作单元 | LLM 调用、工具调用、文件写入、HTTP 请求 |
| **事件日志** | 持久后端存储 | 记录每个活动开始、完成、失败、重试和每个工作流决策 |
| **重放** | 恢复过程 | 工作流代码从头重跑；已完成活动返回记录结果而不重新执行 |

### 2.2 为什么 LLM 调用适合这个模式

| 特征 | LLM 调用 | 活动的需求 |
|------|---------|----------|
| 非确定性 | 温度 > 0；温度 0 也漂移 | 活动预期非确定性 |
| 昂贵 | 金钱和延迟 | 活动预期昂贵 |
| 可能失败 | 速率限制、超时 | 活动需要重试 |
| 有副作用 | 如果调用工具 | 活动需要检查点 |

将每个 LLM 调用包装为活动提供：指数退避重试、跨重启检查点、可重放调试轨迹。

### 2.3 检查点后端选择

| 后端 | 持久性 | 可查询性 | 跨部署存活 | 适用场景 |
|------|--------|---------|----------|---------|
| PostgreSQL | 高 | 高 | 是 | 生产默认 |
| SQLite | 低 | 中 | 否（跨主机丢失） | 本地开发 |
| Redis | 中 | 中 | 取决于配置 | 快速缓存 |
| Cloudflare Durable Objects | 高 | 中 | 是 | 分布式中 |

### 2.4 35 分钟退化

METR 观察到每个被测量的智能体类别在持续运行约 35 分钟后显示可靠性下降。将任务时长加倍大致使失败率翻倍。

```
安全模式：持久执行 + 定期 HITL 检查点 + 预算终止开关
```

持久执行不修复退化。它让你运行得比可靠性曲线支持的时间更长。

### 2.5 何时持久执行是错误的选择

| 场景 | 原因 |
|------|------|
| 运行短于几分钟且无人类输入 | 开销 > 收益 |
| 严格只读的信息检索 | 不需要检查点 |
| 要求单上下文窗口正确的任务 | 检查点在此场景不帮助 |

---

## 3. 从零实现

### 第 1 步：定义事件日志

```python
import json, os
from dataclasses import dataclass

@dataclass
class EventLog:
    """持久事件日志——存储在 JSON 文件中。"""
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def events(self) -> list[dict]:
        with open(self.path) as f:
            return json.load(f)

    def append(self, ev: dict) -> None:
        evs = self.events()
        evs.append(ev)
        with open(self.path, "w") as f:
            json.dump(evs, f)

    def lookup(self, name: str, args: tuple) -> dict | None:
        """查找已完成的活动。"""
        for ev in self.events():
            if ev["name"] == name and ev["args"] == list(args) and ev["status"] == "done":
                return ev
        return None
```

### 第 2 步：实现活动装饰器

```python
import functools

def activity(name: str):
    """将函数标记为可重放的活动。

    执行前：在日志中记录"started"
    执行后：在日志中记录"done" + 结果
    重放时：在日志中查找已完成的记录，返回缓存结果
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(log: EventLog, *args):
            hit = log.lookup(name, args)
            if hit:
                print(f"    [replay] {name}({args}) -> {hit['result']} (from log)")
                return hit["result"]
            log.append({"name": name, "args": list(args), "status": "started"})
            result = fn(*args)
            log.append({"name": name, "args": list(args),
                        "status": "done", "result": result})
            print(f"    [run]    {name}({args}) -> {result}")
            return result
        return wrapper
    return deco
```

### 第 3 步：定义活动和工作流

```python
@activity("fetch_docs")
def fetch_docs(query: str) -> int:
    """示例活动 1：模拟 API 调用。"""
    return len(query) * 3

@activity("call_llm")
def call_llm(doc_count: int) -> str:
    """示例活动 2：模拟 LLM 调用。"""
    return f"summary({doc_count}_docs)"

@activity("write_report")
def write_report(summary: str) -> str:
    """示例活动 3：模拟有副作用的工具调用。"""
    return f"report://{summary}"

def workflow(log: EventLog, query: str, crash_after: int = -1) -> str:
    """三步工作流，带可选的崩溃演示。"""
    doc_count = fetch_docs(log, query)
    if crash_after == 1:
        raise RuntimeError("simulated crash after fetch_docs")
    summary = call_llm(log, doc_count)
    if crash_after == 2:
        raise RuntimeError("simulated crash after call_llm")
    report = write_report(log, summary)
    return report
```

### 第 4 步：运行朴素重试 vs 持久重放对比

```python
import tempfile

def main():
    # 朴素重试：没事件日志——每次从头
    print("朴素重试（事件日志未持久化）")
    for attempt in range(1, 4):
        log = reset_log(...)  # 每次重置日志
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            break
        except RuntimeError:
            print(f"    -> 崩溃; 活动被浪费")

    # 持久重试：事件日志跨尝试保留
    print("\n持久重试（事件日志跨尝试保留）")
    durable_log = EventLog(...)
    for attempt in range(1, 4):
        try:
            r = workflow(durable_log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            break
        except RuntimeError:
            print(f"    -> 崩溃; 重放已完成的活动")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 持久执行对照

| 工具 | 检查点后端 | 恢复粒度 | LLM 调用支持 |
|------|----------|---------|------------|
| Temporal + OpenAI SDK | PostgreSQL / SQLite / Redis | 活动级 | 原生 |
| LangGraph Checkpointer | PostgreSQL / SQLite / Memory | 节点级 | 手动 |
| Claude Code Routines | 服务器端持久化 | 动作级 | 原生 |
| Microsoft Agent Framework | PostgreSQL / Memory | 轮次级 | 手动 |

### 4.2 METR 35 分钟退化数据

| 任务长度 | 成功概率 | 说明 |
|---------|---------|------|
| ~35 分钟 | ~60% | 基线——METR 观察的退化起始点 |
| ~70 分钟 | ~36% | 时长加倍，失败率约翻倍 |
| ~140 分钟 | ~13% | 继续下降 |

持久执行不修复此退化。它使你可以运行得更久，但需要配合 HITL 和预算。

---

## 5. 工程最佳实践

### 5.1 持久执行设计原则

| 原则 | 说明 |
|------|------|
| 工作流必须是确定性的 | 不能有墙钟时间、随机数、LLM 调用直接在工作流中 |
| LLM 调用是活动 | 每个 LLM 调用包装为活动——重试、检查点、重放 |
| 35 分钟退化是真实限制 | 持久性不修复可靠性下降；需要 HITL 和预算 |
| 人类输入是一等状态 | 工作流暂停，外部队列持有请求，恢复从精确位置继续 |

### 5.2 中文场景特别建议

- **检查点后端选择在中文云环境中要考虑网络延迟**——如果 PostgreSQL 在另一区域，每次状态持久化会增加延迟
- **LLM 调用活动要考虑中文 API 的速率限制**——中文 API 的速率限制通常低于 OpenAI，重试策略需要调整
- **35 分钟退化在中文模型上的具体数字可能不同**——不同模型的行为曲线不同，但定性趋势类似

### 5.3 踩坑经验

- **工作流中的非确定性逻辑**——墙钟时间戳直接在工作流中，重放时结果不同。**修复：** 用 `Workflow.now()` API（真实引擎提供）而非直接调用时间函数
- **事件日志只存储在本地**——宿主机重启后日志丢失，持久性失效。**修复：** 用 PostgreSQL 跨主机持久化
- **不设预算终止开关**——持久性使智能体可以运行很远，但如果没有预算限制可能成本失控。**修复：** 持久性 + 预算终止开关配合

---

## 6. 常见错误

### 错误 1：在工作流代码中使用 LLM 调用

**现象：** 工作流函数直接调用了 LLM。重放时每次返回不同结果，重放日志不匹配——因为没有缓存 LLM 的输出。

**原因：** 工作流必须是确定性的。LLM 调用是非确定性的（温度 > 0 甚至温度 0 漂移）。

**修复：**
```python
# ❌ 错误：LLM 调用直接在工作流中
def workflow():
    response = llm("generate code")  # 重放时再次调用，结果不同
    return response

# ✓ 正确：LLM 调用包裹为活动
@activity("llm_call")
def llm_call(prompt: str) -> str: ...

def workflow():
    response = llm_call(log, "generate code")  # 重放时从日志返回缓存结果
    return response
```

### 错误 2：崩溃后没有检查点

**现象：** 长期智能体在宿主机重启后从头开始重跑。已完成的活动重新执行（有副作用的！），LLM 调用重新计费。

**原因：** 事件日志没有持久化。朴素 `while True` 循环在崩溃后不保留状态。

**修复：** 用持久后端（PostgreSQL）存储事件日志。`thread_id` 标识会话。工作流从最新检查点恢复。

### 错误 3：持久执行被当作可靠性解决方案

**现象：** 添加持久执行后让智能体运行 8 小时。触发次数越来越多；成功率越来越低。团队惊讶。

**原因：** 持久执行不修复 35 分钟退化——它让你跑得更久。

**修复：** 持久执行 + 定期 HITL 检查点（每 30 分钟）+ 预算终止开关。

---

## 7. 面试考点

### Q1：工作流和活动的区别是什么？为什么工作流必须是确定性的？（难度：⭐）

**参考答案：**
**工作流**是确定性编排代码——定义活动序列、分支、等待。必须是确定性的，以便可以从事件日志重放。

**活动**是任何非确定性的、可能失败的工作单元——LLM 调用、工具调用、文件写入、HTTP 请求。每个活动在执行前和执行后记录到事件日志。

工作流必须确定性的原因：如果工作流本身是非确定性的（如包含 `time.now()`），重放时会产生与原始执行不同的决策路径。事件日志记录了原始决策，但工作流重放时需要重现相同的决策序列。非确定性会打破这种重现。

### Q2：为什么 LLM 调用适合作为活动？与 Temporal 模式的关系是什么？（难度：⭐⭐）

**参考答案：**
LLM 调用是非确定性的（温度 > 0）、昂贵的（金钱和延迟）、可能失败的（速率限制、超时）、有副作用的（如果调用工具）。这正是 Temporal 活动中定义的活动画像。

将每个 LLM 调用包装为活动后：
- **重试**——指数退避
- **检查点**——跨重启持久化
- **重放**——不重新计费已完成的 LLM 调用
- **轨迹**——完整的可重放调试日志

Temporal 的 OpenAI Agents SDK 集成（2026 年 3 月 GA）直接将 LLM 调用作为一等活动类型。

### Q3：35 分钟退化是什么含义？持久执行如何与之相关？（难度：⭐⭐）

**参考答案：**
METR 观察到每个被测量的智能体类别在持续运行约 35 分钟后可靠性下降。将任务时长加倍大致使失败率翻倍。

持久执行不修复这个退化。它使你可以运行得比可靠性曲线支持的时间更长——这既可以是好事（完成长期任务），也可以是坏事（失败的长期任务消耗更多资源）。

**安全模式：** 持久性 + 定期 HITL 检查点 + 预算终止开关。持久性确保不丢失进度；HITL 检查点确保方向正确；预算终止开关确保成本可控。

### Q4：人类输入状态如何融入持久执行模型？（难度：⭐⭐⭐）

**参考答案：**
人类输入是工作流中的一等状态。工作流暂停，外部队列持有待处理的请求，当批准到达时从精确位置恢复。

没有持久化时，这是尽力而为——宿主机重启丢失挂起请求，人类需要重新批准。

有持久化时，隔夜批准到达事件日志，工作流第二天早上在数据库中看到并恢复。

这在 Temporal 中通过 `Workflow.await` 实现，在 LangGraph 中通过 `interrupt` 实现，在 Claude Code Routines 中通过暂停-恢复钩子实现。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作流 (Workflow) | "智能体的脚本" | 确定性编排代码；可从事件日志重放 |
| 活动 (Activity) | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；执行前后记录日志 |
| 事件日志 (Event Log) | "后端存储" | 每个状态转换的持久记录 |
| 重放 (Replay) | "恢复" | 重跑工作流；已完成活动返回缓存结果不重新执行 |
| 检查点 (Checkpoint) | "保存点" | 由 thread_id 键控的持久状态；最新胜出 |
| thread_id | "会话键" | 标识持久状态的会话范围键 |
| 35 分钟退化 | "可靠性下降" | METR：成功率大约二次方随视界下降 |
| 非确定性 | "重放时漂移" | 墙钟、随机、LLM 输出；必须注册为副作用 |

---

## 📚 小结

生产中的长期智能体不会在 `while True` 中运行。工作流（确定性编排）→ 活动（非确定性工作单元）→ 事件日志（持久存储）→ 重放（从日志恢复）。这是 Temporal、LangGraph、Claude Code Routines 和 Microsoft Agent Framework 使用的模式。

LLM 调用作为非确定性活动干净地适合——昂贵、有副作用、可能失败。35 分钟退化是真实限制——持久执行不修复可靠性下降，它让你运行得更久。安全设计：持久性 + HITL 检查点 + 预算。

下一课：成本门控——在运行失控之前停止它们。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。观察朴素重试和重放之间的活动执行计数差异。改变崩溃点看看重放计数相应变化。

2. **【实现】** 在玩具引擎中显式使用 `thread_id`。模拟两个并发的会话共享引擎，确认事件日志不碰撞。

3. **【实现】** 在一个活动中引入非确定性（工作流决策中的墙钟时间戳）。演示重放上的分歧。解释真实引擎如何处理此问题（副作用注册、`Workflow.now()` API）。

4. **【阅读】** 阅读 LangChain 的 "The Runtime Behind Production Deep Agents"。列出运行时持久化的每个状态以及各覆盖哪个失败模式。

5. **【设计】** 为 6 小时自主编码任务设计检查点策略。在哪里检查点？崩溃恢复看起来如何？什么需要全新 HITL？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 持久执行引擎 | `code/main.py` | @activity 装饰器 + 事件日志 + 重放机制 + 朴素 vs 持久对比 |
| 技能提示词 | `outputs/skill-durable-execution-review.md` | 审查提议的长期智能体部署是否具备正确持久执行形状 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Claude Code Agent SDK: Agent Loop". https://code.claude.com/docs/en/agent-sdk/agent-loop — 预算、轮次、恢复语义
2. [官方文档] Microsoft. "Agent Framework: Human-in-the-Loop and Checkpointing". https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop — RequestInfoEvent 形状
3. [官方文档] LangChain. "The Runtime Behind Production Deep Agents". https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents — 具体运行时要求
4. [博客] OpenAI Agents SDK + Temporal Integration. https://trigger.dev — LLM 调用的活动形状
5. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy — 35 分钟退化参考

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
