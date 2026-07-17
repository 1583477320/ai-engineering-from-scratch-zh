# 长期后台智能体：持久执行

> 生产中的长期智能体不会在 `while True` 中运行。每个 LLM 调用都变成了一个带检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月 GA。Claude Code Routines（Anthropic）运行定时调度的 Claude Code 调用，无需持久本地进程。会话在人类输入时暂停，在部署后存活，并从由 `thread_id` 键控的最新检查点恢复。在新的用户界面背后是一个老模式——工作流编排——只有一个新输入：LLM 调用作为非确定性活动，必须在恢复时确定性地重放。

**类型：** 概念课
**语言：** Python（标准库，最小持久执行状态机）
**前置知识：** 阶段 15 · 10（权限模式）、阶段 15 · 01（长期智能体）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分工作流（确定性编排代码）和活动（非确定性工作单元）
- [ ] 理解事件日志 + 重放模式如何使长期运行在崩溃时幸存
- [ ] 解释为什么 LLM 调用适合作为活动——非确定性、昂贵、有副作用
- [ ] 理解"35 分钟退化"——METR 观察到的可靠性大致随视界二次方下降
- [ ] 识别何时持久执行是错误的选择

---

## 1. 问题

考虑一个运行四小时的智能体。它调用三个工具，提示用户两次，进行四十次 LLM 调用。中途，宿主机重启。会发生什么？

- 在朴素的 `while True` 循环中：一切丢失。运行从头开始重跑。三个工具调用（有真实副作用的）再次执行。用户再次被他们已经批准的事情提示。四十次 LLM 调用被重新计费。
- 使用持久执行：运行从最近的检查点恢复。已完成的活动不重新执行；它们的结果从持久日志重放。用户不重新批准他们已经批准的事情。已经进行的 LLM 调用不重新计费。

这是工作流引擎已经提供了十年的模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于 LLM 调用现在是一种活动——非确定性、昂贵、有副作用——它们干净地适合这个模式。

---

## 2. 概念

### 2.1 工作流、活动、重放

| 术语 | 定义 | 特性 |
|------|------|------|
| **工作流** | 确定性编排代码 | 定义活动序列、分支、等待。必须确定性以便从事件日志重放 |
| **活动** | 非确定性、可能失败的工作单元 | LLM 调用、工具调用、文件写入、HTTP 请求。每个记录输入和输出 |
| **事件日志** | 持久后端存储 | 记录每个活动开始、完成、失败、重试和每个工作流决策 |
| **重放** | 恢复过程 | 工作流代码从头重跑；已完成活动返回记录结果而不重新执行 |

这与 React 对虚拟 DOM 重新渲染或 Git 从提交重建工作树是相同的形状。编排器中的确定性使持久性变得廉价。

### 2.2 为什么 LLM 调用适合这个模式

LLM 调用是：
- **非确定性**（温度 > 0；即使温度 0 也跨模型版本漂移）
- **昂贵**（金钱和延迟）
- **可能失败**（速率限制、超时）
- **有副作用**（如果它们调用工具）

这正是活动的画像。将每个 LLM 调用包装为活动提供指数退避重试、跨重启检查点、可重放的调试轨迹。

### 2.3 检查点后端

| 后端 | 特性 | 适用场景 |
|------|------|---------|
| PostgreSQL | 持久、可查询、跨部署存活 | 生产默认 |
| SQLite | 本地开发 | 跨主机丢失数据 |
| Redis | 快速但易失 | 除非 AOF/快照配置 |
| Cloudflare Durable Objects | 透明分布式 | 小时至周存活 |

### 2.4 35 分钟退化

METR 观察到每个被测量的智能体类别在持续运行约 35 分钟后显示可靠性下降。将任务时长加倍大致使失败率翻倍。

持久执行不修复这个；它让你运行得比可靠性曲线支持的时间更长。安全模式是将持久性与要求在重新进入时全新 HITL 的检查点和上限总计算的预算终止开关结合。

### 2.5 何时持久执行是错误的选择

- 运行短于几分钟且无人类输入——开销 > 收益
- 严格只读的信息检索
- 要求在单个上下文窗口内端到端正确的任务

---

## 3. 从零实现

### 第 1 步：定义事件日志

```python
import json, os
from dataclasses import dataclass

@dataclass
class EventLog:
    path: str

    def events(self) -> list[dict]:
        with open(self.path) as f:
            return json.load(f)

    def append(self, ev: dict) -> None:
        evs = self.events()
        evs.append(ev)
        with open(self.path, "w") as f:
            json.dump(evs, f)

    def lookup(self, name: str, args: tuple) -> dict | None:
        for ev in self.events():
            if ev["name"] == name and ev["args"] == list(args) and ev["status"] == "done":
                return ev
        return None
```

### 第 2 步：实现活动装饰器

```python
import functools

def activity(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(log: EventLog, *args):
            hit = log.lookup(name, args)
            if hit:
                print(f"    [replay] {name}({args}) -> {hit['result']} (from log)")
                return hit["result"]
            log.append({"name": name, "args": list(args), "status": "started"})
            result = fn(*args)
            log.append({"name": name, "args": list(args), "status": "done", "result": result})
            print(f"    [run]    {name}({args}) -> {result}")
            return result
        return wrapper
    return deco
```

### 第 3 步：定义示例活动

```python
@activity("fetch_docs")
def fetch_docs(query: str) -> int:
    return len(query) * 3

@activity("call_llm")
def call_llm(doc_count: int) -> str:
    return f"summary({doc_count}_docs)"

@activity("write_report")
def write_report(summary: str) -> str:
    return f"report://{summary}"
```

### 第 4 步：实现工作流并演示

```python
def workflow(log: EventLog, query: str, crash_after: int = -1) -> str:
    doc_count = fetch_docs(log, query)
    if crash_after == 1:
        raise RuntimeError("simulated crash after fetch_docs")
    summary = call_llm(log, doc_count)
    if crash_after == 2:
        raise RuntimeError("simulated crash after call_llm")
    report = write_report(log, summary)
    return report

def main():
    print("朴素重试（每次从头）：")
    # 每次尝试从头——所有活动重新运行

    print("\n持久重试（事件日志跨尝试保留）：")
    # 重放已完成的活动；只有缺失的活动实际运行
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 持久执行对照

| 工具 | 检查点后端 | 恢复粒度 |
|------|----------|---------|
| Temporal + OpenAI SDK | PostgreSQL / SQLite / Redis | 活动级 |
| LangGraph Checkpointer | PostgreSQL / SQLite / Memory | 节点级 |
| Claude Code Routines | 服务器端持久化 | 动作级 |
| Microsoft Agent Framework | PostgreSQL / Memory | 轮次级 |

---

## 5. 工程最佳实践

### 5.1 持久执行设计原则

| 原则 | 说明 |
|------|------|
| 工作流必须是确定性的 | 不能有墙钟时间、随机数、LLM 调用直接在工作流中 |
| LLM 调用是活动 | 每个 LLM 调用包装为活动——重试、检查点、重放 |
| 35 分钟退化是真实限制 | 持久性不修复可靠性下降；需要 HITL 和预算 |
| 人类输入是一等状态 | 工作流暂停，外部队列持有请求，恢复从精确位置继续 |

---

## 6. 常见错误

### 错误 1：在工作流代码中使用 LLM 调用

**现象：** 工作流函数直接调用了 LLM。重放时每次返回不同结果，重放日志不匹配。

**原因：** 工作流必须是确定性的——LLM 调用是非确定性的。

**修复：** 所有 LLM 调用包裹为活动。工作流只编排活动序列。

### 错误 2：崩溃后没有检查点

**现象：** 长期智能体在宿主机重启后从头开始重跑。已完成的活动重新执行（有副作用的！），LLM 调用重新计费。

**原因：** 事件日志没有持久化。

**修复：** 用 PostgreSQL 或其他持久后端存储事件日志。`thread_id` 标识会话。

### 错误 3：忽略 35 分钟退化

**现象：** 添加持久执行后让智能体运行 8 小时。触发次数越来越多；成功率越来越低。

**原因：** 持久执行不修复可靠性下降——它让你跑得更久。

**修复：** 持久执行 + 定期 HITL 检查点 + 预算终止开关。

---

## 7. 面试考点

### Q1：工作流和活动的区别是什么？（难度：⭐）

**参考答案：**
工作流是确定性编排代码——定义活动序列、分支、等待。必须确定性以便重放。

活动是非确定性的工作单元——LLM 调用、工具调用、文件写入。每个记录输入和输出。

重放时，工作流从头重跑，已完成的从日志返回结果，不重新执行。

### Q2：为什么 LLM 调用适合作为活动？（难度：⭐⭐）

**参考答案：**
LLM 调用是非确定性的（温度 > 0 甚至温度 0 漂移）、昂贵的（金钱和延迟）、可能失败的（速率限制、超时）、有副作用的（如果调用工具）。

这正是活动的画像。包装为活动后：指数退避重试、跨重启检查点、可重放轨迹。

这与 Temporal、LangGraph、Claude Code Routines 使用的模式相同。

### Q3：35 分钟退化是什么？持久执行如何与之相关？（难度：⭐⭐）

**参考答案：**
METR 观察到每个智能体类别在持续运行约 35 分钟后可靠性下降。时长加倍大致使失败率翻倍。

持久执行不修复这个退化。它让你跑得比可靠性曲线支持的时间更长。安全模式：持久性 + 定期 HITL 检查点 + 预算终止开关。

### Q4：多线程持久化中 `thread_id` 的作用是什么？（难度：⭐⭐⭐）

**参考答案：**
`thread_id` 标识会话并限定持久状态的范围。同一引擎上的两个并发会话用不同 id，它们的事件日志不合并。

LangGraph、Microsoft Agent Framework、Claude Code Routines 都收敛到同一个 API 形状：`thread_id` → PostgreSQL/SQLite 后端 → 从最新检查点恢复。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作流 | "智能体的脚本" | 确定性编排代码；可从事件日志重放 |
| 活动 | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；执行前后记录 |
| 事件日志 | "后端存储" | 每个状态转换的持久记录 |
| 重放 | "恢复" | 重跑工作流；已完成活动返回记录结果不重新执行 |
| 检查点 | "保存点" | 由 thread_id 键控的持久状态；最新胜出 |
| 35 分钟退化 | "可靠性下降" | METR：成功率大约二次方随视界下降 |
| 非确定性 | "重放时漂移" | 墙钟、随机、LLM 输出；必须注册为副作用 |

---

## 📚 小结

生产中的长期智能体不会在 `while True` 中运行。工作流 → 活动 → 事件日志 → 重放——这是 Temporal、LangGraph、Claude Code Routines 使用的模式。LLM 调用作为非确定性活动干净地适合。35 分钟退化是真实限制——持久执行不修复它，让你跑得更久。安全设计：持久性 + HITL 检查点 + 预算。

下一课：成本门控——在运行失控之前停止它们。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。观察朴素重试和重放之间的活动执行计数差异。改变崩溃点看看重放计数相应变化。

2. **【实现】** 在玩具引擎中显式使用 `thread_id`。模拟两个并发的会话共享引擎，确认事件日志不碰撞。

3. **【实现】** 在一个活动中引入非确定性（工作流决策中的墙钟时间戳）。演示重放上的分歧。解释真实引擎如何处理。

4. **【阅读】** 阅读 LangChain 的 "Runtime Behind Production Deep Agents"。列出运行时持久化的每个状态以及各覆盖哪个失败模式。

5. **【设计】** 为 6 小时自主编码任务设计检查点策略。你在哪里检查点？崩溃恢复看起来如何？什么需要全新 HITL？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 持久执行引擎 | `code/main.py` | 最小活动/日志/重放引擎 |
| 技能提示词 | `outputs/skill-durable-execution-review.md` | 审查提议的长期智能体部署是否正确 |

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
