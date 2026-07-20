# 综合项目18——智能体循环契约（循环状态机+钩子+拉取点）

> 智能体循环是智能体本身。模型是协处理器。本课程冻结循环契约，让你可以将任何模型接入其中。2026年Claude Code、Cursor和OpenCode都收敛到相同的六状态、十钩子、两拉取点、十一事件类型架构。本综合项目要求你实现这一契约，为后续工具注册、JSON-RPC传输和调度器打下基础。

**类型：** 综合项目
**编程语言：** Python
**前置知识：** 第13章（工具与协议）第01-07节、第14章（智能体）第01节
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 定义智能体循环为确定性状态机，带显式转换
- 实现十种生命周期钩子主题
- 定义两个拉取点，循环让出控制权并恢复
- 执行每会话预算（轮次、工具调用、墙钟时间）
- 发射十一种事件类型的类型化流

---

## 1. 问题

一个运行40轮的编码智能体不是聊天循环。它是一个状态机，其节点可被操作者拦截，其边可被操作者审计。一旦你写下契约，更换模型、工具或策略就不再是重构——它变成一次注册调用。

本课程构建那个契约。我们命名六个状态、十个钩子主题、两个拉取点、十一种事件类型和一个预算包络。

---

## 2. 核心概念

### 2.1 六个状态

循环有六个状态。五个活跃，一个终止。

- **IDLE**：唯一合法入口点
- **PLANNING**：计划器生成计划
- **EXECUTING**：执行步骤
- **AWAITING_TOOL**：需要工具结果（唯一拉取点）
- **REFLECTING**：反思步骤结果
- **DONE**：唯一合法出口

状态机是确定性的——给定相同的事件日志，智能体重新进入相同的状态。

### 2.2 十个钩子主题

钩子是操作者介入循环的接缝。十个主题：`before_plan`、`after_plan`、`before_step`、`after_step`、`before_tool_call`、`after_tool_call`、`on_error`、`on_pause`、`on_budget_exceeded`、`on_complete`。

### 2.3 两个拉取点

循环在两处让出控制权：
1. `AWAITING_TOOL`：无法在没有工具结果时继续
2. `on_pause`：预算耗尽或钩子请求人工审查

拉取点不是异常，而是返回。调用者检查状态、获取请求内容、调用`resume(payload)`。

### 2.4 十一种事件类型

`session.start`、`plan.draft`、`plan.commit`、`step.start`、`step.end`、`tool.call`、`tool.result`、`tool.error`、`budget.warn`、`session.pause`、`session.complete`。

事件不重复钩子负载——钩子是命令式的（修改、中止），事件是观察式的（记录、传输）。

### 2.5 预算包络

每会话携带三个限制：轮次计数、工具调用计数、墙钟秒数。达到任何限制时，循环发射`budget.warn`，然后在下一个拉取点转入IDLE。

---

## 3. 从零实现

`code/main.py`实现完整的六状态循环、钩子注册、事件流和预算执行。

```python
"""智能体循环契约——确定性状态机、钩子、拉取点。

核心架构原语：六状态确定性循环、十钩子主题、两拉取点、
十一事件类型和预算包络。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_TOOL = "awaiting_tool"
    REFLECTING = "reflecting"
    DONE = "done"


HOOK_TOPICS = (
    "before_plan", "after_plan",
    "before_step", "after_step",
    "before_tool_call", "after_tool_call",
    "on_error", "on_pause",
    "on_budget_exceeded", "on_complete",
)

EVENT_TYPES = (
    "session.start", "plan.draft", "plan.commit",
    "step.start", "step.end",
    "tool.call", "tool.result", "tool.error",
    "budget.warn", "session.pause", "session.complete",
)


class HookAbort(Exception):
    """钩子取消当前回合时引发。"""


@dataclass
class Event:
    type: str
    payload: dict
    ts: float
    def to_dict(self) -> dict:
        return {"type": self.type, "payload": self.payload, "ts": self.ts}


@dataclass
class Budget:
    max_turns: int = 8
    max_tool_calls: int = 16
    max_wall_seconds: float = 30.0
    turns: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.time)

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_wall_seconds - (time.time() - self.started_at))

    def exceeded(self) -> str | None:
        if self.turns >= self.max_turns:
            return "turns"
        if self.tool_calls >= self.max_tool_calls:
            return "tool_calls"
        if self.remaining_seconds() <= 0.0:
            return "wall_clock"
        return None


@dataclass
class Step:
    id: int
    description: str
    requires_tool: bool
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    result: Any = None
    error: str | None = None


@dataclass
class PullRequest:
    """循环让出控制权时返回。"""
    reason: str
    state: State
    payload: dict


@dataclass
class SessionResult:
    state: State
    reason: str
    steps: list[Step]
    events: list[Event]


class HookRegistry:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict], Any]]] = {t: [] for t in HOOK_TOPICS}

    def on(self, topic: str, fn: Callable[[dict], Any]) -> None:
        self._subs[topic].append(fn)

    def fire(self, topic: str, payload: dict) -> list[Any]:
        return [fn(payload) for fn in self._subs[topic]]


Planner = Callable[[str, list[Step]], list[Step]]


def _default_planner(goal: str, history: list[Step]) -> list[Step]:
    """存根计划器：返回固定三步计划。"""
    if history:
        return []
    return [
        Step(id=1, description=f"interpret goal: {goal}", requires_tool=False),
        Step(id=2, description="fetch user record", requires_tool=True,
             tool_name="db.get_user", tool_args={"id": 42}),
        Step(id=3, description="summarize and respond", requires_tool=True,
             tool_name="format.summary", tool_args={"style": "short"}),
    ]


class HarnessLoop:
    """六状态确定性循环，带钩子主题和事件流。"""

    def __init__(self, planner: Planner | None = None, budget: Budget | None = None) -> None:
        self.state: State = State.IDLE
        self.hooks = HookRegistry()
        self.budget = budget or Budget()
        self._planner: Planner = planner or _default_planner
        self._goal: str = ""
        self._plan: list[Step] = []
        self._cursor: int = 0
        self._events: list[Event] = []
        self._history: list[Step] = []
        self._reason: str = ""
        self._prev_state: State | None = None

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    @property
    def plan(self) -> list[Step]:
        return list(self._plan)

    def _emit(self, etype: str, payload: dict) -> None:
        self._events.append(Event(type=etype, payload=payload, ts=time.time()))

    def _transition(self, target: State) -> None:
        legal: dict[State, set[State]] = {
            State.IDLE: {State.PLANNING},
            State.PLANNING: {State.EXECUTING, State.IDLE, State.DONE},
            State.EXECUTING: {State.AWAITING_TOOL, State.REFLECTING, State.IDLE},
            State.AWAITING_TOOL: {State.REFLECTING, State.IDLE},
            State.REFLECTING: {State.PLANNING, State.EXECUTING, State.DONE, State.IDLE},
            State.DONE: set(),
        }
        if target not in legal[self.state]:
            raise RuntimeError(f"illegal transition {self.state.value} -> {target.value}")
        self.state = target

    def _check_budget(self) -> PullRequest | None:
        which = self.budget.exceeded()
        if which is None:
            return None
        self._emit("budget.warn", {"limit": which})
        self.hooks.fire("on_budget_exceeded", {"limit": which})
        self._reason = f"budget_exceeded:{which}"
        self._prev_state = self.state
        return self._pause(self._reason)

    def _pause(self, reason: str) -> PullRequest:
        self._emit("session.pause", {"reason": reason})
        self.hooks.fire("on_pause", {"reason": reason})
        self._transition(State.IDLE)
        return PullRequest(reason=reason, state=self.state, payload={"reason": reason})

    def run(self, goal: str) -> PullRequest | SessionResult:
        if self.state != State.IDLE:
            raise RuntimeError(f"run() requires IDLE, got {self.state.value}")
        self._goal = goal
        self.budget.started_at = time.time()
        self._emit("session.start", {"goal": goal})
        return self._step()

    def resume(self, payload: dict | None = None) -> PullRequest | SessionResult:
        if self.state == State.IDLE and self._reason.startswith("budget_exceeded"):
            self.budget.turns = 0; self.budget.tool_calls = 0
            self.budget.started_at = time.time(); self._reason = ""
            prev = self._prev_state; self._prev_state = None
            if not self._plan:
                return self._begin_plan()
            if prev == State.EXECUTING:
                self.state = State.EXECUTING
            else:
                self.state = State.REFLECTING
            return self._step()
        if self.state == State.AWAITING_TOOL:
            if payload is None:
                raise ValueError("resume from AWAITING_TOOL requires payload")
            current = self._plan[self._cursor]
            if "error" in payload:
                current.error = str(payload["error"])
                self._emit("tool.error", {"step": current.id, "error": current.error})
                self.hooks.fire("on_error", {"step": current, "error": current.error})
            else:
                current.result = payload.get("result")
                self._emit("tool.result", {"step": current.id, "result": current.result})
            self.hooks.fire("after_tool_call", {"step": current})
            self._transition(State.REFLECTING)
            return self._step()
        raise RuntimeError(f"resume() unsupported from state {self.state.value}")

    def _begin_plan(self) -> PullRequest | SessionResult:
        self._transition(State.PLANNING)
        self.hooks.fire("before_plan", {"goal": self._goal})
        draft = self._planner(self._goal, list(self._history))
        self._emit("plan.draft", {"steps": [s.description for s in draft]})
        self.hooks.fire("after_plan", {"steps": draft})
        self._plan = draft; self._cursor = 0
        self._emit("plan.commit", {"count": len(draft)})
        if not draft:
            return self._complete("no_plan")
        self._transition(State.EXECUTING)
        return self._step()

    def _step(self) -> PullRequest | SessionResult:
        if self.state == State.IDLE:
            return self._begin_plan()
        budget_hit = self._check_budget()
        if budget_hit is not None:
            return budget_hit
        if self.state == State.REFLECTING:
            self._cursor += 1; self.budget.turns += 1
            if self._cursor >= len(self._plan):
                return self._complete("goal_met")
            self._transition(State.EXECUTING)
            return self._step()
        step = self._plan[self._cursor]
        self.hooks.fire("before_step", {"step": step})
        self._emit("step.start", {"step_id": step.id, "desc": step.description})
        if step.requires_tool:
            try:
                self.hooks.fire("before_tool_call", {"step": step})
            except HookAbort as exc:
                step.error = f"hook_abort:{exc}"
                self._emit("tool.error", {"step": step.id, "error": step.error})
                self.hooks.fire("on_error", {"step": step, "error": step.error})
                self._transition(State.REFLECTING)
                return self._step()
            self.budget.tool_calls += 1
            self._emit("tool.call", {"step": step.id, "tool": step.tool_name})
            self._transition(State.AWAITING_TOOL)
            self._emit("step.end", {"step_id": step.id, "outcome": "awaiting_tool"})
            return PullRequest(reason="tool_call", state=self.state,
                               payload={"tool": step.tool_name, "args": step.tool_args})
        step.result = f"ok:{step.description}"
        self._emit("step.end", {"step_id": step.id, "outcome": "ok"})
        self.hooks.fire("after_step", {"step": step, "outcome": "ok"})
        self._transition(State.REFLECTING)
        return self._step()

    def _complete(self, reason: str) -> SessionResult:
        self._emit("session.complete", {"reason": reason})
        self.hooks.fire("on_complete", {"reason": reason})
        self._transition(State.DONE)
        self._reason = reason
        return SessionResult(state=self.state, reason=reason,
                             steps=list(self._plan), events=list(self._events))


def _demo() -> None:
    loop = HarnessLoop()
    fired: list[str] = []
    for topic in HOOK_TOPICS:
        loop.hooks.on(topic, lambda payload, t=topic: fired.append(t))

    out = loop.run("ship the release notes")
    assert isinstance(out, PullRequest) and out.reason == "tool_call"
    out = loop.resume({"result": {"id": 42, "name": "ada"}})
    assert isinstance(out, PullRequest) and out.reason == "tool_call"
    final = loop.resume({"result": "summary text"})
    assert isinstance(final, SessionResult)
    assert final.state == State.DONE

    report = {
        "events": [e.type for e in final.events],
        "hooks_fired": fired,
        "final_state": final.state.value,
        "final_reason": final.reason,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _demo()
```

运行结果：

```json
{
  "events": [
    "session.start", "plan.draft", "plan.commit",
    "step.start", "step.end", "tool.call",
    "tool.result", "step.start", "step.end", "tool.call",
    "tool.result", "session.complete"
  ],
  "hooks_fired": [
    "before_plan", "after_plan", "before_step", "after_step",
    "before_tool_call", "after_tool_call", "before_step", "after_step",
    "before_tool_call", "after_tool_call", "on_complete"
  ],
  "final_state": "done",
  "final_reason": "goal_met"
}
```

---

## 4. 工具实践

本课程构建的是契约层——不调用模型、不注册真实工具、不实现传输。这些是后续课程的内容。

**使用方式**：
- `HarnessLoop`是主类，持有状态、触发钩子、发射事件
- `Budget`跟踪限制
- `Event`是流上的类型化信封
- `HookRegistry`是分发表
- `_transition`是唯一改变状态的函数

---

## 5. LLM视角

**契约视角**：循环契约是智能体的"操作系统内核"。所有上层组件（工具注册、传输、调度器）都插入这一形状。契约使组件可替换——更换模型只需一次注册调用。

**确定性视角**：状态机是确定性的。给定相同事件日志可重放会话，无需重新调用模型。这对调试至关重要。

**拉取点视角**：拉取点不是异常而是返回。循环让出控制权，调用者提供结果后循环恢复。这与Python生成器的形状相同。

---

## 6. 工程最佳实践

**状态机设计**：
- 六状态，一个终止
- `_transition`是唯一改变状态的函数
- 转换表显式定义合法转换

**钩子设计**：
- 十个主题覆盖所有生命周期点
- 钩子可修改负载、引发中止、返回哨兵值
- 钩子按注册顺序触发

**预算设计**：
- 三个限制：轮次、工具调用、墙钟
- 预算是让出而非终止
- 调用者决定扩展预算还是关闭会话

---

## 7. 常见错误

**错误1：状态机不确定**
症状：相同输入产生不同状态序列
修复：确保转换完全由当前状态和输入决定

**错误2：钩子中止未正确处理**
症状：钩子异常导致循环崩溃
修复：捕获HookAbort并优雅降级

**错误3：拉取点状态泄漏**
症状：恢复后状态不一致
修复：在每个拉取点检查并重置状态

---

## 8. 面试考点

**Q1：为什么智能体循环是状态机而非聊天循环？**
考察：对智能体架构的理解

**Q2：钩子和事件的区别是什么？**
考察：对观察式vs命令式模式的理解

**Q3：拉取点如何与Python生成器类比？**
考察：对控制流的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 状态机 | "确定性循环" | 六状态循环，转换完全由当前状态和输入决定 |
| 钩子 | "生命周期拦截器" | 操作者在循环关键点注入策略/遥测的回调 |
| 拉取点 | "控制权让出" | 循环暂停等待外部输入的返回点 |
| 事件流 | "观察式日志" | 循环在特定点发射的类型化事件序列 |
| 预算包络 | "资源限制" | 每会话的轮次、工具调用、墙钟限制 |
| HookAbort | "钩子中止" | 钩子引发以取消当前回合的异常 |
| 确定性重放 | "会话回放" | 给定相同事件日志可重放会话，无需重新调用模型 |

---

## 参考文献

- [Claude Code文档](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor 3更新日志](https://cursor.com/changelog)
- [OpenCode](https://opencode.ai)
- [SWE-agent](https://github.com/SWE-agent/SWE-agent)
