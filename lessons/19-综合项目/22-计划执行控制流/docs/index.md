# 综合项目22——计划执行控制流（重新规划+计划差异+双预算）

> 无法在失败中存活的计划是脚本。可以重新规划的脚本是智能体。先构建重新规划器。链式思考智能体发出token，让循环猜测工具调用何时结束。计划执行智能体先发出结构化计划，然后确定性地执行每个步骤。计划是智能体可自省的数据。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第13章（工具与协议）第01-07节、第14章（智能体）第01节
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 将计划表示为有序的类型化步骤列表
- 按顺序执行步骤，失败时有控制地回传给计划器
- 从当前游标带错误上下文重新规划
- 在每次修订时发出计划差异
- 执行两个硬预算：步骤上限和重新规划上限

---

## 1. 问题

链式思考智能体发出token，让循环猜测工具调用何时结束。计划执行智能体先发出结构化计划，然后确定性地执行。

两个组件：产生计划的计划器。运行计划的执行器。失败时有三个选项：

1. **中止**：返回失败，显示错误
2. **跳过**：标记步骤失败，继续其余
3. **重新规划**：将错误交给计划器，从游标获取新计划

重新规划是将脚本变成智能体的关键。

---

## 2. 核心概念

### 2.1 步骤形状

```
Step
  id              : int           (计划修订内单调递增)
  tool_name       : str
  args            : dict
  expected_outcome: str           (计划器声明的成功条件)
  result          : Any | None
  error           : str | None
```

### 2.2 计划器形状

```python
def planner(goal: str, history: list[Step], last_error: str | None) -> list[Step]:
    ...
```

纯函数。`goal`是用户目标。`history`是已执行的步骤。`last_error`首次调用为None，后续为最近失败信息。

### 2.3 计划差异

每次修订时发出`plan.diff`事件，包含`removed`（移除的步骤ID）、`added`（新增的步骤ID）、`revised`（tool_name或args改变的步骤ID）。

### 2.4 双预算

- **max_steps**：整个会话的总步骤执行数上限（包括重新规划），默认12
- **max_replans**：首次计划后调用计划器的次数上限，默认5

---

## 3. 从零实现

`code/main.py`实现`PlanExecuteAgent`、`Step`、`PlanDiff`、`SessionResult`和确定性计划器。

```python
"""计划执行智能体——失败重新规划、计划差异、双预算。

核心：结构化计划 → 顺序执行 → 失败时带错误上下文重新规划。
两个硬预算：步骤上限和重新规划上限。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Step:
    id: int; tool_name: str; args: dict; expected_outcome: str
    result: Any | None = None; error: str | None = None
    def signature(self): return (self.tool_name, json.dumps(self.args, sort_keys=True))


@dataclass
class PlanDiff:
    revision: int; removed: list[int]; added: list[int]; revised: list[int]
    def to_dict(self): return {"revision":self.revision,"removed":self.removed,"added":self.added,"revised":self.revised}


@dataclass
class Event:
    type: str; payload: dict; ts: float = field(default_factory=time.time)


@dataclass
class SessionResult:
    status: str; reason: str; history: list[Step]; revisions: list[PlanDiff]; events: list[Event]


Planner = Callable[[str, list[Step], str | None], list[Step]]
ToolExecutor = Callable[[str, dict], Any]


class ToolFailure(Exception): pass


def _diff_plans(old, new, revision):
    old_ids={s.id for s in old}; new_ids={s.id for s in new}
    removed=sorted(old_ids-new_ids); added=sorted(new_ids-old_ids)
    revised=[]
    old_by_id={s.id:s for s in old}
    for s in new:
        if s.id in old_ids and old_by_id[s.id].signature()!=s.signature(): revised.append(s.id)
    return PlanDiff(revision=revision,removed=removed,added=added,revised=revised)


class PlanExecuteAgent:
    def __init__(self, planner, executor, *, max_steps=12, max_replans=5):
        self._planner=planner; self._executor=executor
        self.max_steps=max_steps; self.max_replans=max_replans; self._events=[]

    def _emit(self, etype, payload): self._events.append(Event(type=etype,payload=payload))

    def run(self, goal):
        self._events=[]; history=[]; revisions=[]; steps_taken=0; replans_used=0; last_error=None
        plan=self._planner(goal,history,None)
        self._emit("plan.commit",{"revision":0,"steps":[s.expected_outcome for s in plan]})
        if not plan: return SessionResult("failed","no_plan",history,revisions,list(self._events))
        cursor=0; revision=0
        while cursor<len(plan):
            if steps_taken>=self.max_steps: return SessionResult("failed","step_budget",history,revisions,list(self._events))
            step=plan[cursor]
            self._emit("step.start",{"id":step.id,"tool":step.tool_name})
            try:
                step.result=self._executor(step.tool_name,step.args)
                self._emit("step.end",{"id":step.id,"outcome":"ok"})
                history.append(step); cursor+=1; steps_taken+=1; continue
            except Exception as exc:
                step.error=f"{type(exc).__name__}: {exc}"
                self._emit("step.end",{"id":step.id,"outcome":"error","error":step.error})
                history.append(step); steps_taken+=1; last_error=step.error
            if replans_used>=self.max_replans: return SessionResult("failed","replan_budget",history,revisions,list(self._events))
            replans_used+=1; revision+=1
            new_plan=self._planner(goal,history,last_error)
            self._emit("plan.draft",{"revision":revision,"steps":[s.expected_outcome for s in new_plan]})
            if not new_plan: return SessionResult("failed","no_plan",history,revisions,list(self._events))
            diff=_diff_plans(plan[cursor:],new_plan,revision); revisions.append(diff)
            self._emit("plan.diff",diff.to_dict())
            plan=new_plan; cursor=0
            self._emit("plan.commit",{"revision":revision,"steps":[s.expected_outcome for s in plan]})
        return SessionResult("completed","goal_met",history,revisions,list(self._events))


def make_planner(fail_step_id=None, recovery="route_around"):
    def planner(goal,history,last_error):
        if last_error is None:
            init=[Step(1,"fetch",{"key":"input"},"loaded"),Step(2,"transform",{"mode":"v1"},"computed v1"),
                  Step(3,"render",{},"rendered"),Step(4,"submit",{},"submitted")]
            if fail_step_id is not None:
                for s in init:
                    if s.id==fail_step_id: s.args={**s.args,"_force_fail":True}
            return init
        if recovery=="route_around" and "transform" in last_error:
            return [Step(2,"transform",{"mode":"v2"},"fallback"),Step(3,"render",{},"rendered"),Step(4,"submit",{},"submitted")]
        if recovery=="give_up":
            return [Step(98,"log_failure",{"why":last_error or ""},"logged"),Step(99,"notify_user",{},"notified")]
        return []
    return planner


def _demo():
    def executor(tool,args):
        if args.get("_force_fail"): raise ToolFailure(f"{tool} forced failure")
        if tool=="fetch": return {"k":"v"}
        if tool=="transform":
            if args.get("mode")=="v1": raise ToolFailure("transform v1 down")
            return {"ok":True}
        if tool=="render": return "html"
        if tool=="submit": return {"id":1}
        if tool in ("log_failure","notify_user"): return "ok"
        raise ToolFailure(f"unknown {tool}")

    agent=PlanExecuteAgent(planner=make_planner(fail_step_id=2,recovery="route_around"),executor=executor,max_steps=12,max_replans=5)
    res=agent.run("ship report")
    print(json.dumps({"status":res.status,"reason":res.reason,
        "history":[(s.id,s.tool_name,bool(s.error)) for s in res.history],
        "revisions":[r.to_dict() for r in res.revisions],
        "events":[e.type for e in res.events]},indent=2))


if __name__=="__main__": _demo()
```

运行结果：

```json
{
  "status": "completed",
  "reason": "goal_met",
  "history": [
    [1, "fetch", false],
    [2, "transform", true],
    [2, "transform", false],
    [3, "render", false],
    [4, "submit", false]
  ],
  "revisions": [
    {"revision": 1, "removed": [1], "added": [], "revised": [2]}
  ],
  "events": ["plan.commit", "step.start", "step.end", "step.start", "step.end", "plan.draft", "plan.diff", "plan.commit", "step.start", "step.end", "step.start", "step.end", "step.start", "step.end", "session.complete"]
}
```

---

## 4. 工具实践

**与第18-21节的集成**：
- 第18节的HarnessLoop提供事件流
- 第19节的ToolRegistry验证步骤参数
- 第20节的StdioTransport将流程暴露给模型客户端
- 第21节的Dispatcher执行每个步骤

**扩展方向**：
- 部分计划缓存：成功的前N步不需要重新执行
- 并行分支：独立步骤通过Dispatcher并发执行

---

## 5. LLM视角

**计划vs链式思考**：计划执行将"思考"转化为可审计的数据结构。链式思考让循环猜测工具调用何时结束；计划执行让执行器确定性地运行结构化数据。

**重新规划视角**：重新规划将脚本变成智能体。失败时带错误上下文重新规划，使下一次计划"知情"。

**差异视角**：计划差异是可观察事件而非静默重写。追踪器或UI可以显示计划为什么改变。

---

## 6. 工程最佳实践

**预算设计**：双预算——步骤上限防止无限循环，重新规划上限使失败更快、原因更清晰。

**确定性**：执行器是确定性的——给定相同历史+错误，产生相同新计划。

**步骤签名**：用于计算差异——(tool_name, args)元组。

---

## 7. 常见错误

**错误1：不限制重新规划次数**
症状：计划器返回相同错误计划无限循环
修复：设置max_replans上限

**错误2：不追踪已执行步骤**
症状：重新规划后重做已完成的步骤
修复：从当前游标继续

---

## 8. 面试考点

**Q1：计划执行与链式思考的关键区别是什么？**
考察：对智能体架构的理解

**Q2：为什么需要计划差异而非静默重写？**
考察：对可观察性的理解

**Q3：双预算各自防止什么？**
考察：对安全边界的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 计划执行 | "结构化执行" | 先发出计划，再确定性执行每个步骤 |
| 重新规划 | "失败恢复" | 将错误交给计划器，从游标获取新计划 |
| 计划差异 | "修订追踪" | 每次修订时的步骤增删改记录 |
| 步骤签名 | "步骤指纹" | (tool_name, args)元组，用于差异计算 |
| 双预算 | "步骤+重新规划限制" | max_steps和max_replans两个硬上限 |
| expected_outcome | "成功条件" | 计划器声明的步骤成功条件 |

---

## 参考文献

- [Plan-and-Execute范式](https://smith.langchain.com/blog/plan-and-execute)
- [LangGraph计划执行](https://langchain-ai.github.io/langgraph/)
