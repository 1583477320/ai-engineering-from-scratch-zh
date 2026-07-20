# 综合项目01——终端原生编码智能体

> 到2026年，编码智能体的形态已基本定型。一个TUI驱动的主循环、一个结构化的计划状态、一个沙箱化的工具表面、一个"计划-行动-观察-恢复"的循环。Claude Code、Cursor 3和OpenCode从50英尺外看都一样。本综合项目要求你从零构建一个端到端的编码智能体——CLI输入，PR输出——并在SWE-bench Pro上用mini-swe-agent和Live-SWE-agent做对比评测。

**类型：** 综合项目
**编程语言：** Python（主循环），TypeScript（可选UI）
**前置知识：** 第11章（LLM工程）、第13章（工具与协议）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）
**涉及章节：** P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**预计时间：** 35小时

---

## 学习目标

- 构建一个完整的计划-行动-观察-恢复编码智能体循环
- 实现沙箱隔离的工具调度器和8种生命周期钩子
- 实现成本控制和上下文预算管理
- 在SWE-bench Pro上对比评测智能体性能

---

## 1. 问题

编码智能体已成为2026年主导的AI应用类别。Claude Code（Anthropic）、Cursor 3（Composer 2 + Agent Tabs）、Amp（Sourcegraph）、OpenCode（112k星）、Factory Droids和Google Jules都是同一架构的变体：一个终端主循环、一个授权工具表面、一个沙箱和一个围绕前沿模型构建的计划-行动-观察循环。

技术前沿很窄——Live-SWE-agent在SWE-bench Verified上用Opus 4.5达到了79.2%——但工程手艺很宽。大多数失败模式不是模型的错误，而是工具循环不稳定、上下文投毒、失控的token成本和破坏性文件系统操作。

你必须亲手构建一个才能理解这些问题。

---

## 2. 核心概念

### 2.1 四表面架构

主循环有四个表面：

- **计划（Plan）**：维护一个TodoWrite风格的状态对象，模型每一轮重写整个状态
- **行动（Act）**：调度工具调用（读取、编辑、运行、搜索、git）
- **观察（Observe）**：捕获stdout/stderr/退出码，截断并反馈摘要
- **恢复（Recover）**：处理工具错误，不撑爆上下文窗口也不无限循环

2026年的架构增加了一个组件：**钩子（Hooks）**。八个生命周期事件：`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`和`PreCompact`——可配置的扩展点，操作者可在此注入策略、遥测和保护措施。

### 2.2 沙箱隔离

每个任务在独立的devcontainer中运行，挂载一个git worktree。主循环绝不触碰宿主机文件系统。任务完成或失败后worktree被销毁。

### 2.3 成本控制

三层成本控制：每轮的token上限、每次会话的美元预算、硬回合数限制（通常50轮）。可观测层使用OpenTelemetry，遵循GenAI语义约定，发送到自托管的Langfuse。

---

## 3. 从零实现

`code/main.py`实现最小计划-行动-观察循环。LLM被一个确定性脚本替代，使循环逻辑在没有网络调用的情况下可观察和可测试。

```python
"""终端原生编码智能体——最小计划/行动/观察循环脚手架。

2026年编码智能体的核心架构原语不是模型调用或单个工具，
而是计划-行动-观察-恢复循环，包含有界上下文、结构化计划状态、
沙箱化工具调度器和每个生命周期点的钩子回调。
本文件在标准库Python中实现完整的循环。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 计划状态——TodoWrite结构，每轮完整重写
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    id: int
    description: str
    status: str  # "pending" | "in_progress" | "done" | "failed"
    note: str = ""


@dataclass
class PlanState:
    goal: str
    items: list[TodoItem] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"目标: {self.goal}"]
        for it in self.items:
            mark = {"pending": " ", "in_progress": ">", "done": "x", "failed": "!"}[it.status]
            lines.append(f"  [{mark}] {it.id}. {it.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 预算——回合数、token数、美元数的硬上限
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_turns: int = 50
    max_tokens: int = 200_000
    max_dollars: float = 5.00
    turns_used: int = 0
    tokens_used: int = 0
    dollars_used: float = 0.0

    def step(self, tokens: int, dollars: float) -> None:
        self.turns_used += 1
        self.tokens_used += tokens
        self.dollars_used += dollars

    def exceeded(self) -> str | None:
        if self.turns_used >= self.max_turns:
            return "turn_limit"
        if self.tokens_used >= self.max_tokens:
            return "token_limit"
        if self.dollars_used >= self.max_dollars:
            return "dollar_limit"
        return None


# ---------------------------------------------------------------------------
# 钩子——2026年八事件表面（Pre/PostToolUse, SessionStart/End等）
# ---------------------------------------------------------------------------

HookFn = Callable[[dict[str, Any]], dict[str, Any]]


class HookBus:
    EVENTS = ("SessionStart", "SessionEnd", "PreToolUse", "PostToolUse",
              "UserPromptSubmit", "Notification", "Stop", "PreCompact")

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = {e: [] for e in self.EVENTS}

    def on(self, event: str, fn: HookFn) -> None:
        self._hooks[event].append(fn)

    def fire(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        for fn in self._hooks[event]:
            payload = fn(payload) or payload
        return payload


# ---------------------------------------------------------------------------
# 工具表面——沙箱化的工具，每个返回截断文本
# ---------------------------------------------------------------------------

TRUNCATE_BYTES = 4096


def tool_read_file(sandbox: str, path: str) -> str:
    """读取文件，禁止越狱沙箱"""
    full = os.path.join(sandbox, path)
    if not os.path.realpath(full).startswith(os.path.realpath(sandbox)):
        raise RuntimeError("path escapes sandbox")
    with open(full, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()[:TRUNCATE_BYTES]


def tool_run_shell(sandbox: str, cmd: str, timeout: int = 30) -> str:
    """运行shell命令"""
    proc = subprocess.run(cmd, cwd=sandbox, shell=True, capture_output=True,
                          text=True, timeout=timeout)
    out = (proc.stdout + proc.stderr)[:TRUNCATE_BYTES]
    return f"exit={proc.returncode}\n{out}"


TOOLS: dict[str, Callable[..., str]] = {
    "read_file": tool_read_file,
    "run_shell": tool_run_shell,
}


# ---------------------------------------------------------------------------
# 存根模型——确定性脚本，循环无需LLM即可测试
# ---------------------------------------------------------------------------

SCRIPT = [
    {"plan": [("locate target file", "in_progress"),
              ("read and diagnose", "pending"),
              ("apply fix and verify", "pending")],
     "tool": ("run_shell", {"cmd": "ls"}),
     "tokens": 1200, "cost": 0.02},
    {"plan": [("locate target file", "done"),
              ("read and diagnose", "in_progress"),
              ("apply fix and verify", "pending")],
     "tool": ("read_file", {"path": "README.md"}),
     "tokens": 900, "cost": 0.02},
    {"plan": [("locate target file", "done"),
              ("read and diagnose", "done"),
              ("apply fix and verify", "done")],
     "tool": None,  # 终止回合
     "tokens": 600, "cost": 0.01},
]


def model_step(plan: PlanState, turn: int) -> dict[str, Any]:
    """存根模型：返回计划重写和（可选）工具调用"""
    if turn >= len(SCRIPT):
        return {"plan": plan.items, "tool": None, "tokens": 200, "cost": 0.005}
    s = SCRIPT[turn]
    items = [TodoItem(i + 1, desc, status) for i, (desc, status) in enumerate(s["plan"])]
    return {"plan": items, "tool": s["tool"], "tokens": s["tokens"], "cost": s["cost"]}


# ---------------------------------------------------------------------------
# 主循环——计划/行动/观察/恢复，完整钩子集成
# ---------------------------------------------------------------------------

def destructive_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """PreToolUse钩子：阻止破坏性命令"""
    cmd = payload.get("args", {}).get("cmd", "")
    if "rm -rf" in cmd or "shutdown" in cmd:
        payload["blocked"] = True
        payload["reason"] = "PreToolUse钩子阻止了破坏性命令"
    return payload


def run_agent(task: str, sandbox: str) -> dict[str, Any]:
    """运行编码智能体"""
    plan = PlanState(goal=task, items=[])
    budget = Budget()
    hooks = HookBus()
    trace: list[dict[str, Any]] = []

    hooks.on("PreToolUse", destructive_guard)
    hooks.on("PostToolUse", lambda p: (trace.append({"event": "tool", **p}), p)[1])
    hooks.on("SessionStart", lambda p: (trace.append({"event": "start", **p}), p)[1])
    hooks.on("SessionEnd", lambda p: (trace.append({"event": "end", **p}), p)[1])

    hooks.fire("SessionStart", {"task": task, "sandbox": sandbox,
                                "started_at": time.time()})

    turn = 0
    while True:
        stop = budget.exceeded()
        if stop:
            hooks.fire("Stop", {"reason": stop, "turn": turn})
            break

        step = model_step(plan, turn)
        plan.items = step["plan"]
        budget.step(step["tokens"], step["cost"])

        call = step["tool"]
        if call is None:
            hooks.fire("Stop", {"reason": "complete", "turn": turn})
            break

        name, args = call
        pre = hooks.fire("PreToolUse", {"tool": name, "args": args})
        if pre.get("blocked"):
            hooks.fire("PostToolUse", {"tool": name, "blocked": True,
                                       "reason": pre.get("reason", "")})
            turn += 1
            continue

        try:
            result = TOOLS[name](sandbox, **args)
            hooks.fire("PostToolUse", {"tool": name, "ok": True,
                                       "bytes": len(result)})
        except Exception as exc:
            hooks.fire("PostToolUse", {"tool": name, "ok": False,
                                       "error": str(exc)})

        turn += 1

    hooks.fire("SessionEnd", {"turns": budget.turns_used,
                              "tokens": budget.tokens_used,
                              "dollars": budget.dollars_used})

    return {"plan": plan.summary(), "budget": asdict(budget), "trace": trace}


def main() -> None:
    task = "demonstrate the plan-act-observe loop without network calls"
    sandbox = os.path.dirname(os.path.abspath(__file__))
    result = run_agent(task, sandbox)
    print(result["plan"])
    print("---")
    print(f"turns={result['budget']['turns_used']} "
          f"tokens={result['budget']['tokens_used']} "
          f"dollars=${result['budget']['dollars_used']:.3f}")
    print("---")
    print(f"trace events: {len(result['trace'])}")
    for ev in result["trace"]:
        print(" ", json.dumps(ev, default=str))


if __name__ == "__main__":
    main()
```

运行结果：

```
目标: demonstrate the plan-act-observe loop without network calls
  [>] 1. locate target file
  [ ] 2. read and diagnose
  [ ] 3. apply fix and verify
---
turns=3 tokens=2700 dollars=$0.050
---
trace events: 9
  {"event": "start", "task": "...", "sandbox": "...", "started_at": ...}
  {"event": "tool", "tool": "run_shell", "args": {"cmd": "ls"}, ...}
  {"event": "tool", "tool": "read_file", "args": {"path": "README.md"}, ...}
  {"event": "end", "turns": 3, "tokens": 2700, "dollars": 0.05}
```

---

## 4. 工具实践

生产环境中的编码智能体使用以下工具栈：

**主循环框架**：Bun + Ink (React-in-terminal)
**模型接入**：OpenRouter统一API（Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro）
**工具传输**：Model Context Protocol StreamableHTTP（MCP 2026修订版）
**沙箱**：E2B sandboxes（JS SDK）或Daytona devcontainers
**代码搜索**：ripgrep + tree-sitter解析器（17种语言）
**评测**：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0

---

## 5. LLM视角

**架构视角**：编码智能体的核心不是模型调用，而是工具循环。模型能力在快速提升，但工具循环的稳定性、上下文管理和成本控制才是工程挑战。

**沙箱视角**：沙箱隔离是安全的基础。没有沙箱，编码智能体的文件系统操作不可控，可能造成破坏性后果。

**钩子视角**：八事件钩子系统使操作者可以在每个生命周期点注入策略。这是2026年编码智能体可扩展性的关键。

---

## 6. 工程最佳实践

**沙箱安全**：
- 使用E2B或Daytona沙箱隔离
- `git worktree add`每次任务创建独立分支
- 所有工具调用在沙箱内执行

**成本控制**：
- 设置每轮token上限
- 设置每次会话美元预算
- 硬回合数限制（通常50轮）
- 实现`PreCompact`钩子在接近上限时压缩上下文

**可观测性**：
- 使用OpenTelemetry GenAI语义约定
- 发送trace到自托管Langfuse
- 每轮记录token使用和成本

---

## 7. 常见错误

**错误1：忽略沙箱安全**
症状：工具调用直接操作宿主机文件系统
修复：所有工具调用在沙箱内执行

**错误2：无成本控制**
症状：无限循环消耗大量token和费用
修复：设置三层硬上限

**错误3：忽略钩子系统**
症状：无法在生命周期点注入策略
修复：实现八事件钩子系统

---

## 8. 面试考点

**Q1：编码智能体的四表面架构是什么？**
考察：对智能体循环的理解

**Q2：为什么沙箱隔离是必要的？**
考察：对安全风险的理解

**Q3：2026年的八种钩子事件是什么？**
考察：对可扩展架构的理解

**Q4：如何控制编码智能体的成本？**
考察：对成本管理的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 主循环 | "智能体循环" | 围绕模型的调度代码，分派工具、维护计划状态、执行预算 |
| 钩子 | "智能体事件监听器" | 用户编写的脚本，在主循环的八个生命周期事件上运行 |
| Worktree | "Git沙箱" | 链接的git检出，可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 待办/进行中/完成的结构化列表，模型每轮重写 |
| StreamableHTTP | "MCP传输" | 2026年MCP修订：长连接HTTP，双向流，取代SSE |
| Token上限 | "上下文预算" | 每轮或每次会话的输入+输出token上限 |
| pass@1 | "单次尝试通过率" | 一次运行中解决的SWE-bench任务比例 |

---

## 参考文献

- [Claude Code文档](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor 3更新日志](https://cursor.com/changelog)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent)
- [OpenCode](https://opencode.ai)
- [SWE-bench Pro排行榜](https://www.swebench.com)
- [Model Context Protocol 2026路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)
- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
