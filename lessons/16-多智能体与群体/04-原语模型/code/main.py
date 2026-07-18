"""四多智能体原语——纯标准库。

四个原语：智能体（提示词+工具）、移交、共享状态、编排者。
在三种编排类型上运行相同流水线（研究→写作→审查）。
智能体是脚本化策略，不是 LLM 调用——重点在协调结构。

运行：python3 code/main.py
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


Message = dict


# ── 共享状态 ──────────────────────────────────────────────

@dataclass
class SharedState:
    """线程安全的消息池——多智能体系统中唯一有状态的部分。"""
    messages: list[Message] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, msg: Message) -> None:
        with self._lock:
            self.messages.append(msg)

    def snapshot(self) -> list[Message]:
        with self._lock:
            return list(self.messages)

    def last_by(self, name: str) -> Optional[Message]:
        with self._lock:
            for m in reversed(self.messages):
                if m["from"] == name:
                    return m
            return None


# ── 智能体 ────────────────────────────────────────────────

@dataclass
class Agent:
    """无状态智能体——提示词 + 策略函数。"""
    name: str
    system_prompt: str
    policy: Callable[[SharedState], Message]

    def run(self, state: SharedState) -> Message:
        msg = self.policy(state)
        msg.setdefault("from", self.name)
        return msg


def researcher_policy(state: SharedState) -> Message:
    n = len([m for m in state.snapshot() if m["from"] == "researcher"])
    notes = f"note {n + 1}: FIPA-ACL 于 2000 年批准；20 种言语行为。"
    return {"content": notes, "handoff": "writer" if n == 0 else "done"}


def writer_policy(state: SharedState) -> Message:
    research = [m["content"] for m in state.snapshot() if m["from"] == "researcher"]
    draft = "草稿摘要: " + " | ".join(research) if research else "无研究的草稿。"
    return {"content": draft, "handoff": "reviewer"}


def reviewer_policy(state: SharedState) -> Message:
    last = state.last_by("writer")
    verdict = "approved" if last and "摘要" in last["content"] else "需要修订"
    return {"content": f"审查结论: {verdict}。", "handoff": "done"}


def make_team() -> dict[str, Agent]:
    return {
        "researcher": Agent("researcher", "收集事实。", researcher_policy),
        "writer": Agent("writer", "基于研究起草。", writer_policy),
        "reviewer": Agent("reviewer", "审查草稿。", reviewer_policy),
    }


# ── 三种编排者 ────────────────────────────────────────────

class StaticOrchestrator:
    """固定顺序——LangGraph 式确定性边。"""
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        for name in self.order[:max_steps]:
            msg = team[name].run(state)
            state.append(msg)


class HandoffOrchestrator:
    """OpenAI Swarm 式：当前智能体返回移交目标。"""
    def __init__(self, start: str) -> None:
        self.start = start

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        current = self.start
        for _ in range(max_steps):
            if current not in team:
                return
            msg = team[current].run(state)
            state.append(msg)
            nxt = msg.get("handoff", "done")
            if nxt == "done":
                return
            current = nxt


class LLMSelectorOrchestrator:
    """AutoGen GroupChat 式说话者选择。"""
    def __init__(self, start: str, selector: Callable) -> None:
        self.start = start
        self.selector = selector

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        current = self.start
        for _ in range(max_steps):
            if current not in team:
                return
            msg = team[current].run(state)
            state.append(msg)
            current = self.selector(state, team)


def round_robin_selector(state: SharedState, team: dict[str, Agent]) -> Optional[str]:
    """轮询选择——AutoGen GroupChat 式。"""
    if not state.messages:
        return None
    last = state.messages[-1]["from"]
    names = list(team.keys())
    idx = (names.index(last) + 1) % len(names)
    if len([m for m in state.messages if m["from"] == "reviewer"]) >= 1:
        return None
    return names[idx]


# ── 主函数 ────────────────────────────────────────────────

def render_pool(label: str, state: SharedState) -> None:
    print(f"\n=== {label} ===")
    for i, m in enumerate(state.snapshot()):
        ho = f" -> {m['handoff']}" if "handoff" in m else ""
        print(f"  [{i}] {m['from']:10s} | {m['content']}{ho}")


def main() -> None:
    print("四多智能体原语演示")
    print("-" * 42)

    team = make_team()
    state_a = SharedState()
    StaticOrchestrator(["researcher", "writer", "reviewer"]).run(team, state_a)
    render_pool("静态（LangGraph 式）", state_a)

    team = make_team()
    state_b = SharedState()
    HandoffOrchestrator("researcher").run(team, state_b)
    render_pool("移交驱动（Swarm 式）", state_b)

    team = make_team()
    state_c = SharedState()
    LLMSelectorOrchestrator("researcher", round_robin_selector).run(team, state_c)
    render_pool("LLM 选择（AutoGen 式）", state_c)

    print("\n要点: 智能体和状态在所有运行中相同；")
    print("只有编排者选择改变谁何时发言。")


if __name__ == "__main__":
    main()
