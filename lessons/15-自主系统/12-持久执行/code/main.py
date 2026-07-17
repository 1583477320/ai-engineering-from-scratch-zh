"""最小持久执行引擎——纯标准库。

建模 Temporal、LangGraph 检查点、Microsoft Agent Framework
和 Claude Code Routines 使用的工作流/活动/事件日志模式。

活动在执行前记录输入，执行后记录输出。
工作流重放运行时重新执行工作流代码但返回已完成活动的缓存输出。
崩溃中运行时只丢失未完成的活动。

运行：python3 code/main.py
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from dataclasses import dataclass


# ── 事件日志 ──────────────────────────────────────────────

@dataclass
class EventLog:
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
        for ev in self.events():
            if ev["name"] == name and ev["args"] == list(args) and ev["status"] == "done":
                return ev
        return None


# ── 活动装饰器 ──────────────────────────────────────────

def activity(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(log: EventLog, *args):
            hit = log.lookup(name, args)
            if hit:
                print(f"    [重放] {name}({args}) -> {hit['result']} (来自日志)")
                return hit["result"]
            log.append({"name": name, "args": list(args), "status": "started"})
            result = fn(*args)
            log.append({"name": name, "args": list(args), "status": "done", "result": result})
            print(f"    [运行] {name}({args}) -> {result}")
            return result
        return wrapper
    return deco


# ── 示例活动 ──────────────────────────────────────────────

@activity("fetch_docs")
def fetch_docs(query: str) -> int:
    return len(query) * 3

@activity("call_llm")
def call_llm(doc_count: int) -> str:
    return f"summary({doc_count}_docs)"

@activity("write_report")
def write_report(summary: str) -> str:
    return f"report://{summary}"


# ── 工作流 ────────────────────────────────────────────────

def workflow(log: EventLog, query: str, crash_after: int = -1) -> str:
    """三步活动工作流，带可选的崩溃演示。"""
    doc_count = fetch_docs(log, query)
    if crash_after == 1:
        raise RuntimeError("simulated crash after fetch_docs")
    summary = call_llm(log, doc_count)
    if crash_after == 2:
        raise RuntimeError("simulated crash after call_llm")
    report = write_report(log, summary)
    return report


# ── 驱动 ──────────────────────────────────────────────────

def reset_log(path: str) -> EventLog:
    if os.path.exists(path):
        os.remove(path)
    return EventLog(path)


def count_runs(log: EventLog) -> int:
    return sum(1 for ev in log.events() if ev["status"] == "started")


def main() -> None:
    print("=" * 70)
    print("持久执行（阶段 15，第 12 课）")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()

    # 朴素重试：崩溃时丢失事件日志。每次重启重新运行一切。
    print("\n朴素重试（事件日志未持久化）")
    print("-" * 70)
    for attempt in range(1, 4):
        log = reset_log(os.path.join(tmpdir, "naive.json"))
        print(f"  尝试 {attempt}:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            print(f"    -> 本次尝试 {count_runs(log)} 个活动开始")
            break
        except RuntimeError as e:
            print(f"    -> 崩溃: {e}; {count_runs(log)} 个活动开始被浪费")

    # 持久重试：事件日志跨尝试保留；重放不重新执行已完成活动。
    print("\n持久重试（事件日志跨尝试保留）")
    print("-" * 70)
    durable_path = os.path.join(tmpdir, "durable.json")
    if os.path.exists(durable_path):
        os.remove(durable_path)

    for attempt in range(1, 4):
        log = EventLog(durable_path)
        print(f"  尝试 {attempt}:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> 结果 {r}")
            print(f"    -> {count_runs(log)} 个总活动开始（跨尝试）")
            break
        except RuntimeError as e:
            print(f"    -> 崩溃: {e}")

    print()
    print("=" * 70)
    print("要点: 持久性使长期运行在失败时也负担得起")
    print("-" * 70)
    print("  朴素重试每次尝试重新执行每个活动。")
    print("  持久重试从日志重放已完成的活动；")
    print("  只有缺失的活动实际运行。与 Temporal、")
    print("  LangGraph 检查点、Microsoft Agent Framework")
    print("  和 Claude Code Routines 使用的设计相同。")
    print("  LLM 调用只是日志中的另一个非确定性活动。")


if __name__ == "__main__":
    main()
