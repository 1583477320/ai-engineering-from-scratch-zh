"""并行 AAR 论坛模拟器——纯标准库。

三个自动对齐研究员并行运行。每个在两种模式之一下解决研究任务：
固定流程（人类规定计划）或自由分解。发现发布到追加不可变论坛，
记录存在于智能体沙箱之外。

一个智能体尝试日志篡改。防篡改链在验证时捕获该尝试。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


# ── 论坛和哈希链 ──────────────────────────────────────────

@dataclass
class ForumRecord:
    author: str
    task: str
    regime: str
    result: float
    prev_hash: str
    my_hash: str = ""


@dataclass
class Forum:
    records: list[ForumRecord] = field(default_factory=list)
    genesis: str = "0" * 16

    def head(self) -> str:
        return self.records[-1].my_hash if self.records else self.genesis

    def post(self, rec: ForumRecord) -> None:
        """追加记录并计算哈希链——追加不可变。"""
        rec.prev_hash = self.head()
        payload = f"{rec.author}|{rec.task}|{rec.regime}|{rec.result:.3f}|{rec.prev_hash}"
        rec.my_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.records.append(rec)

    def verify(self) -> list[int]:
        """返回任何被篡改的记录索引。"""
        bad = []
        prev = self.genesis
        for i, r in enumerate(self.records):
            payload = f"{r.author}|{r.task}|{r.regime}|{r.result:.3f}|{prev}"
            expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
            if expected != r.my_hash or r.prev_hash != prev:
                bad.append(i)
            prev = r.my_hash
        return bad


# ── 任务和求解器 ──────────────────────────────────────────

TASKS = [
    ("weak-to-strong-distill", 0.40),
    ("reward-model-diagnosis", 0.30),
    ("in-context-safety-probe", 0.50),
    ("RLHF-hyperparam-sweep", 0.45),
    ("alignment-faking-detector", 0.25),
]


def solve(agent: str, task: tuple[str, float], regime: str) -> float:
    """模拟智能体求解任务。"""
    _name, base = task
    if regime == "fixed":
        # 规定流程：有界上限，低方差
        return base + random.random() * 0.25
    # 自由分解：更宽尾部；有时好得多，有时更差
    return base + random.gauss(0.15, 0.22)


def run_regime(regime: str) -> list[ForumRecord]:
    """运行一种模式下的所有任务。"""
    forum = Forum()
    agents = ["AAR-A", "AAR-B", "AAR-C"]
    for task in TASKS:
        for agent in agents:
            r = solve(agent, task, regime)
            forum.post(ForumRecord(
                author=agent, task=task[0], regime=regime,
                result=r, prev_hash="",
            ))
    return forum.records


# ── 篡改检测 ──────────────────────────────────────────────

def attempt_tamper(forum_records: list[ForumRecord]) -> Forum:
    """重建论坛并尝试静默编辑 AAR-A 的最差结果。"""
    f = Forum()
    for r in forum_records:
        f.post(ForumRecord(
            author=r.author, task=r.task, regime=r.regime,
            result=r.result, prev_hash="",
        ))
    worst_idx = -1
    worst_val = float("inf")
    for i, r in enumerate(f.records):
        if r.author == "AAR-A" and r.result < worst_val:
            worst_val = r.result
            worst_idx = i
    if worst_idx >= 0:
        f.records[worst_idx].result += 0.5  # 静默编辑
    return f


# ── 报告 ──────────────────────────────────────────────────

def regime_report(regime: str) -> None:
    records = run_regime(regime)
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(r.result)
    print(f"\n模式: {regime}")
    print(f"  {'任务':<32}  {'均值':>8}  {'最大':>8}  {'最小':>8}")
    means = []
    for task, values in by_task.items():
        mean = sum(values) / len(values)
        means.append(mean)
        print(f"  {task:<32}  {mean:>8.3f}  {max(values):>8.3f}  {min(values):>8.3f}")
    print(f"  跨任务总均值              : {sum(means) / len(means):.3f}")


def tamper_demo() -> None:
    """演示篡改检测。"""
    print("\n日志篡改检测")
    print("-" * 70)
    baseline = run_regime("free")
    tampered = attempt_tamper(baseline)
    bad = tampered.verify()
    print(f"  verify() 标记了 {len(bad)} 条损坏记录，索引 {bad}")
    print("  追加不可变链在第一条编辑记录处断裂。")
    print("  沙箱外备份通过交叉检查揭示篡改。")


# ── 主函数 ────────────────────────────────────────────────

def main() -> None:
    random.seed(3)
    print("=" * 70)
    print("自动化对齐研究论坛（阶段 15，第 6 课）")
    print("=" * 70)

    regime_report("fixed")
    regime_report("free")

    print("\n固定 vs 自由——固定方差更低，自由上限更高")
    tamper_demo()

    print()
    print("=" * 70)
    print("要点: 沙箱外日志 + 自由分解 + 人类审查")
    print("-" * 70)
    print("  自由分解给 AAR 更高上限——以及更多审计工作。")
    print("  防篡改日志使审计成为可能。人类审查员仍然决定")
    print("  发表什么。AAR 压缩管道的中间部分，不是两端。")


if __name__ == "__main__":
    main()
