# 综合项目56——迭代调度器（Iteration Scheduler）

> 没有调度器的研究循环是一个有妄想症的队列。调度器是循环决定停止探索什么的地方，而这个决定就是整个游戏。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 将研究工作流建模为假设队列，平行实验槽位的结果汇合回来
- 使用 asyncio 并行运行多个实验，使调度器保持所有槽位繁忙
- 用 UCB（上置信界）为每个假设分支评分，使调度器能修剪低产分支
- 将完成的结果扇出到论文写作和重新入队阶段
- 输出每轮迹线，包含分支分数、槽位占用率和修剪决策

---

## 1. 问题

一个扁平的工作列表按提交顺序运行任务。每个任务独立时这是可以的。研究不是独立的——实验三的发现改变了实验四和实验五的优先级。一个读取结果汇合并重新排序队列的调度器，每单位计算能完成更多有用工作。

核心设计选择是评分规则。贪心评分器总是选择当前领先者，从不探索。均匀评分器从不利用。UCB（上置信界）是中间路径：利用领先者的同时为尝试较少的分支保留容量。

---

## 2. 核心概念

### 2.1 UCB1 公式

```text
ucb(branch) = mean_reward(branch) + c × sqrt(ln(total_runs) / runs(branch))
```

`total_runs` 是所有分支上完成的实验总数。`c` 是探索权重（默认 `sqrt(2)`）。运行次数为零的分支获得 `+inf`——未尝试的分支总是优先调度。

### 2.2 并行槽位

调度器用 `asyncio.create_task` 驱动实验。主循环通过 `asyncio.wait(FIRST_COMPLETED)` 等待完成，每次完成时触发评分更新。

### 2.3 修剪门

分支平均奖励低于 `0.2` 且至少运行了 `3` 次试验时，从未来调度中移除。

### 2.4 扇出

分支平均奖励超过 `0.7` 且该分支尚未产出论文时，调度器触发 `paper.trigger` 事件。

---

## 3. 从零实现

```python
"""迭代调度器——UCB 评分+并行槽位+修剪+扇出。"""
from __future__ import annotations
import asyncio, math, random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Hypothesis:
    id: int; branch: str; payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Result:
    hypothesis_id: int; branch: str; reward: float; metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BranchStats:
    runs: int = 0; reward_sum: float = 0.0
    @property
    def mean(self) -> float:
        return self.reward_sum / self.runs if self.runs > 0 else 0.0

@dataclass
class SchedulerReport:
    per_branch: Dict[str, BranchStats]; total_runs: int; paper_triggers: List[str]
    stop_reason: str; wall_time_s: float; trace: List[Dict]


class IterationScheduler:
    def __init__(self, runner: Callable, expander: Optional[Callable] = None,
                 n_slots: int = 3, c: float = math.sqrt(2),
                 paper_threshold: float = 0.7, prune_floor: float = 0.2,
                 prune_after: int = 3, max_experiments: int = 50, max_seconds: float = 120):
        self.runner = runner; self.expander = expander
        self.n_slots = n_slots; self.c = c
        self.paper_threshold = paper_threshold; self.prune_floor = prune_floor
        self.prune_after = prune_after; self.max_experiments = max_experiments
        self.max_seconds = max_seconds

    def _ucb(self, branch: str, stats: Dict[str, BranchStats], total: int) -> float:
        s = stats.get(branch)
        if s is None or s.runs == 0: return float("inf")
        if total <= 1: return s.mean
        return s.mean + self.c * math.sqrt(math.log(total) / s.runs)

    async def run(self, seed_hypotheses: List[Hypothesis]) -> SchedulerReport:
        queue = list(seed_hypotheses)
        stats: Dict[str, BranchStats] = {}
        paper_triggers: List[str] = []
        trace: List[Dict] = []
        paper_fired: set = set()
        total_runs = 0
        start = asyncio.get_event_loop().time()

        def get_hypothesis() -> Optional[Hypothesis]:
            if not queue: return None
            total = max(1, sum(s.runs for s in stats.values()))
            queue.sort(key=lambda h: -self._ucb(h.branch, stats, total))
            return queue.pop(0)

        in_flight: set = set()
        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if total_runs >= self.max_experiments or elapsed >= self.max_seconds:
                for t in in_flight: t.cancel()
                break

            while len(in_flight) < self.n_slots:
                hyp = get_hypothesis()
                if hyp is None: break
                if hyp.branch in stats and stats[hyp.branch].runs >= self.prune_after and stats[hyp.branch].mean < self.prune_floor:
                    trace.append({"event": "prune", "branch": hyp.branch})
                    continue
                task = asyncio.create_task(self.runner(hyp))
                in_flight.add(task)
                trace.append({"event": "dispatch", "hyp_id": hyp.id, "branch": hyp.branch})

            if not in_flight: break

            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                result: Result = t.result()
                total_runs += 1
                if result.branch not in stats:
                    stats[result.branch] = BranchStats()
                stats[result.branch].runs += 1
                stats[result.branch].reward_sum += result.reward
                trace.append({"event": "result", "branch": result.branch, "reward": result.reward})
                if result.branch not in paper_fired and stats[result.branch].mean >= self.paper_threshold:
                    paper_triggers.append(result.branch)
                    paper_fired.add(result.branch)
                    trace.append({"event": "paper_trigger", "branch": result.branch})
                if self.expander and result.reward >= self.paper_threshold:
                    for h in self.expander(result):
                        queue.append(h)

        stop = "queue_empty" if not queue and not in_flight else \
               "max_experiments" if total_runs >= self.max_experiments else "deadline"
        return SchedulerReport(stats, total_runs, paper_triggers, stop,
                               asyncio.get_event_loop().time() - start, trace)


async def demo_runner(hyp: Hypothesis) -> Result:
    await asyncio.sleep(0.01)
    reward = min(1.0, max(0.0, random.gauss(0.5, 0.2)))
    return Result(hyp.id, hyp.branch, reward)


def main():
    seeds = [Hypothesis(i, f"branch_{chr(65+i)}", {"topic": f"topic_{i}"}) for i in range(5)]
    sched = IterationScheduler(demo_runner, max_experiments=12)
    report = asyncio.run(sched.run(seeds))
    print(f"停止原因: {report.stop_reason}")
    print(f"总实验数: {report.total_runs}")
    for branch, stat in report.per_branch.items():
        print(f"  {branch}: 运行={stat.runs} 平均奖励={stat.mean:.3f}")
    print(f"论文触发: {report.paper_triggers}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| UCB1 | 上置信界——平衡探索和利用的分支评分公式 |
| 并行槽位 | asyncio 驱动的并发实验运行槽 |
| 修剪门 | 低平均奖励分支的停止条件 |
| 扇出 | 完成结果触发论文写作和新假设入队 |
| 迹线 | 每个调度决策的日志事件 |

---

## 5. 工程最佳实践

- **未尝试分支总是优先**：UCB 中 `runs=0` 时返回 `+inf`，确保新分支立即被探索。
- **平台检测需要两轮**：一轮的低改进可能是偶然的标准化波动。
- **中文场景建议**：分支 ID 使用字母数字，避免中文字符在 asyncio 事件循环的日志中出现编码问题。

---

## 6. 常见错误

- **c 值设置不当**：c 太小（<0.5）导致缺乏探索；c 太大（>3）导致忽视利用。`sqrt(2)` 是经验上的好起点。
- **未处理实验运行器崩溃**：如果 `create_task` 中的协程序崩溃，异常被吞没。应在任务上添加 `add_done_callback` 捕获异常。
- **扇出无限循环**：扩展器可能产出无限多的后续假设。设置 `max_experiments` 总预算防止无限循环。

---

## 📖 参考资料

1. [论文] Auer et al. "Finite-time Analysis of the Multiarmed Bandit Problem". 2002.
2. [官方文档] Python `asyncio`. https://docs.python.org/3/library/asyncio.html
