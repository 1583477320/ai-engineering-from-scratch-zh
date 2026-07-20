# 综合项目56——迭代调度器（Iteration Scheduler）

> 没有调度器的研究循环是一个有妄想症的队列。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 建模为假设队列 + 并行槽位 + 结果汇合
- 用 UCB1 评分分支并修剪低产分支
- 高产出结果扇出到论文写作和后续假设

---

## 1. 问题

扁平工作列表按提交顺序运行。研究不是独立的——实验 3 的发现改变实验 4 的优先级。读取结果并重排序队列的调度器获得更多有用计算。

---

## 2. 核心概念

### 2.1 UCB1

```
ucb(branch) = mean_reward(branch) + sqrt(2) × sqrt(ln(total_runs) / runs(branch))
```

未尝试分支返回 `+inf`。

### 2.2 修剪门

平均奖励 < 0.2 且 ≥ 3 次试验后移除。

### 2.3 扇出

平均奖励 ≥ 0.7 时触发论文写作和后续假设生成。

---

## 3. 从零实现

```python
"""迭代调度器——UCB+并行+修剪+扇出。"""
from __future__ import annotations
import asyncio, math, random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class Hypothesis:
    id: int; branch: str; payload: Dict[str,Any] = field(default_factory=dict)
@dataclass
class Result:
    hypothesis_id: int; branch: str; reward: float; metrics: Dict[str,Any] = field(default_factory=dict)
@dataclass
class BranchStats:
    runs: int = 0; reward_sum: float = 0.0
    @property
    def mean(self): return self.reward_sum/self.runs if self.runs>0 else 0.0
@dataclass
class SchedulerReport:
    per_branch: Dict[str,BranchStats]; total_runs: int; paper_triggers: List[str]
    stop_reason: str; trace: List[Dict] = field(default_factory=list)

class IterationScheduler:
    def __init__(self, runner: Callable, expander: Optional[Callable]=None,
                 n_slots=3, c=math.sqrt(2), pt=0.7, pf=0.2, pa=3, me=50, ms=120):
        self.runner=runner; self.expander=expander; self.n_slots=n_slots; self.c=c
        self.paper_threshold=pt; self.prune_floor=pf; self.prune_after=pa
        self.max_experiments=me; self.max_seconds=ms

    def _ucb(self, br, stats, total):
        s=stats.get(br)
        if s is None or s.runs==0: return float("inf")
        if total<=1: return s.mean
        return s.mean + self.c*math.sqrt(math.log(total)/s.runs)

    async def run(self, seeds: List[Hypothesis]) -> SchedulerReport:
        queue=list(seeds); stats={}; triggers=[]; trace=[]; fired=set()
        total_runs=0; start=asyncio.get_event_loop().time()
        in_flight=set()
        while True:
            elapsed=asyncio.get_event_loop().time()-start
            if total_runs>=self.max_experiments or elapsed>=self.max_seconds:
                for t in in_flight: t.cancel(); break
            while len(in_flight)<self.n_slots:
                if not queue: break
                tot=max(1,sum(s.runs for s in stats.values()))
                queue.sort(key=lambda h:-self._ucb(h.branch,stats,tot))
                hyp=queue.pop(0)
                s=stats.get(hyp.branch)
                if s and s.runs>=self.prune_after and s.mean<self.prune_floor:
                    trace.append({"event":"prune","branch":hyp.branch}); continue
                task=asyncio.create_task(self.runner(hyp)); in_flight.add(task)
                trace.append({"event":"dispatch","branch":hyp.branch})
            if not in_flight: break
            done,in_flight=await asyncio.wait(in_flight,return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                r=t.result(); total_runs+=1
                if r.branch not in stats: stats[r.branch]=BranchStats()
                stats[r.branch].runs+=1; stats[r.branch].reward_sum+=r.reward
                trace.append({"event":"result","branch":r.branch,"reward":r.reward})
                if r.branch not in fired and stats[r.branch].mean>=self.paper_threshold:
                    triggers.append(r.branch); fired.add(r.branch)
                    trace.append({"event":"trigger","branch":r.branch})
                if self.expander and r.reward>=self.paper_threshold:
                    for h in self.expander(r): queue.append(h)
        stop="queue_empty" if not queue else "max_experiments" if total_runs>=self.max_experiments else "deadline"
        return SchedulerReport(stats,total_runs,triggers,stop,trace)

async def demo_runner(hyp):
    await asyncio.sleep(0.01); return Result(hyp.id, hyp.branch, min(1,max(0,random.gauss(0.5,0.2))))

def main():
    seeds=[Hypothesis(i,f"branch_{chr(65+i)}") for i in range(5)]
    r=asyncio.run(IterationScheduler(demo_runner,max_experiments=12).run(seeds))
    print(f"停止: {r.stop_reason}  实验: {r.total_runs}")
    for br,s in r.per_branch.items(): print(f"  {br}: {s.runs}次 平均{s.mean:.3f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
```

---

## 4. 工业工具

| 系统 | 调度算法 | 并行执行 | 持久状态 |
|:----|:--------|:--------|:--------|
| Airflow | DAG 依赖 | 内建 | 数据库 |
| Ray | UCB/Thompson | 内建 | GCS |
| Optuna | TPE/CMA-ES | 可选 | 数据库 |

---

## 5. 工程最佳实践

- 未尝试分支总是优先（runs=0 → +inf）
- 平台检测需两轮
- **中文场景建议**：分支 ID 使用字母数字避免编码问题

---

## 6. 常见错误

- **c 值不当**：c<0.5 缺乏探索，c>3 忽视利用。`sqrt(2)` 是良好起点
- **扩展器无限循环**：用 `max_experiments` 总预算兜底

---

## 7. 面试考点

**Q1：UCB 如何平衡探索与利用？**（难度：⭐⭐⭐）

**参考答案：** UCB 的均值项（利用）和不确定性项（探索）相加。不确定性项随 `runs(branch)` 增加递减，已充分探索的分支逐渐被均值主导。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| UCB1 | 上置信界探索与利用平衡 |
| 并行槽位 | asyncio 驱动的并发实验 |
| 修剪门 | 低产分支停止条件 |

---

## 📚 小结

迭代调度器将假设队列转化为并行探索-利用循环。下一节演示端到端集成。

---

## ✏️ 练习

1. 【实现】为 UCB 添加 `tie_break="random"` 参数
2. 【实验】用不同 `c` 值（0.5, 1.4, 3.0）测试分支探索比例

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 调度器 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Auer et al. "Finite-time Analysis of the Multiarmed Bandit Problem". 2002.
