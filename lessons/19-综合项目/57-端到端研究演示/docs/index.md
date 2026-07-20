# 综合项目57——端到端研究演示（End-to-End Research Demo）

> 演示是你之前编写的每个合约都必须组合的地方。如果有任何一个合约泄漏，演示就是抓住它的那一课。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第50-56节
**预计时间：** 90分钟

---

## 学习目标

- 端到端连接自动研究循环：假设种子、实验运行器、调度器、评审循环、论文写作器
- 通过纯 Python import 而非框架，组合来自四个课程的基元
- 运行循环到自终止结束，发出包含每个阶段产出的演示报告
- 保持演示确定性，使测试套件可以断言最终形状
- 当任何阶段的合约被破坏时暴露清晰的失败模式

---

## 1. 问题

五个独立课程（假设生成、实验运行、结果评估、评审循环、论文写作）各自工作良好。但将它们组合成一个连贯的自动研究循环会揭示集成问题：数据结构不匹配、错误处理不兼容、确定性保证泄漏。

演示存在的意义就是捕获这些问题。它不添加新功能——它证明五个课程可以组合。

---

## 2. 核心概念

### 2.1 组合结构

```mermaid
flowchart LR
    Seed[种子假设] --> Sched[迭代调度器]
    Sched --> Exp[实验运行器]
    Exp --> Bus[结果总线]
    Bus --> Sched
    Bus --> Trig[论文触发]
    Trig --> Pick[最佳结果选择器]
    Pick --> Critic[评审循环]
    Critic --> Writer[论文写作器]
    Writer --> Report[演示报告]
```

五个阶段。种子是三个假设。调度器在三个并行槽位上运行六个实验。总线报告一个或多个论文触发。选择器选择单个最佳结果。评审循环在基于该结果的草稿上迭代。论文写作器发出最终的 LaTeX、BibTeX 和清单。

### 2.2 import 而非复制

每个早期课程提供公共数据类和函数。演示通过 `sys.path` 调整导入它们。

### 2.3 失败模式

每个阶段要么成功，要么抛出类型化错误：

```text
调度器 ........ 返回 SchedulerReport（含 stop_reason）
最佳选择 ...... 无触发时抛出 NoTriggerError
评审循环 ...... 返回 LoopResult（含收敛状态）
论文写作器 .... 合约被破坏时抛出 PaperValidationError
```

任何阶段的失败都通过类型化异常使演示短路。

---

## 3. 从零实现

```python
"""端到端研究演示——组合 5 个研究循环课程。"""
from __future__ import annotations
import json, os, sys, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── 简化版本：内联前面课程的核心数据类 ──

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
    def mean(self): return self.reward_sum / self.runs if self.runs > 0 else 0.0

@dataclass
class SchedulerReport:
    per_branch: Dict[str, BranchStats]; total_runs: int
    paper_triggers: List[str]; stop_reason: str; trace: List[Dict] = field(default_factory=list)

@dataclass
class LoopResult:
    convergence: str; rounds: int; trace: List[Dict] = field(default_factory=list)

@dataclass
class DemoReport:
    scheduler_report: Dict; best_branch: str; best_reward: float
    critic_result: Dict; paper_manifest: Dict; stop_reason: str

class NoTriggerError(Exception): pass
class BestResultError(Exception): pass

# ── 简化的调度器 ──

def run_scheduler(seeds: List[Hypothesis]) -> SchedulerReport:
    stats: Dict[str, BranchStats] = {}
    triggers = []
    for i, h in enumerate(seeds):
        reward = min(1.0, max(0.0, random.gauss(0.6, 0.2)))
        if h.branch not in stats: stats[h.branch] = BranchStats()
        stats[h.branch].runs += 1; stats[h.branch].reward_sum += reward
        if stats[h.branch].mean >= 0.7:
            triggers.append(h.branch)
    return SchedulerReport(stats, len(seeds), triggers, "queue_empty")

# ── 最佳分支选择器 ──

def pick_best(triggers: List[str], stats: Dict[str, BranchStats]) -> str:
    if not triggers: raise NoTriggerError("无论文触发")
    best = max(triggers, key=lambda b: stats[b].mean)
    if stats[best].mean < 0.7: raise BestResultError("最佳分支奖励不足")
    return best

# ── 简化评审循环 ──

def run_critic() -> LoopResult:
    return LoopResult("target", 3, [{"round": 1, "scores": {"clarity": 7}}, {"round": 2, "scores": {"clarity": 8}}, {"round": 3, "scores": {"clarity": 9}}])

# ── 简化论文写作器 ──

def write_paper(branch: str, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "title": f"关于 {branch} 的实证研究",
        "sections": ["引言", "方法", "结果", "讨论"],
        "figures": [{"id": "fig_1", "caption": "实验结果"}],
        "citations": ["ref01", "ref02"],
    }
    tex = r"""\documentclass{article}
\begin{document}
\title{Empirical Study of """ + branch + r"""}
\maketitle
\section{引言}
本节介绍研究背景。
\end{document}"""
    with open(os.path.join(out_dir, "paper.tex"), "w") as f: f.write(tex)
    with open(os.path.join(out_dir, "references.bib"), "w") as f: f.write("@article{ref01,\n  title={A Reference},\n  year={2024}\n}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
    return manifest


def run_demo(out_dir: str = "/tmp/research_demo") -> DemoReport:
    random.seed(42)
    seeds = [Hypothesis(i, f"branch_{chr(65+i)}") for i in range(3)]
    sched = run_scheduler(seeds)
    if not sched.paper_triggers:
        raise NoTriggerError("调度器未产生论文触发")
    best = pick_best(sched.paper_triggers, sched.per_branch)
    critic = run_critic()
    manifest = write_paper(best, out_dir)
    return DemoReport(
        scheduler_report={"total_runs": sched.total_runs, "triggers": sched.paper_triggers},
        best_branch=best, best_reward=sched.per_branch[best].mean,
        critic_result={"convergence": critic.convergence, "rounds": critic.rounds},
        paper_manifest=manifest, stop_reason="completed",
    )


def main():
    report = run_demo()
    print(f"最佳分支: {report.best_branch} (奖励={report.best_reward:.3f})")
    print(f"调度: {report.scheduler_report}")
    print(f"评审: {report.critic_result}")
    print(f"论文章节: {report.paper_manifest['sections']}")
    print(f"停止原因: {report.stop_reason}")
    print("✓ 端到端演示完成")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 演示报告 | 包含所有阶段产出的统一报告 |
| 组合 | 五个独立课程通过 import 而非框架连接 |
| 类型化错误 | 每个阶段抛出特定异常，使调用者能区分故障 |
| 确定性种子 | 固定种子确保每次运行产出相同报告 |

---

## 5. 工程最佳实践

- **先导入再运行**：如果 import 失败（依赖缺失、路径错误），演示应在执行任何逻辑前快速失败。
- **临时目录隔离**：论文写作输出写入临时目录，演示完成后清理。
- **中文场景建议**：演示的最终输出使用中文，但中间阶段的数据结构保持英文键名。

---

## 6. 常见错误

- **未处理 import 错误**：`sys.path` 调整失败时导入静默失败。应使用 try/except 包装并输出清晰的诊断。
- **种子未跨阶段传递**：调度器、评审器和写作器使用不同的随机状态，破坏确定性。
- **演示不清理产物**：每次运行在磁盘上留下 LaTeX 文件，积累占用空间。

---

## 📖 参考资料

1. [GitHub] `sys.path` 导入技巧. https://docs.python.org/3/library/sys.html
2. [论文] 自动研究循环综述. "AI for Scientific Discovery". 2024.
