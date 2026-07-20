# 综合项目74——排行榜聚合（Leaderboard Aggregation）

> 逐任务分数容易。跨异构任务的模型排名更难。千预测排行榜的统计显著性是大家跳过的部分。这节课不跳过。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第70-73节
**预计时间：** 90分钟

---

## 学习目标

- 将多模型多任务的逐任务分数聚合为每模型一行
- 归一化异构分数
- 按均值和胜率排名并解释各自适用场景
- 计算 bootstrap 置信区间
- 输出 JSON 和 markdown 排行榜

---

## 1. 问题

逐任务分数简单——每个任务一个数字。但排行榜需要跨任务汇总：模型 A 在数学上比 B 强，在代码上比 B 弱，综合谁更好？需要归一化、聚合、统计检验。

---

## 2. 核心概念

### 2.1 输入形状

```text
EvalRun(model_id, task_id, metric_name, score, category)
```

每个分数已在 [0, 1] 范围内。

### 2.2 均值 vs 胜率

- **均值**：每个模型跨任务的平均分——排行榜的头条数字
- **胜率**：每个任务中该模型击败所有其他模型的比例

### 2.3 Bootstrap 置信区间

对每模型任务分数有放回重采样 B 次，计算均值分布的百分位区间。

### 2.4 排行榜行

```text
LeaderboardRow(model_id, mean_score, ci_lo, ci_hi, win_rate, tasks_completed, categories)
```

---

## 3. 从零实现

```python
"""排行榜聚合——均值+胜率+bootstrap CI。"""
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EvalRun:
    model_id: str; task_id: str; metric_name: str; score: float; category: str = ""

@dataclass
class LeaderboardRow:
    model_id: str; mean_score: float; ci_lo: float; ci_hi: float
    win_rate: float; tasks_completed: int; categories: Dict[str, float] = field(default_factory=dict)


def bootstrap_mean_ci(scores, b=500, alpha=0.05):
    if not scores: return 0.0, 0.0, 0.0
    means = [sum(random.choices(scores, k=len(scores))) / len(scores) for _ in range(b)]
    means.sort()
    lo = means[int(b * alpha / 2)]
    hi = means[int(b * (1 - alpha / 2))]
    return sum(scores) / len(scores), lo, hi


def win_rate(model_id, runs_by_task, all_models):
    wins, total = 0, 0
    for task_id, runs in runs_by_task.items():
        scores = {r.model_id: r.score for r in runs}
        if model_id not in scores: continue
        total += 1
        best = max(scores[sid] for sid in all_models if sid in scores)
        if scores[model_id] >= best: wins += 1
    return wins / total if total else 0.0


def aggregate(runs: List[EvalRun], b=500) -> List[LeaderboardRow]:
    runs_by_model = defaultdict(list)
    for r in runs:
        runs_by_model[r.model_id].append(r)
    runs_by_task = defaultdict(list)
    for r in runs:
        runs_by_task[r.task_id].append(r)
    all_models = list(runs_by_model.keys())

    rows = []
    for mid, mruns in runs_by_model.items():
        scores = [r.score for r in mruns]
        mean, ci_lo, ci_hi = bootstrap_mean_ci(scores, b=b)
        wr = win_rate(mid, runs_by_task, all_models)
        cats = defaultdict(list)
        for r in mruns:
            if r.category: cats[r.category].append(r.score)
        cat_means = {c: sum(s)/len(s) for c, s in cats.items()} if cats else {}
        rows.append(LeaderboardRow(mid, mean, ci_lo, ci_hi, wr, len(mruns), cat_means))
    rows.sort(key=lambda r: -r.mean_score)
    return rows


def render_markdown(rows: List[LeaderboardRow]) -> str:
    lines = ["| Rank | Model | Mean | 95% CI | Win Rate | Tasks |",
             "|------|-------|------|--------|----------|-------|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r.model_id[:20]} | {r.mean_score:.3f} | "
                     f"{r.ci_lo:.3f}-{r.ci_hi:.3f} | {r.win_rate:.2f} | {r.tasks_completed} |")
    return "\n".join(lines)


def main():
    import random
    random.seed(42)
    models = ["gpt-4o", "claude-opus", "random"]
    tasks = [f"task_{i}" for i in range(20)]
    runs = []
    base_scores = {"gpt-4o": 0.78, "claude-opus": 0.75, "random": 0.10}
    for mid in models:
        for tid in tasks:
            score = min(1, max(0, base_scores[mid] + random.gauss(0, 0.1)))
            cat = random.choice(["math", "code", "reasoning"])
            runs.append(EvalRun(mid, tid, "exact_match", score, cat))

    rows = aggregate(runs, b=200)
    print(render_markdown(rows))
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | Bootstrap | 格式 | 特点 |
|:----|:---------|:-----|:-----|
| Open LLM Leaderboard | ✓ | HTML | HF 标准 |
| Chatbot Arena | ✓ | HTML | ELO 排名 |
| 本课 | ✓ | JSON+MD | 可审计 |

---

## 5. 工程最佳实践

- 所有分数必须在 [0, 1] 范围——验证器拒绝越界值
- Bootstrap 默认 500 次；生产用 1000 次
- **中文场景建议**：排行榜中文化，模型名保留英文

---

## 6. 常见错误

- **归一化遗漏**：分数超 [0,1] 范围导致均值偏差
- **小样本 bootstrap 无意义**：少于 5 个任务无法可靠估计 CI
- **胜率不处理平局**：平局时平分胜利

---

## 7. 面试考点

**Q1：均值和胜率排名何时产生不同结果？**（难度：⭐⭐⭐）

**参考答案：** 当模型 A 在 3 个任务上极高分、17 个任务上中等分，模型 B 在 20 个任务上都略低于 A 的均值但从未低于太多时——均值排名 A > B，但胜率可能 B > A（B 在更多任务上"赢"或平局）。均值对极端值敏感，胜率更鲁棒但丢失幅度信息。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 均值排名 | 所有任务的平均分排序 |
| 胜率排名 | 在多少任务上击败对手的比例 |
| Bootstrap CI | 有放回重采样的百分位置信区间 |
| 配对 CI | 任务 A-B 差异的 bootstrap 区间 |

---

## 📚 小结

排行榜聚合将多任务分数转化为可比较的模型排名。你实现了均值、胜率、bootstrap CI 和 markdown 渲染。下一节构建端到端评估运行器。

---

## ✏️ 练习

1. 【实现】添加类别权重：每个类别总权重为 1，权重在类别内平均
2. 【实验】对比 B=100 和 B=1000 的 bootstrap CI 宽度差异

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 排行榜聚合 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Buckley & Voorhees. "Evaluating Evaluation Measure Stability". SIGIR 2000.
2. [论文] Efron & Tibshirani. "An Introduction to the Bootstrap". 1993.
