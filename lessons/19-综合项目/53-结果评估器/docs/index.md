# 综合项目53——结果评估器（Result Evaluator）

> 运行器产生了数字。评估器决定这些数字是改进、回归还是噪音。构建将指标转化为一行结论的裁决路径。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 使用方向感知改进和固定阈值比较候选运行与基线
- 从零运行配对 t 检验，读取得到的 p 值
- 归一化对数尺度指标以便与线性指标混合
- 每条假设发出一个编排器可附加到队列的裁决
- 保持每一步纯函数化，相同输入始终产生相同裁决

---

## 1. 问题

来自运行器的单一数字不能说明变化是否真实。相同配置的不同种子给出不同的困惑度。变化可能是噪音。正确的比较是配对的：相同种子、相同数据、候选和基线各运行一次。每个种子贡献一个差异值的平均值是效应大小，这些差异的标准误是噪音基底。

---

## 2. 核心概念

### 2.1 配对 t 检验

```text
diffs    = [a_i - b_i for i in seeds]
mean     = sum(diffs) / n
variance = sum((d - mean)^2 for d in diffs) / (n - 1)
t_stat   = mean / sqrt(variance / n)
df       = n - 1
p_value  = two_sided_p(t_stat, df)
```

双侧 p 值使用正则化不完全 beta 函数。课程使用 Lentz 连分式实现，约 60 行标准库数学代码。

### 2.2 方向感知改进

```text
if direction == "higher_is_better":
    improvement = (candidate - baseline) / abs(baseline)
elif direction == "lower_is_better":
    improvement = (baseline - candidate) / abs(baseline)
```

### 2.3 裁决路径

```
1. 如果任何候选结果的 terminal != "ok": 裁决 = "failed"
2. 如果 |improvement| < improvement_threshold:  裁决 = "noise"
3. 如果 p_value 为 None 或 p_value > significance: 裁决 = "noise"
4. 如果 improvement > 0:                          裁决 = "improved"
5. 否则:                                          裁决 = "regressed"
```

---

## 3. 从零实现

```python
"""结果评估器——配对 t 检验+方向改进+裁决。"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricSpec:
    name: str; direction: str = "lower_is_better"  # or "higher_is_better"
    scale: str = "linear"  # or "log"


@dataclass
class Verdict:
    hypothesis_id: int; metric: str; direction: str; scale: str
    candidate_mean: float; baseline_mean: float
    improvement: float
    p_value: Optional[float]
    significance_threshold: float; improvement_threshold: float
    verdict: str  # "improved" | "regressed" | "noise" | "failed"
    rationale: str


def regularized_beta(x: float, a: float, b: float) -> float:
    """正则化不完全 beta 函数——Lentz 连分式。"""
    if x < 0 or x > 1:
        raise ValueError(f"x must be in [0,1], got {x}")
    if a <= 0 or b <= 0:
        raise ValueError(f"a,b must be positive, got {a},{b}")
    if x == 0 or x == 1:
        return 0.0
    ft = 1.0
    ab = a + b
    d = 1.0 - ab * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        fm = m
        num = fm * (b - m) * x / ((a + 2 * fm - 1) * (a + 2 * fm))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / c if (c := 1.0 + num / (h if m == 1 else c)) is not None else 1.0
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h * math.exp(a * math.log(x) + b * math.log(1 - x)) / (a * math.beta(a, b))


def two_sided_p(t_stat: float, df: int) -> float:
    """双侧 t 检验 p 值。"""
    if df <= 0 or not math.isfinite(t_stat):
        return 1.0
    x = df / (df + t_stat * t_stat)
    p = regularized_beta(x, df / 2, 0.5)
    return min(p, 1 - p) * 2 if t_stat >= 0 else min(p, 1 - p) * 2


class Evaluator:
    def evaluate(self, hypothesis_id: int, metric: MetricSpec,
                 candidates: List[Dict], baselines: List[Dict]) -> Verdict:
        if any(r.get("terminal", "ok") != "ok" for r in candidates):
            return Verdict(hypothesis_id, metric.name, metric.direction, metric.scale,
                           0.0, 0.0, 0.0, None, 0.05, 0.02, "failed", "候选实验运行失败")

        c_vals = [r["metrics"][metric.name] for r in candidates if metric.name in r.get("metrics", {})]
        b_vals = [r["metrics"][metric.name] for r in baselines if metric.name in r.get("metrics", {})]
        if not c_vals or not b_vals:
            return Verdict(hypothesis_id, metric.name, metric.direction, metric.scale,
                           0.0, 0.0, 0.0, None, 0.05, 0.02, "noise", "指标数据不足")

        c_mean = sum(c_vals) / len(c_vals)
        b_mean = sum(b_vals) / len(b_vals)

        def transform(v): return math.log(v) if metric.scale == "log" else v
        c_t, b_t = transform(c_mean), transform(b_mean)

        if metric.direction == "higher_is_better":
            improvement = (c_t - b_t) / abs(b_t) if b_t != 0 else 0.0
        else:
            improvement = (b_t - c_t) / abs(b_t) if b_t != 0 else 0.0

        if len(c_vals) < 2 or len(b_vals) < 2:
            p_val = None
        else:
            diffs = [transform(c) - transform(b) for c, b in zip(c_vals, b_vals)]
            n = len(diffs)
            mean_d = sum(diffs) / n
            var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
            t_stat = mean_d / math.sqrt(var_d / n) if var_d > 0 else 0.0
            p_val = two_sided_p(t_stat, n - 1) if n > 1 else None

        if abs(improvement) < 0.02:
            verdict, rationale = "noise", "改进幅度低于阈值"
        elif p_val is not None and p_val > 0.05:
            verdict, rationale = "noise", f"p 值 {p_val:.4f} 超过显著性阈值"
        elif improvement > 0:
            verdict, rationale = "improved", f"改进 {improvement:.2%}"
        else:
            verdict, rationale = "regressed", f"回归 {improvement:.2%}"

        return Verdict(hypothesis_id, metric.name, metric.direction, metric.scale,
                       c_mean, b_mean, improvement, p_val, 0.05, 0.02, verdict, rationale)


def main():
    cand = [{"metrics": {"loss": 0.52}}, {"metrics": {"loss": 0.51}}, {"metrics": {"loss": 0.49}}]
    base = [{"metrics": {"loss": 0.55}}, {"metrics": {"loss": 0.54}}, {"metrics": {"loss": 0.53}}]
    eval = Evaluator()
    v = eval.evaluate(1, MetricSpec("loss"), cand, base)
    print(f"裁决: {v.verdict}")
    print(f"改进: {v.improvement:.2%}")
    print(f"p 值: {v.p_value:.4f}" if v.p_value else "p 值: N/A")
    print(f"理由: {v.rationale}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 配对 t 检验 | 相同种子下的成对比较，消除随机初始化方差 |
| 方向感知改进 | 指标方向感知（越高越好 vs 越低越好） |
| 对数归一化 | 对困惑度等指数尺度指标取对数再比较 |
| 裁决路径 | 将 p 值、改进幅度和终端状态组合为最终结论的决策表 |
| 改进阈值 | 忽视低于此阈值的微小变化（默认 2%） |

---

## 5. 工程最佳实践

- **始终报告 p 值**：即使裁决为"noise"，p 值也是下游分析的重要信息。
- **改进阈值防止过度敏感**：N=100 时，即使 0.1% 的变化也会达到统计显著——但在实践中毫无意义。
- **中文场景特别建议**：记录裁决理由时使用中文，便于非英语使用者理解评估结果。

---

## 6. 常见错误

- **种子未配对**：候选和基线的种子必须一一对应，否则 t 检验无效。
- **未处理对数尺度**：困惑度是指数尺度——不取对数时改进百分比被放大。
- **样本量不足**：少于 3 个种子时 p 值不可靠，裁决应降级为"noise"。

---

## 📖 参考资料

1. [论文] Student. "The Probable Error of a Mean". Biometrika, 1908.
2. [数学] 正则化不完全 beta 函数连分式实现. https://en.wikipedia.org/wiki/Beta_function
