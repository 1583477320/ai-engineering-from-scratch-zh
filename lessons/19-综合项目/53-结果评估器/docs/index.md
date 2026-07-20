# 综合项目53——结果评估器（Result Evaluator）

> 运行器产生了数字。评估器决定这些数字是改进、回归还是噪声。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 使用方向感知改进比较候选与基线
- 从零实现配对 t 检验
- 归一化对数尺度指标
- 发出可附加到假设的裁决

---

## 1. 问题

来自运行器的单一数字不能说明变化是否真实。相同配置的不同种子给出不同困惑度。正确比较是配对的——相同种子、相同数据、候选和基线各运行一次。

---

## 2. 核心概念

### 2.1 配对 t 检验

```
diffs  = [a_i - b_i for i in seeds]
mean   = sum(diffs) / n
var    = sum((d - mean)^2) / (n - 1)
t_stat = mean / sqrt(var / n)
p_val  = two_sided_p(t_stat, n - 1)
```

### 2.2 方向感知改进

```python
if direction == "higher_is_better":
    improvement = (candidate - baseline) / abs(baseline)
elif direction == "lower_is_better":
    improvement = (baseline - candidate) / abs(baseline)
```

### 2.3 裁决逻辑

失败 → 改进幅度不足 → p 值不显著 → 改进正负 → 最终裁决

---

## 3. 从零实现

```python
"""结果评估器——配对 t 检验+裁决。"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricSpec:
    name: str; direction: str = "lower_is_better"
    scale: str = "linear"

@dataclass
class Verdict:
    hypothesis_id: int; metric: str; direction: str
    improvement: float; p_value: Optional[float]
    verdict: str; rationale: str


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1: return 0.0
    ft = 1.0; ab = a + b
    d = 1.0 - ab * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, 301):
        fm = float(m)
        num = fm * (b - fm) * x / ((a + 2*fm - 1) * (a + 2*fm))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / (h if m == 1 else (c := 1.0 + num / c) if m > 1 else 1.0)
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; delta = c * d; h *= delta
        if abs(delta - 1.0) < 1e-10: break
    return h * math.exp(a * math.log(x) + b * math.log(1-x)) / (a * math.beta(a, b))

def two_sided_p(t_stat: float, df: int) -> float:
    if df <= 0: return 1.0
    x = df / (df + t_stat * t_stat)
    return min(regularized_beta(x, df/2, 0.5), 1 - regularized_beta(x, df/2, 0.5)) * 2

class Evaluator:
    def evaluate(self, hid: int, ms: MetricSpec, cand: List[Dict], base: List[Dict]) -> Verdict:
        if any(r.get("terminal","ok")!="ok" for r in cand):
            return Verdict(hid, ms.name, ms.direction, 0.0, None, "failed", "实验失败")
        cv = [r["metrics"][ms.name] for r in cand if ms.name in r.get("metrics",{})]
        bv = [r["metrics"][ms.name] for r in base if ms.name in r.get("metrics",{})]
        if not cv or not bv: return Verdict(hid, ms.name, ms.direction, 0.0, None, "noise", "缺指标")
        cm, bm = sum(cv)/len(cv), sum(bv)/len(bv)
        def tr(v): return math.log(v) if ms.scale=="log" else v
        imp = (tr(cm)-tr(bm))/abs(tr(bm)) if ms.direction=="higher_is_better" else (tr(bm)-tr(cm))/abs(tr(bm))
        if len(cv) < 2: return Verdict(hid, ms.name, ms.direction, imp, None, "noise", f"样本不足 imp={imp:.2%}")
        diffs = [tr(c)-tr(b) for c,b in zip(cv,bv)]
        md = sum(diffs)/len(diffs)
        p = two_sided_p(md/math.sqrt(sum((d-md)**2 for d in diffs)/(len(diffs)-1)/len(diffs)), len(diffs)-1) if sum((d-md)**2 for d in diffs)>0 else 1.0
        if abs(imp) < 0.02: v,r = "noise", "改进不足 2%"
        elif p > 0.05: v,r = "noise", f"p={p:.4f} 不显著"
        elif imp > 0: v,r = "improved", f"改进 {imp:.2%}"
        else: v,r = "regressed", f"回归 {imp:.2%}"
        return Verdict(hid, ms.name, ms.direction, imp, p, v, r)


def main():
    c = [{"metrics":{"loss":0.52}},{"metrics":{"loss":0.51}}]
    b = [{"metrics":{"loss":0.55}},{"metrics":{"loss":0.54}}]
    v = Evaluator().evaluate(1, MetricSpec("loss"), c, b)
    print(f"裁决: {v.verdict}  改进: {v.improvement:.2%}  p={v.p_value:.4f}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | t 检验 | 贝叶斯检验 | 仪表板 |
|:----|:------|:----------|:------|
| SciPy | ✓ | ✓ | 无 |
| MLflow | ✓ | ✗ | ✓ |
| Weights & Biases | ✓ | ✓ | ✓ |

---

## 5. 工程最佳实践

- 始终报告 p 值和改进幅度
- 种子必须配对，否则 t 检验无效
- **中文场景建议**：裁决理由使用中文

---

## 6. 常见错误

- **种子未配对**：候选和基线种子必须一一对应
- **对数尺度未处理**：困惑度不取对数时改进百分比被放大
- **样本量不足**：少于 3 个种子时 p 值不可靠

---

## 7. 面试考点

**Q1：为什么配对 t 检验比独立 t 检验更适合？**（难度：⭐⭐）

**参考答案：** 配对 t 检验消除了种子之间随机初始化的方差，只比较种子内的差异，统计功效更高。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 配对 t 检验 | 消除随机初始化方差的成对比较 |
| 方向感知 | 指标上升或下降方向确定的改进计算 |
| 改进阈值 | 默认 2%，低于此视为噪声 |

---

## 📚 小结

结果评估器将运行器数字转化为"改进/回归/噪声"的裁决。下一节将构建基于这些裁决撰写论文的写作器。

---

## ✏️ 练习

1. 【理解】配对 t 检验的零假设是什么？p 值意味着什么？
2. 【实现】为 `Verdict` 添加 `rationale` 中文字段

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 评估器 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Student. "The Probable Error of a Mean". 1908.
