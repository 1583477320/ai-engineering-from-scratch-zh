# 综合项目27——评估框架与固定任务（Eval Harness with Fixture Tasks）

> 编码智能体的好坏取决于你测量它的任务套件。本节构建一个评估框架：接收固定任务目录，逐个运行候选智能体，通过确定性验证器评分通过/失败，聚合为 pass@1、pass@k、平均延迟和成本。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 第19章第25-26节
**预计时间：** 90分钟

---

## 学习目标

- 将固定任务定义为目标、设置和验证器三元组
- 每任务多次采样计算 pass@1 和 pass@k
- 聚合延迟和成本为均值与 P95
- 发射结构化 JSON 报告供回归追踪脚本消费

---

## 1. 问题

没有评估框架的智能体基准测试有三种失败模式。未验证的通过——智能体声称修复了但实际没有。未检测的回归——提示模板修改使某任务提升 4% 而另一任务下降 14%。任务漂移——周一 100 个任务周五 95 个，通过率虚假提升 5%。

评估框架将这些失败转化为事实：每次运行所有固定任务，以确定性验证器返回真/假。

---

## 2. 核心概念

### 2.1 固定任务结构

```text
FixtureTask
  id: str          # 任务唯一标识
  goal: str        # 提供给智能体的提示词
  setup: dict      # 放入暂存目录的文件
  verifier: dict   # 验证器名称 + 参数
```

### 2.2 三种验证器

| 验证器 | 作用 | 适用场景 |
|:------|:-----|:---------|
| `file_equals` | 比较文件内容 | "按指定方式修复这个 bug" |
| `regex_match` | 正则匹配 | "函数必须存在并返回 X" |
| `shell_exit_zero` | 命令退出码为 0 | "测试必须通过" |

### 2.3 pass@k 计算

```
pass@k = 1 - (1 - p)^k
```

p 是经验通过率，k 是采样次数。同时报告原始计数以检测方差。

### 2.4 聚合报告

```text
EvalReport
  total_tasks: int
  pass_at_1: float
  pass_at_k: float
  mean_latency_ms: float
  p95_latency_ms: float
  mean_cost: float
  per_task: [TaskReport]
```

---

## 3. 从零实现

```python
"""评估框架——固定任务+确定性验证+pass@k。"""
import json, os, re, shutil, tempfile, time, math
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any


@dataclass
class TaskSpec:
    id: str; goal: str; setup: Dict[str, str] = field(default_factory=dict)
    verifier: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    success: bool; latency_ms: float; cost: float = 0.0; edits: List[str] = field(default_factory=list)


@dataclass
class TaskReport:
    task_id: str; k: int; passes: int; pass_rate: float
    mean_latency_ms: float; p95_latency_ms: float; mean_cost: float


@dataclass
class EvalReport:
    total_tasks: int; pass_at_1: float; pass_at_k: float
    mean_latency_ms: float; p95_latency_ms: float; mean_cost: float
    per_task: List[TaskReport] = field(default_factory=list)

    def to_dict(self):
        return {
            "total_tasks": self.total_tasks, "pass_at_1": round(self.pass_at_1, 4),
            "pass_at_k": round(self.pass_at_k, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "mean_cost": round(self.mean_cost, 4),
        }


def verify_file_equals(scratch_dir, filename, expected_content):
    path = os.path.join(scratch_dir, filename)
    if not os.path.exists(path): return False
    return open(path).read().strip() == expected_content.strip()


def verify_regex_match(scratch_dir, filename, pattern):
    path = os.path.join(scratch_dir, filename)
    if not os.path.exists(path): return False
    return bool(re.search(pattern, open(path).read()))


def verify_shell_exit_zero(scratch_dir, command):
    import subprocess
    r = subprocess.run(command, shell=True, cwd=scratch_dir, capture_output=True, timeout=30)
    return r.returncode == 0


VERIFIERS = {"file_equals": verify_file_equals, "regex_match": verify_regex_match, "shell_exit_zero": verify_shell_exit_zero}


class EvalHarness:
    def __init__(self, k: int = 5):
        self.k = k

    def run(self, tasks: List[TaskSpec], candidate_fn: Callable) -> EvalReport:
        task_reports = []
        all_latencies, all_costs = [], []
        total_passes = 0

        for task in tasks:
            passes = 0; latencies = []; costs = []
            for _ in range(self.k):
                scratch = tempfile.mkdtemp()
                try:
                    for fname, content in task.setup.items():
                        os.makedirs(os.path.join(scratch, os.path.dirname(fname)), exist_ok=True)
                        with open(os.path.join(scratch, fname), "w") as f: f.write(content)
                    t0 = time.perf_counter()
                    result = candidate_fn(task, scratch)
                    latency = (time.perf_counter() - t0) * 1000
                    vname = task.verifier.get("type", "file_equals")
                    if vname in VERIFIERS:
                        passed = VERIFIERS[vname](scratch, **{k: v for k, v in task.verifier.items() if k != "type"})
                    else:
                        passed = result.success
                    if passed: passes += 1
                    latencies.append(latency); costs.append(result.cost)
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)

            pr = passes / self.k
            latencies.sort()
            p95_idx = max(0, int(len(latencies) * 0.95) - 1)
            tr = TaskReport(task.id, self.k, passes, pr,
                           sum(latencies) / len(latencies) if latencies else 0,
                           latencies[p95_idx] if latencies else 0,
                           sum(costs) / len(costs) if costs else 0)
            task_reports.append(tr); all_latencies.extend(latencies); all_costs.extend(costs)
            total_passes += passes

        n_tasks = len(tasks)
        total_samples = n_tasks * self.k
        pass_at_1 = total_passes / total_samples if total_samples else 0
        pass_at_k = 1 - (1 - pass_at_1) ** self.k if pass_at_1 < 1 else 1.0
        all_latencies.sort()
        p95_idx = max(0, int(len(all_latencies) * 0.95) - 1)
        return EvalReport(n_tasks, pass_at_1, pass_at_k,
                         sum(all_latencies) / len(all_latencies) if all_latencies else 0,
                         all_latencies[p95_idx] if all_latencies else 0,
                         sum(all_costs) / len(all_costs) if all_costs else 0,
                         task_reports)


def demo_candidate(task, scratch_dir):
    """参考候选——对固定任务总是通过。"""
    return SampleResult(success=True, latency_ms=1.0, cost=0.0)


def build_fixtures():
    return [
        TaskSpec("fizz_off_by_one", "Fix the off-by-one in fizzbuzz function",
                 {"src/fizz.py": "def fizzbuzz(n):\n    for i in range(1, n):\n        if i % 15 == 0: print('FizzBuzz')\n        elif i % 3 == 0: print('Fizz')\n        elif i % 5 == 0: print('Buzz')\n        else: print(i)"},
                 {"type": "file_equals", "filename": "src/fizz.py",
                  "expected_content": "def fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0: print('FizzBuzz')\n        elif i % 3 == 0: print('Fizz')\n        elif i % 5 == 0: print('Buzz')\n        else: print(i)"}),
        TaskSpec("factorial_missing_return", "Fix missing return in factorial",
                 {"src/fact.py": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    # missing return"},
                 {"type": "regex_match", "filename": "src/fact.py", "pattern": r"return\s+result"}),
    ]


def main():
    harness = EvalHarness(k=3)
    report = harness.run(build_fixtures(), demo_candidate)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 验证 | pass@k | 特点 |
|:----|:-----|:------|:-----|
| HumanEval | 测试用例 | ✓ | 代码生成标准 |
| SWE-bench | 文件 diff | ✓ | 软件工程基准 |
| BigCodeBench | 单元测试 | ✓ | 大规模代码 |
| 本课 | 三种验证器 | ✓ | 教学可扩展 |

---

## 5. 工程最佳实践

- pass@1 和 pass@k 同时报告——pass@k 可能掩盖低 pass@1
- 每次运行固定任务集和顺序——可复现是评估的生命线
- **中文场景建议**：固定任务文件使用 UTF-8 编码，验证器正确处理中文注释

---

## 6. 常见错误

- **任务漂移**：运行中修改任务集导致通过率不可比——锁定任务哈希
- **未清理暂存目录**：每次任务后必须清理临时文件
- **验证器类型错误**：`shell_exit_zero` 不应该在 `file_equals` 的参数上调用

---

## 7. 面试考点

**Q1：pass@1 和 pass@k 分别告诉你什么？**（难度：⭐⭐）

**参考答案：** pass@1 衡量单次运行的成功率——最接近实际部署场景（用户不会让模型重试 20 次）。pass@k 衡量"多次尝试中至少一个成功"的概率——揭示模型是否在正确答案附近但选择了错误的。两者结合才能判断模型能力。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 固定任务 | 目标 + 设置 + 验证器三元组 |
| pass@k | k 次采样中至少一次通过的概率 |
| 确定性验证 | 基于文件/正则/命令退出码的自动判断 |
| 回归追踪 | 固定任务集在不同运行间保持一致 |

---

## 📚 小结

评估框架是智能体开发的真相来源。你实现了三种验证器、pass@k 计算和结构化报告。下一节构建 OTel 可观测性。

---

## ✏️ 练习

1. 【实现】添加第四种验证器 `output_contains`：检查 stdout 中包含指定字符串
2. 【实验】运行 pass@k 曲线（k=1 到 k=10），找到通过率趋于饱和的 k 值

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 评估框架 | `code/main.py` |
| 固定任务 | `tasks/` |

---

## 📖 参考资料

1. [论文] Chen et al. "HumanEval". 2021.
2. [GitHub] SWE-bench. https://github.com/princeton-nlp/SWE-bench
