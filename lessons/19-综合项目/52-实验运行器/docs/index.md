# 综合项目52——实验运行器（Experiment Runner）

> 研究循环的诚实程度与其测量结果一致。构建一个接受规范、在沙箱子进程中执行并发出可信 JSON 指标块的运行器。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 将实验编码为类型化规范，运行器可将其序列化到子进程
- 启动具有硬墙上时钟超时和软内存上限的子进程
- 捕获 stdout、stderr 和结构化指标块到单个结果记录
- 构建在固定基线上一次扫描一个配置旋钮的消融表
- 保持结果在给定种子下的确定性

---

## 1. 问题

研究循环运行不可信代码。假设来自采样器，实验脚本也来自相同的路径。将两者视为安全的进程中运行是让崩溃带倒编排器的风险。子进程是语言提供的最简单的隔离方式：独立的进程、独立的地址空间、父进程端的信号处理。

---

## 2. 核心概念

### 2.1 实验规范

```text
ExperimentSpec
  spec_id        : str           (稳定 ID)
  hypothesis_id  : int           (关联假设)
  script_path    : str           (运行的 Python 脚本路径)
  config         : dict          (作为 JSON 参数传入)
  seed           : int           (确定性种子)
  wall_timeout_s : float         (硬超时)
  memory_cap_mb  : int           (软上限)
  metric_keys    : list[str]     (评估器读取的字段)
```

### 2.2 软内存上限

硬内存上限需要 `resource.setrlimit` 且仅 POSIX 支持。课程提供可移植方法：从平台轮询常驻集大小，超过上限则终止子进程。上限是软的，因为轮询器有非零间隔。

### 2.3 消融表

给定基础规范和旋钮名称，每个配置值生成一个规范。一次一个旋钮——全因子扫描指数级爆炸且产生评估器无法解读的结果。

---

## 3. 从零实现

```python
"""实验运行器——子进程隔离+超时+内存上限+消融。"""
from __future__ import annotations
import json, os, subprocess, sys, time, threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentSpec:
    spec_id: str; hypothesis_id: int; script_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0; wall_timeout_s: float = 60.0
    memory_cap_mb: int = 1024; metric_keys: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    spec_id: str; hypothesis_id: int; exit_code: int
    terminal: str  # "ok" | "timeout" | "oom" | "crash"
    wall_time_s: float; peak_rss_mb: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    stdout_tail: str = ""; stderr_tail: str = ""


class ExperimentRunner:
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        config = {**spec.config, "__seed": spec.seed}
        config_path = f"/tmp/{spec.spec_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, spec.script_path, config_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        timed_out = [False]; peak_rss = [0.0]

        def poll_memory():
            while proc.poll() is None:
                try:
                    rss = int(open(f"/proc/{proc.pid}/status").read().split("VmRSS:")[1].split()[0])
                    peak_rss[0] = max(peak_rss[0], rss)
                    if rss > spec.memory_cap_mb * 1024:
                        proc.kill(); return
                except: pass
                time.sleep(0.5)

        mem_thread = threading.Thread(target=poll_memory, daemon=True)
        mem_thread.start()
        try:
            stdout, stderr = proc.communicate(timeout=spec.wall_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill(); stdout, stderr = proc.communicate()
            timed_out[0] = True
        wall = time.perf_counter() - start

        metrics = {}
        for line in reversed(stdout.strip().split("\n")):
            try:
                parsed = json.loads(line)
                if all(k in parsed for k in spec.metric_keys):
                    metrics = parsed; break
            except: continue
        exit_code = proc.returncode or 0
        if timed_out[0]:
            terminal = "timeout"
        elif peak_rss[0] > spec.memory_cap_mb * 1024:
            terminal = "oom"
        elif exit_code != 0:
            terminal = "crash"
        else:
            terminal = "ok"
        return ExperimentResult(spec_id=spec.spec_id, hypothesis_id=spec.hypothesis_id,
                                exit_code=exit_code, terminal=terminal, wall_time_s=wall,
                                peak_rss_mb=peak_rss[0] / 1024 if peak_rss[0] else None,
                                metrics=metrics, stdout_tail=stdout[-500:], stderr_tail=stderr[-500:])


def ablate(base: ExperimentSpec, knob: str, values: List[Any]) -> List[ExperimentSpec]:
    return [ExperimentSpec(spec_id=f"{base.spec_id}_{knob}_{v}", hypothesis_id=base.hypothesis_id,
                           script_path=base.script_path, config={**base.config, knob: v},
                           seed=base.seed, wall_timeout_s=base.wall_timeout_s,
                           memory_cap_mb=base.memory_cap_mb, metric_keys=base.metric_keys)
            for v in values]


def main():
    import tempfile
    script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    script.write("""
import json, sys, time, numpy as np
cfg = json.load(open(sys.argv[1]))
np.random.seed(cfg.get("__seed", 0))
loss = float(np.random.randn() * 0.1)
time.sleep(cfg.get("sleep_s", 0.01))
print(json.dumps({"loss": loss, "perplexity": float(np.exp(loss))}))
""")
    script.close()
    spec = ExperimentSpec(spec_id="test_01", hypothesis_id=1, script_path=script.name,
                          config={"sleep_s": 0.01}, seed=42, metric_keys=["loss", "perplexity"])
    runner = ExperimentRunner()
    result = runner.run(spec)
    print(f"终端状态: {result.terminal}")
    print(f"指标: {result.metrics}")
    print(f"墙上时间: {result.wall_time_s:.3f}s")
    specs = ablate(spec, "sleep_s", [0.01, 0.02, 0.05])
    print(f"\n消融表: {len(specs)} 个规范")
    for s in specs:
        print(f"  {s.spec_id}")
    os.unlink(script.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 子进程隔离 | 使用独立进程运行实验，崩溃不波及编排器 |
| 墙上时钟超时 | 硬限制——实验必须在设定时间内完成 |
| 软内存上限 | 轮询 RSS 并在超限时终止进程 |
| 消融表 | 一次改变一个配置旋钮的实验系列 |
| 种子确定性 | 同一种子产生完全相同的指标值 |

---

## 5. 工程最佳实践

- **始终设置超时**：即使是正确的实验也可能因数据问题无限挂起。默认值 60 秒是安全的起点。
- **记录峰值 RSS**：内存泄漏是训练实验中常见且难以诊断的问题。记录每个实验的峰值 RSS。
- **中文场景特别建议**：临时文件路径避免使用中文字符，防止在某些文件系统上出错。

---

## 6. 常见错误

- **在子进程中导入 GPU 库失败**：确保实验脚本检查 CUDA 可用性。运行器本身不应依赖 GPU。
- **超时后未清理子进程**：`TimeoutExpired` 后必须调用 `proc.kill()`，否则子进程成为孤儿进程。
- **配置中未传递种子**：种子通过 `config["__seed"]` 传递，但实验脚本必须显式读取并设置 numpy 和 torch 种子。

---

## 📖 参考资料

1. [官方文档] Python `subprocess` 模块. https://docs.python.org/3/library/subprocess.html
2. [官方文档] `resource.setrlimit` 内存限制. https://docs.python.org/3/library/resource.html
