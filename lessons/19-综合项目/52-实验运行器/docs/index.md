# 综合项目52——实验运行器（Experiment Runner）

> 研究循环的诚实程度与其测量结果一致。构建一个接受规范、在沙箱子进程中执行并发出可信 JSON 指标块的运行器。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 将实验编码为类型化规范
- 启动具有硬超时和软内存上限的子进程
- 捕获 stdout、stderr 和指标到统一结果
- 构建一次扫描一个配置旋钮的消融表

---

## 1. 问题

研究循环运行不可信代码。假设来自采样器，实验脚本也来自相同路径。将两者视为安全的进程内运行，是让崩溃带倒编排器的风险。子进程是语言提供的最简单隔离方式。

---

## 2. 核心概念

### 2.1 实验规范

```text
ExperimentSpec(spec_id, hypothesis_id, script_path, config,
               seed, wall_timeout_s, memory_cap_mb, metric_keys)
```

### 2.2 子进程生命周期

1. 序列化配置到临时 JSON 文件
2. `subprocess.Popen([python, script, config_path], pipe stdout/stderr)`
3. 启动墙上时钟定时器和内存轮询线程
4. 超时或超内存时 `proc.kill()`
5. 解析最终 JSON 行作为指标

### 2.3 内存上限

轮询 `/proc/{pid}/status` 中的 `VmRSS`，超过上限则终止。回退到无操作当平台不支持时。记录峰值 RSS。

---

## 3. 从零实现

```python
"""实验运行器——子进程隔离+超时+消融。"""
from __future__ import annotations
import json, os, subprocess, sys, time, threading
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    wall_time_s: float; peak_rss_mb: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        config_path = f"/tmp/{spec.spec_id}_cfg.json"
        with open(config_path, "w") as f:
            json.dump({**spec.config, "__seed": spec.seed}, f)
        start = time.perf_counter()
        proc = subprocess.Popen([sys.executable, spec.script_path, config_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timed_out = [False]; peak_rss = [0.0]

        def poll():
            while proc.poll() is None:
                try:
                    rss = int(open(f"/proc/{proc.pid}/status").read().split("VmRSS:")[1].split()[0])
                    peak_rss[0] = max(peak_rss[0], rss)
                    if rss > spec.memory_cap_mb * 1024: proc.kill(); return
                except: pass
                time.sleep(0.5)

        t = threading.Thread(target=poll, daemon=True); t.start()
        try: stdout, _ = proc.communicate(timeout=spec.wall_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill(); stdout, _ = proc.communicate(); timed_out[0] = True
        wall = time.perf_counter() - start
        metrics = {}
        for line in reversed(stdout.strip().split("\n")):
            try:
                p = json.loads(line)
                if all(k in p for k in spec.metric_keys): metrics = p; break
            except: pass
        if timed_out[0]: terminal = "timeout"
        elif peak_rss[0] > spec.memory_cap_mb * 1024: terminal = "oom"
        elif proc.returncode: terminal = "crash"
        else: terminal = "ok"
        return ExperimentResult(spec.spec_id, spec.hypothesis_id, proc.returncode or 0,
                                terminal, wall, peak_rss[0]/1024, metrics)


def ablate(base: ExperimentSpec, knob: str, vals: List) -> List[ExperimentSpec]:
    return [ExperimentSpec(f"{base.spec_id}_{knob}_{v}", base.hypothesis_id, base.script_path,
                           {**base.config, knob: v}, base.seed, base.wall_timeout_s,
                           base.memory_cap_mb, base.metric_keys) for v in vals]


def main():
    import tempfile
    s = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    s.write("import json,sys,time,numpy as np; cfg=json.load(open(sys.argv[1]))\nnp.random.seed(cfg.get('__seed',0))\nloss=float(np.random.randn()*0.1)\ntime.sleep(0.01)\nprint(json.dumps({'loss':loss,'perplexity':float(np.exp(loss))}))")
    s.close()
    spec = ExperimentSpec("test", 1, s.name, {"sleep_s":0.01}, 42, 10, metric_keys=["loss","perplexity"])
    r = ExperimentRunner().run(spec)
    print(f"终端: {r.terminal}  指标: {r.metrics}  时间: {r.wall_time_s:.3f}s")
    os.unlink(s.name); return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. 工业工具

| 工具 | 超时 | 内存限制 | GPU |
|:----|:----|:--------|:----|
| subprocess（本课） | 软件轮询 | /proc 轮询 | 手动 |
| Firecracker | 硬件 | cgroup | 是 |
| Docker | 内建 | cgroup | 是 |
| Ray Tasks | 内建 | 内建 | 是 |

---

## 5. 工程最佳实践

- 始终设置超时；默认 60 秒是安全起点
- 记录峰值 RSS 用于检测内存泄漏
- **中文场景建议**：临时文件路径避免使用中文

---

## 6. 常见错误

- **超时后未 kill**：`TimeoutExpired` 后必须 `proc.kill()` 防止孤儿进程
- **配置中未传递种子**：实验脚本必须显式读取 `__seed`
- **标准输出中多行 JSON 混淆**：只取最后一行含所有 metric_keys 的 JSON

---

## 7. 面试考点

**Q1：为什么使用子进程而不是线程？**（难度：⭐⭐）

**参考答案：** 子进程提供独立地址空间，实验崩溃不会影响编排器。线程共享地址空间——一个段错误就带倒整个进程。

**Q2：软内存上限为什么是"软"的？**（难度：⭐⭐）

**参考答案：** 轮询间隔内进程可能短暂峰值超限然后回落。`resource.setrlimit` 提供硬限制但不跨平台。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 子进程隔离 | 独立进程运行实验 |
| 墙上时钟超时 | 实验必须在此时间内完成 |
| 软内存上限 | 轮询 RSS 超限时终止 |
| 消融表 | 一次改变一个配置的实验系列 |

---

## 📚 小结

实验运行器是研究循环的执行引擎。你实现了子进程隔离、超时控制、内存监控和消融表，确保实验安全可靠运行。下一节将构建评估运行结果的评估器。

---

## ✏️ 练习

1. 【实现】添加 `--epochs` 配置项使实验脚本可以运行指定轮次
2. 【实验】用 `timeout` 和 `oom` 两种终端状态验证运行器行为

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 实验运行器 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] `subprocess` 模块. https://docs.python.org/3/library/subprocess.html
