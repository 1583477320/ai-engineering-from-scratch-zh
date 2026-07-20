# 综合项目25——评测工具与固定任务（pass@k）

> 编码智能体只和你衡量它的一套任务一样好。本课程构建一个评测工具，它接受一个固定任务文件夹，通过候选智能体运行每个任务，通过确定性验证器评分通过或失败，并将结果聚合为pass@1、pass@k、平均延迟和平均成本。工具是区分回归和重构的唯一真实来源。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** 第19章 · 第23节（验证门）、第24节（沙箱运行器）
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 将固定任务定义为目标、设置和验证器的三元组
- 对每个任务运行多个样本，计算pass@1和pass@k
- 将延迟和成本聚合为均值和第95百分位指标
- 将确定性验证器（文件差异、退出码、正则匹配）接入可复用函数
- 输出结构化JSON报告供回归跟踪脚本摄取

---

## 1. 问题

没有评测工具构建的智能体基准有三类失败模式：

1. **未验证的通过**：智能体说修复了bug，人类瞥了一眼diff，套件标记为绿色。三周后回归测试浮现相同bug。
2. **未检测的回归**：提示模板更改使智能体在响亮任务上好4%，在安静任务上差14%。没有黄金集和每任务分数，回归进入main。
3. **每任务漂移**：评测周一运行100个任务，周五运行95个（有人重命名了5个固定文件）。通过率看起来提升5%——其实没有。

---

## 2. 核心概念

### 2.1 FixtureTask

一个小JSON文件加可选的`expected/`目录。JSON声明id、goal（给智能体的提示）、setup块（放入scratch目录的文件）和verifier块。

### 2.2 三种验证器

1. **file_equals**：比较命名文件与期望内容
2. **regex_match**：命名文件内容匹配正则
3. **shell_exit_zero**：通过沙箱运行shell命令，退出码为零则通过

### 2.3 pass@k

pass@k = 1 - (1-p)^k，其中p是经验通过率。工具报告原始计数以发现方差。

### 2.4 EvalReport

```text
EvalReport
  task_reports: list[TaskReport]
  pass_at_1: float
  pass_at_k: float
  k: int
  mean_latency_ms: float
  p95_latency_ms: float
  total_cost: float
```

---

## 3. 从零实现

`code/main.py`实现`EvalHarness`、`FixtureTask`、`SampleResult`、三种验证器和确定性参考候选。

```python
"""评测工具——固定任务、评分样本、pass@k。

核心：确定性验证器（file_equals/regex_match/shell_exit_zero）+pass@k数学。
参考候选复制期望文件到scratch目录，演示pass@1=1.0。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, os, re, shutil, statistics, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class FixtureTask:
    id:str; goal:str; setup_dir:str; expected_dir:str; verifier_name:str; verifier_args:dict[str,Any]; root:str=""

@dataclass
class SampleResult:
    task_id:str; sample_index:int; latency_ms:float; cost_units:float=0.0; notes:str=""

@dataclass
class VerificationOutcome:
    passed:bool; detail:str

@dataclass
class TaskReport:
    task_id:str; k:int; passes:int; pass_rate:float; pass_at_k:float
    mean_latency_ms:float; p95_latency_ms:float; mean_cost:float; samples:list[dict]=field(default_factory=list)

@dataclass
class EvalReport:
    task_reports:list[TaskReport]; pass_at_1:float; pass_at_k:float; k:int
    mean_latency_ms:float; p95_latency_ms:float; total_cost:float

def pass_at_k(p,k):
    if k<=0: return 0.0
    p=max(0.0,min(1.0,p))
    return 1.0-(1.0-p)**k

def p95(values):
    if not values: return 0.0
    s=sorted(values); idx=max(0,int(round(0.95*len(s)))-1)
    return s[min(idx,len(s)-1)]

Verifier=Callable[[FixtureTask,str,dict],VerificationOutcome]

def verify_file_equals(task,scratch,args):
    rel=args.get("path")
    if not isinstance(rel,str): return VerificationOutcome(False,"missing 'path'")
    actual=os.path.join(scratch,rel); expected=os.path.join(task.expected_dir,rel)
    if not os.path.isfile(actual): return VerificationOutcome(False,f"missing: {rel}")
    if not os.path.isfile(expected): return VerificationOutcome(False,f"expected missing: {rel}")
    a=open(actual).read(); e=open(expected).read()
    if bool(args.get("normalize_trailing_newline",True)):
        a=a.rstrip("\n")+"\n"; e=e.rstrip("\n")+"\n"
    return VerificationOutcome(a==e,f"file {rel!r} {'matches' if a==e else 'differs'}")

def verify_regex_match(task,scratch,args):
    rel=args.get("path"); pat=args.get("pattern")
    if not isinstance(rel,str) or not isinstance(pat,str): return VerificationOutcome(False,"need 'path' and 'pattern'")
    actual=os.path.join(scratch,rel)
    if not os.path.isfile(actual): return VerificationOutcome(False,f"missing: {rel}")
    text=open(actual).read()
    return VerificationOutcome(bool(re.search(pat,text,re.MULTILINE)),f"matched {pat!r}")

def verify_shell_exit_zero(task,scratch,args):
    argv=args.get("argv")
    if not isinstance(argv,list) or not argv: return VerificationOutcome(False,"need 'argv' list")
    try: proc=subprocess.run(list(argv),cwd=scratch,capture_output=True,timeout=float(args.get("timeout_seconds",10)))
    except subprocess.TimeoutExpired: return VerificationOutcome(False,"timed out")
    except FileNotFoundError as exc: return VerificationOutcome(False,f"not found: {exc}")
    return VerificationOutcome(proc.returncode==0,f"exit {proc.returncode}")

VERIFIERS={"file_equals":verify_file_equals,"regex_match":verify_regex_match,"shell_exit_zero":verify_shell_exit_zero}

def load_fixture(task_dir):
    with open(os.path.join(task_dir,"task.json")) as f: spec=json.load(f)
    return FixtureTask(id=spec["id"],goal=spec["goal"],setup_dir=os.path.join(task_dir,"buggy"),
        expected_dir=os.path.join(task_dir,"expected"),verifier_name=spec["verifier"]["name"],
        verifier_args=spec["verifier"].get("args",{}),root=task_dir)

def load_all(root):
    return [load_fixture(os.path.join(root,n)) for n in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root,n)) and os.path.isfile(os.path.join(root,n,"task.json"))]

Candidate=Callable[[FixtureTask,str],SampleResult]

def apply_known_fixes(task,scratch):
    start=time.perf_counter()
    if os.path.isdir(task.expected_dir):
        for dp,_,files in os.walk(task.expected_dir):
            rel=os.path.relpath(dp,task.expected_dir); dst=scratch if rel=="." else os.path.join(scratch,rel)
            os.makedirs(dst,exist_ok=True)
            for fn in files: shutil.copy2(os.path.join(dp,fn),os.path.join(dst,fn))
    return SampleResult(task.id,0,(time.perf_counter()-start)*1000,1.0,"reference")

def noop_candidate(task,scratch): return SampleResult(task.id,0,0,0,"noop")

@dataclass
class EvalHarness:
    tasks:list[FixtureTask]; k:int=1; verifier_registry:dict[str,Verifier]=field(default_factory=lambda:dict(VERIFIERS))

    def run(self,candidate):
        task_reports=[]
        for task in self.tasks:
            samples=[]; lats=[]; costs=[]; passes=0
            for si in range(self.k):
                scratch=tempfile.mkdtemp(prefix=f"eval-{task.id}-")
                try:
                    sample=candidate(task,scratch)
                    outcome=self.verifier_registry[task.verifier_name](task,scratch,task.verifier_args)
                    lats.append(sample.latency_ms); costs.append(sample.cost_units)
                    if outcome.passed: passes+=1
                    samples.append({"index":si,"latency_ms":round(sample.latency_ms,3),"passed":outcome.passed,"detail":outcome.detail})
                finally: shutil.rmtree(scratch,ignore_errors=True)
            pr=passes/self.k if self.k else 0.0
            task_reports.append(TaskReport(task.id,self.k,passes,pr,pass_at_k(pr,self.k),
                statistics.mean(lats) if lats else 0,p95(lats),statistics.mean(costs) if costs else 0,samples))
        p1=[min(1.0,r.pass_rate) for r in task_reports]; pk=[r.pass_at_k for r in task_reports]
        all_lat=[s["latency_ms"] for r in task_reports for s in r.samples]
        tc=sum(s.get("cost_units",0) for r in task_reports for s in r.samples)
        return EvalReport(task_reports,statistics.mean(p1) if p1 else 0,statistics.mean(pk) if pk else 0,
            self.k,statistics.mean(all_lat) if all_lat else 0,p95(all_lat),tc)

def _demo():
    tasks_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tasks")
    tasks=load_all(tasks_dir)
    if not tasks: print("no fixtures"); return 1
    print(f"loaded {len(tasks)} fixture tasks")
    harness=EvalHarness(tasks=tasks,k=1); report=harness.run(apply_known_fixes)
    print(json.dumps({"pass_at_1":round(report.pass_at_1,4),"mean_latency":round(report.mean_latency_ms,3),
        "tasks":[{"id":t.task_id,"passes":t.passes,"pass_rate":round(t.pass_rate,4)} for t in report.task_reports]},indent=2))
    return 0 if report.pass_at_1>=1.0 else 1

if __name__=="__main__": sys.exit(_demo())
```

---

## 4. 工具实践

**评测工具位置**：工具是验证门（第23节）和沙箱（第24节）之上的质量层。它比较智能体结果与期望。

**pass@k vs pass@1**：真实LLM智能体是随机的。pass@1=0.6看起来像失败。pass@5=0.95说智能体大多时候得到正确答案但早期样本选错。修复是采样和排名，而非更多训练。

---

## 5. LLM视角

**确定性验证视角**：验证器是确定性的——给定相同文件产生相同结果。这使评测可重现，不依赖模型随机性。

**回归检测视角**：工具是区分回归和重构的唯一真实来源。没有它，"智能体变好了"可能只是测试覆盖不同。

---

## 6. 工程最佳实践

**固定任务设计**：每个任务含task.json（id/goal/verifier）+ buggy/（有bug的源文件）+ expected/（期望的修复文件）。

**scratch目录**：每个样本在tempfile.mkdtemp()中运行，完成后清理。

**参考候选**：用于工具自检——复制expected文件到scratch目录。

---

## 7. 常见错误

**错误1：不清理scratch目录**
症状：磁盘空间泄漏
修复：finally块中shutil.rmtree

**错误2：不追踪每任务分数**
症状：整体通过率掩盖任务级回归
修复：每个TaskReport单独跟踪

---

## 8. 面试考点

**Q1：为什么需要pass@k而不仅是pass@1？**
考察：对随机性和采样的理解

**Q2：确定性验证器为什么比人类评估更可靠？**
考察：对可重现性的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 固定任务 | "评测用例" | goal+setup+verifier三元组 |
| pass@k | "k次采样通过率" | 1-(1-p)^k，k次独立样本中至少一次通过的概率 |
| 确定性验证器 | "自动检查" | file_equals/regex_match/shell_exit_zero |
| EvalReport | "评测报告" | 聚合所有任务的pass@1/pass@k/延迟/成本 |
| 参考候选 | "黄金标准" | 复制expected文件的确定性候选 |
| scratch目录 | "临时工作区" | tempfile.mkdtemp()创建的隔离工作目录 |

---

## 参考文献

- [SWE-bench评测方法论](https://www.swebench.com/)
- [pass@k指标](https://arxiv.org/abs/2107.03374)
