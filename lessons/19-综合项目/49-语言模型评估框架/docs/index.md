# 综合项目49——语言模型评估框架（Language Model Evaluation Harness）

> 一个你在无法定义的任务上表现良好的模型，是碰巧表现良好。评估框架就是任务定义、指标、运行器和排行榜——封装在一个短小、可替换的形状中。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第42-45节
**预计时间：** 90分钟

---

## 学习目标

- 将评估任务定义为 JSONL 文件——每行包含提示词、目标答案、指标和可选扩展字段
- 实现五种指标：精确匹配、ROUGE-L F1、代码可执行检查、多项选择和子串包含
- 构建一个按任务批次运行示例并分发到可替换模型适配器的运行器
- 输出一个带有每任务分数、延迟和总体平均分的排行榜 JSON，结果可复现

---

## 1. 问题

每周都有一个新语言模型问世。营销宣称它表现良好。诚实的问题是：在什么任务上表现良好？诚实的答案是你自己编写的排行榜——因为供应商的排行榜是他们为了自己产品调过的。

如果仓库中没有评估框架，你只能凭感觉比较两个模型。有了评估框架，你可以通过固定任务集和固定指标上的分数来比较它们，输出的是可以 diff 的 JSON。评估框架是昨天运行和今天运行之间的合约。没有它，回归就会上线。

陷阱是过度适应单个模型。解决方法与陷阱相同：评估框架足够小（十五分钟就能读完），任务足够小（可以放进仓库），指标从头编写（同事可以审计），模型适配器是模型特定代码唯一存在的地方。替换适配器，排行榜随之变化。替换任务，排行榜随之变化。其他任何东西都不应该变化。

---

## 2. 核心概念

### 2.1 任务规范

每条评估样本是一条 JSONL 行：

```json
{"id": "arith-00", "prompt": "compute: 2 + 2", "targets": ["4"], "metric": "exact_match"}
```

对于需要评分辅助工具的指标，`extras` 携带附加载荷：

```json
{
  "id": "code-00",
  "prompt": "python: 写一个函数 f，将其输入翻倍",
  "targets": ["ok"],
  "metric": "code_exec",
  "extras": {"io_pairs": [[1, 2], [3, 6]]}
}
```

一个任务是一个 `.jsonl` 文件，放在 `outputs/tasks/` 目录下。文件名就是任务名。一个文件中的所有示例共享一个指标。

### 2.2 五个内置任务

| 任务 | 指标 | 测试内容 |
|------|------|----------|
| 算术（arithmetic） | 精确匹配（exact_match） | 确定性答案上的词元级正确性 |
| 摘要（summary） | ROUGE-L（rouge_l） | 最长公共子序列 F1 |
| 代码执行（code-exec） | 代码执行（code_exec） | 预测的函数必须满足输入-输出对 |
| 多项选择（multiple-choice） | 多项选择（multiple_choice） | 预测的首字母必须匹配允许的字母 |
| 文本生成（generation） | 子串包含（substring_contains） | 自由文本必须包含至少一个目标子串 |

### 2.3 指标合约

每个指标是一个函数：`(prediction, targets, extras) -> float in [0.0, 1.0]`。评估框架对各示例分数取平均得到任务分数，再对任务分数取平均得到总体分。

- **精确匹配**：小写化、折叠空白、相等比较
- **子串包含**：相同归一化，子串测试
- **多项选择**：首字母大写
- **ROUGE-L**：最长公共子序列长度除以预测和参考的长度，精确率和召回率的 F1
- **代码执行**：在受限的命名空间中执行预测代码，对每个输入-输出对调用 `f(x)`，统计匹配数

代码执行指标在剥离了内置函数的命名空间中运行。测试断言 `import os` 会爆炸——因为 `os` 不在命名空间中。

### 2.4 模型适配器

适配器是接缝。课程附带了 `ToyAdapter`，一个确定性的模式匹配器——它对五个内置任务中的每个提示词都能返回正确的答案。真正的适配器调用模型并返回其输出。评估框架不关心哪个。

```python
class ModelAdapter(Protocol):
    def generate(self, prompts: Sequence[str]) -> List[str]: ...
    @property
    def name(self) -> str: ...
```

### 2.5 运行器

`run_task` 将示例按 `batch_size` 分组并将批次分发到适配器。`run_leaderboard` 遍历所有任务并取平均。`write_leaderboard` 输出带 schema 字符串的 JSON，使未来的格式更改不会静默破坏仪表板。

---

## 3. 从零实现

```python
"""语言模型评估框架从零实现。

任务规范：JSONL 每行包含 prompt、targets 和 metric。
五种指标：精确匹配、ROUGE-L、代码执行、多项选择、子串包含。
"""
from __future__ import annotations
import ast, json, operator, re, sys, textwrap, time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

OUT_DIR = Path(__file__).parent.parent / "outputs"
TASKS_DIR = OUT_DIR / "tasks"
LEADERBOARD_PATH = OUT_DIR / "leaderboard.json"

@dataclass
class Example:
    id: str; prompt: str; targets: List[str]; metric: str; extras: Dict[str, object] = field(default_factory=dict)

@dataclass
class TaskResult:
    task: str; metric: str; score: float; correct: int; total: int
    per_example: List[Dict[str, object]] = field(default_factory=list); latency_ms: float = 0.0

@dataclass
class Leaderboard:
    schema: str; timestamp: float; overall_score: float; tasks: List[TaskResult]

class ModelAdapter(Protocol):
    def generate(self, prompts: Sequence[str]) -> List[str]: ...
    @property
    def name(self) -> str: ...

class ToyAdapter:
    """确定性适配器，通过模式匹配每个任务。"""
    name = "toy.v1"
    def generate(self, prompts: Sequence[str]) -> List[str]:
        return [self._answer(p) for p in prompts]

    def _answer(self, prompt: str) -> str:
        text = prompt.strip()
        if text.startswith("compute:"):
            try: return str(safe_arith_eval(text[len("compute:"):].strip()))
            except: return ""
        if text.startswith("summarize:"):
            sents = re.split(r"(?<=[.!?])\s+", text[len("summarize:"):].strip())
            return sents[0] if sents else text
        if text.startswith("python:"):
            body = text[len("python:"):].strip()
            FNS = {"double": "def f(x):\n    return x * 2\n",
                   "increment": "def f(x):\n    return x + 1\n",
                   "square": "def f(x):\n    return x * x\n"}
            for key, fn in FNS.items():
                if key in body: return fn
            return "def f(x):\n    return x\n"
        if text.startswith("choose:"):
            return text[len("choose:"):].strip().split("|", 1)[0].strip()[:1].upper()
        if text.startswith("write:"):
            return text[len("write:"):].strip()
        return text

_ARITH_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Pow: operator.pow}

def safe_arith_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _safe_eval(tree.body)

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不安全节点: {ast.dump(node)}")

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def metric_exact_match(prediction: str, targets: List[str]) -> float:
    return 1.0 if any(normalize(t) == normalize(prediction) for t in targets) else 0.0

def metric_substring_contains(prediction: str, targets: List[str]) -> float:
    np = normalize(prediction)
    return 1.0 if any(normalize(t) in np for t in targets) else 0.0

def metric_multiple_choice(prediction: str, targets: List[str]) -> float:
    pred = prediction.strip()[:1].upper()
    return 1.0 if pred in {t.strip()[:1].upper() for t in targets} else 0.0

def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b: return 0
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b):
            cur[j + 1] = prev[j] + 1 if ai == bj else max(prev[j + 1], cur[j])
        prev = cur
    return prev[-1]

def metric_rouge_l(prediction: str, targets: List[str]) -> float:
    pred = _tokens(prediction)
    if not pred: return 0.0
    best = 0.0
    for ref in targets:
        rt = _tokens(ref)
        if not rt: continue
        lcs = _lcs_length(pred, rt)
        if lcs == 0: continue
        prec, rec = lcs / len(pred), lcs / len(rt)
        if prec + rec == 0: continue
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

def metric_code_exec(prediction: str, targets: List[str], extras: Dict) -> float:
    pairs = extras.get("io_pairs") or []
    if not isinstance(pairs, list) or not pairs: return 0.0
    safe_globals = {"__builtins__": {"range": range, "len": len, "min": min, "max": max, "abs": abs, "int": int, "float": float}}
    local: Dict[str, object] = {}
    try: exec(prediction, safe_globals, local)
    except: return 0.0
    fn = local.get("f")
    if not callable(fn): return 0.0
    correct = sum(1 for pair in pairs if isinstance(pair, list) and len(pair) == 2
                  and (lambda x, e: fn(x) == e)(*pair))
    return correct / len(pairs)

METRIC_FNS = {"exact_match": lambda p, t, e: metric_exact_match(p, t),
              "substring_contains": lambda p, t, e: metric_substring_contains(p, t),
              "multiple_choice": lambda p, t, e: metric_multiple_choice(p, t),
              "rouge_l": lambda p, t, e: metric_rouge_l(p, t),
              "code_exec": lambda p, t, e: metric_code_exec(p, t, e)}

def load_task_jsonl(path: Path) -> List[Example]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"): continue
            obj = json.loads(raw)
            examples.append(Example(id=str(obj.get("id", f"ex-{line_num}")),
                                    prompt=obj["prompt"], targets=list(obj["targets"]),
                                    metric=obj["metric"], extras=dict(obj.get("extras", {}))))
    return examples

def write_task_jsonl(examples: Iterable[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"id": ex.id, "prompt": ex.prompt, "targets": ex.targets,
                                "metric": ex.metric, **({"extras": ex.extras} if ex.extras else {})}) + "\n")

def run_task(task_name: str, examples: List[Example], adapter: ModelAdapter, *, batch_size=8) -> TaskResult:
    if not examples: return TaskResult(task=task_name, metric="none", score=0.0, correct=0, total=0)
    metric = examples[0].metric
    assert all(ex.metric == metric for ex in examples), f"任务 {task_name} 混合了多种指标"
    metric_fn = METRIC_FNS[metric]
    per_example, correct_sum, total = [], 0.0, 0
    start = time.perf_counter()
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        outputs = adapter.generate([ex.prompt for ex in chunk])
        for ex, out in zip(chunk, outputs):
            score = metric_fn(out, ex.targets, ex.extras)
            correct_sum += score; total += 1
            per_example.append({"id": ex.id, "prompt": ex.prompt, "prediction": out, "targets": ex.targets, "score": score})
    return TaskResult(task=task_name, metric=metric, score=correct_sum / total if total else 0.0,
                      correct=int(round(correct_sum)), total=total, per_example=per_example,
                      latency_ms=(time.perf_counter() - start) * 1000.0)

def run_leaderboard(tasks: Dict[str, List[Example]], adapter: ModelAdapter, *, batch_size=8) -> Leaderboard:
    results = [run_task(name, tasks[name], adapter, batch_size=batch_size) for name in sorted(tasks)]
    overall = sum(r.score for r in results) / len(results) if results else 0.0
    return Leaderboard(schema="leaderboard.v1", timestamp=time.time(), overall_score=overall, tasks=results)

def write_leaderboard(board: Leaderboard, path: Path, *, adapter_name: str, include_per_example=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": board.schema, "timestamp": board.timestamp,
        "overall_score": board.overall_score, "adapter": adapter_name,
        "tasks": [{"task": r.task, "metric": r.metric, "score": r.score, "correct": r.correct,
                    "total": r.total, "latency_ms": r.latency_ms,
                    **({"per_example": r.per_example} if include_per_example else {})}
                   for r in board.tasks]}, indent=2) + "\n")

def seed_fixture_tasks(target_dir: Path) -> None:
    """生成五个内置任务的 JSONL 文件。"""
    tasks = {
        "arithmetic": [Example(id=f"arith-{i:02d}", prompt=f"compute: {q}", targets=[a], metric="exact_match")
                       for i, (q, a) in enumerate([("2 + 2", "4"), ("7 - 3", "4"), ("6 * 4", "24"),
                                                    ("100 / 4", "25.0"), ("12 + 9", "21")])],
        "summary": [Example(id=f"sum-{i:02d}", prompt=f"summarize: {p}", targets=[t], metric="rouge_l")
                    for i, (p, t) in enumerate([("Cats are mammals. Mammals are warm blooded.", "cats are mammals"),
                                                ("Python uses indentation. Indentation defines blocks.", "python uses indentation"),
                                                ("The river flows east. Boats pass slowly.", "the river flows east"),
                                                ("Storms approach the coast. Waves rise quickly.", "storms approach the coast"),
                                                ("Bread bakes at high heat. Crust forms last.", "bread bakes at high heat")])],
        "code-exec": [Example(id=f"code-{i:02d}", prompt=f"python: {p}", targets=["ok"], metric="code_exec",
                              extras={"io_pairs": pairs})
                      for i, (p, pairs) in enumerate([("写一个函数 f，将其输入翻倍", [[1, 2], [3, 6]]),
                                                       ("写一个函数 f，将其输入加一", [[1, 2], [5, 6]]),
                                                       ("写一个函数 f，将其输入平方", [[2, 4], [3, 9]])])],
        "multiple-choice": [Example(id=f"mc-{i:02d}", prompt=f"choose: {q}", targets=t, metric="multiple_choice")
                            for i, (q, t) in enumerate([("A | 猫, B | 狗, C | 鸟", ["A"]),
                                                        ("A | 苹果, B | 汽车, C | 树", ["A"]),
                                                        ("A | 水, B | 铁, C | 木头", ["A"]),
                                                        ("A | 正方形, B | 三角形, C | 圆形", ["A"])])],
        "generation": [Example(id=f"gen-{i:02d}", prompt=f"write: {p}", targets=t, metric="substring_contains")
                       for i, (p, t) in enumerate([("hello world", ["hello"]), ("training language models", ["language"]),
                                                   ("evaluation harness", ["evaluation"]),
                                                   ("gradient accumulation", ["gradient"])])],
    }
    for name, examples in tasks.items():
        write_task_jsonl(examples, target_dir / f"{name}.jsonl")

def load_all_tasks(task_dir: Path) -> Dict[str, List[Example]]:
    tasks = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        tasks[path.stem] = load_task_jsonl(path)
    return tasks

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--out", type=Path, default=LEADERBOARD_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--include-per-example", action="store_true")
    parser.add_argument("--seed-fixtures", action="store_true")
    args = parser.parse_args()

    if args.seed_fixtures or not args.task_dir.exists() or not list(args.task_dir.glob("*.jsonl")):
        print(f"生成内置任务到 {args.task_dir}")
        seed_fixture_tasks(args.task_dir)

    tasks = load_all_tasks(args.task_dir)
    print(f"加载了 {len(tasks)} 个任务: {sorted(tasks)}")
    adapter = ToyAdapter()
    board = run_leaderboard(tasks, adapter, batch_size=args.batch_size)
    write_leaderboard(board, args.out, adapter_name=adapter.name, include_per_example=args.include_per_example)
    print(f"总体分数 = {board.overall_score:.3f}")
    for r in board.tasks:
        print(f"  {r.task:>16}  指标={r.metric:>18}  分数={r.score:0.3f}  ({r.correct}/{r.total})  延迟={r.latency_ms:.1f}ms")
    print(f"写入 {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 任务规范 | JSONL 文件，每行包含 prompt、targets、metric 和可选 extras |
| 指标 | 从 (prediction, targets, extras) 到 [0, 1] 范围内浮点数的函数 |
| 适配器 | 具有 `generate(prompts) -> list[str]` 方法的对象；唯一的模型特定代码 |
| 排行榜 | 包含每任务分数、总计数、延迟和总体平均分的 JSON 文件 |
| 代码执行指标 | 在受限命名空间中执行预测代码并与输入-输出对比较 |

---

## 5. 工程最佳实践

### 5.1 将评估框架集成到生产中的三个原则

- **锁定任务文件**。排行榜 JSON 应携带任务内容的哈希值，或者将 JSONL 文件与排行榜一起存档——否则当任务文件变化时分数会移动，你无法确定原因。
- **对比预测结果，不仅是分数**。`--include-per-example` 标志让你在看到分数下降时查看模型当时说了什么。
- **限制批次大小**。真正的适配器有速率限制。较小的批次大小使评估框架跨供应商兼容。

### 5.2 替换为真实模型

将真正的模型接入只需编写一个适配器：

```python
class HttpAdapter:
    name = "vendor.v1"
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint; self.api_key = api_key
    def generate(self, prompts):
        outputs = []
        for prompt in prompts:
            response = requests.post(self.endpoint, json={"prompt": prompt}, headers={"Authorization": f"Bearer {self.api_key}"})
            outputs.append(response.json()["text"])
        return outputs
```

在 `main()` 顶部将 `ToyAdapter` 替换为 `HttpAdapter`。评估框架、任务、指标和排行榜保持不变。

### 5.3 中文场景特别建议

- **对中文评估任务使用适当的归一化**：英文评估常用的小写化对中文没有意义。在 `normalize()` 中，对中文文本考虑做繁简转换和全角半角归一化。
- **中文摘要评估**：ROUGE-L 在中文上工作良好，但需要注意分词——一个中文字符可能是一个完整词元，而 ROUGE 的词元分割方式会影响分数。
- **代码执行任务中的中文注释**：如果模型的输出包含中文注释，`exec` 仍然可以正常工作——Python 3 支持 Unicode 标识符。

---

## 6. 常见错误

### 错误 1：适配器返回的输出数量与提示词数量不匹配

**现象：** 任务评分时数组索引越界或分数为 0。

**原因：** 适配器对一批 N 个提示词返回了少于 N 个输出。

**修复：** 在 `run_task` 中如果 `len(outputs) != len(chunk)` 则抛出一个值错误。

### 错误 2：代码执行指标中的命名空间不安全

**现象：** 模型的预测代码可以访问文件系统或网络。

**原因：** `exec` 的全局命名空间没有严格限制，或者在原始命名空间中运行。

**修复：** 使用剥离了危险内置函数的命名空间：`{"__builtins__": {"range": range, "len": len, ...}}`。

### 错误 3：任务文件中混合了不同的指标

**现象：** 任务的分数在 0 和 1 之间振荡，难以解读。

**原因：** 一个任务文件中的不同示例使用不同的指标，指标计算方式不同。

**修复：** 断言任务文件中所有示例的指标相同。一个任务文件 = 一个指标。

---

## 7. 面试考点

### Q1：评估框架中适配器模式解决了什么问题？（难度：⭐⭐）

**参考答案：** 适配器模式将模型特定代码与评估框架分开。不修改评估框架就可以切换模型（本地模型、API 调用、mock），不修改适配器就可以添加新任务。这是依赖倒置原则的体现——高层模块（评估框架）不依赖低层模块（具体模型），两者都依赖抽象（适配器接口）。

### Q2：ROUGE-L 与精确匹配有什么本质区别？各自适合什么场景？（难度：⭐⭐⭐）

**参考答案：** 精确匹配要求输出与参考完全相同，适合有确定正确答案的任务（算术、多项选择）。ROUGE-L 基于最长公共子序列计算 F1 分数，允许输出存在措辞差异但保留核心信息，适合摘要、翻译等输出多样化的任务。在生产中，对生成式任务使用精确匹配会低估模型的真实能力。

---

## 📚 小结

语言模型评估框架让你能够客观、可复现地比较不同模型的能力。你从零实现了五种评估指标、一个可替换的模型适配器和一个排行榜生成器。评估框架是现代 LLM 工程的基石——没有它，模型选择就变成了猜谜游戏。

下一节将构建假设生成器——一个通过温度调度采样和新颖性过滤来自动生成可测试研究假设的智能体。

---

## ✏️ 练习

1. 【实现】添加第六个任务，从零编写自定义指标（如 BLEU 或基于嵌入的语义相似度）。

2. 【实验】扩展 `code_exec` 以捕获 stdout 并接受预期的 stdout 列表作为 targets。

3. 【实现】添加一个排行榜 diff 命令：给定两个 `leaderboard.json` 文件，打印哪些任务发生了变化以及变化量。

4. 【思考】为适配器调用添加超时——将适配器调用包装在超时中；在排行榜中单独列出超时列。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 评估框架 | `code/main.py` | 完整评估框架：加载 → 运行 → 评分 → 排行榜 |
| 内置任务数据 | `outputs/tasks/` | 五个任务的 JSONL 文件 |

---

## 📖 参考资料

1. [GitHub] LM Evaluation Harness. https://github.com/EleutherAI/lm-evaluation-harness
2. [HuggingFace] Lighteval. https://github.com/huggingface/lighteval
3. [论文] Lin. "ROUGE: A Package for Automatic Evaluation of Summaries". ACL 2004. https://aclanthology.org/W04-1013/
4. [官方文档] Python `ast` 模块 — 安全表达式求值. https://docs.python.org/3/library/ast.html
