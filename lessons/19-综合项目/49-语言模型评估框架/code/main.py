"""语言模型评估框架从零实现。

运行：python3 code/main.py
"""
from __future__ import annotations
import ast, json, operator, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

OUT_DIR = Path(__file__).parent.parent / "outputs"
TASKS_DIR = OUT_DIR / "tasks"


@dataclass
class Example:
    id: str; prompt: str; targets: List[str]; metric: str
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass
class TaskResult:
    task: str; metric: str; score: float; correct: int; total: int
    latency_ms: float = 0.0


class ModelAdapter(Protocol):
    def generate(self, prompts: Sequence[str]) -> List[str]: ...
    @property
    def name(self) -> str: ...


class ToyAdapter:
    name = "toy.v1"

    def generate(self, prompts: Sequence[str]) -> List[str]:
        return [self._answer(p) for p in prompts]

    def _answer(self, prompt: str) -> str:
        text = prompt.strip()
        if text.startswith("compute:"):
            try:
                return str(_safe_eval_arith(text[len("compute:"):].strip()))
            except:
                return ""
        if text.startswith("summarize:"):
            sents = re.split(r"(?<=[.!?])\s+", text[len("summarize:"):].strip())
            return sents[0] if sents else text
        if text.startswith("python:"):
            body = text[len("python:"):].strip()
            for k, v in [("double", "def f(x):\n    return x * 2\n"),
                         ("increment", "def f(x):\n    return x + 1\n"),
                         ("square", "def f(x):\n    return x * x\n")]:
                if k in body:
                    return v
            return "def f(x):\n    return x\n"
        if text.startswith("choose:"):
            return text[len("choose:"):].strip().split("|", 1)[0].strip()[:1].upper()
        if text.startswith("write:"):
            return text[len("write:"):].strip()
        return text


_ARITH_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv}


def _safe_eval_arith(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError(f"不支持: {ast.dump(node)}")


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def metric_exact_match(p, t, e):
    np = normalize(p)
    return 1.0 if any(normalize(tg) == np for tg in t) else 0.0


def metric_substring(p, t, e):
    np = normalize(p)
    return 1.0 if any(normalize(tg) in np for tg in t) else 0.0


def metric_mc(p, t, e):
    return 1.0 if p.strip()[:1].upper() in {tg.strip()[:1].upper() for tg in t} else 0.0


def metric_code_exec(p, t, e):
    pairs = e.get("io_pairs") or []
    if not pairs:
        return 0.0
    safe = {"__builtins__": {"range": range, "len": len, "int": int, "float": float}}
    local = {}
    try:
        exec(p, safe, local)
    except:
        return 0.0
    fn = local.get("f")
    if not callable(fn):
        return 0.0
    correct = sum(1 for pair in pairs if isinstance(pair, list) and len(pair) == 2 and fn(pair[0]) == pair[1])
    return correct / len(pairs)


METRICS = {"exact_match": metric_exact_match, "substring_contains": metric_substring,
           "multiple_choice": metric_mc, "code_exec": metric_code_exec}


def run_task(name: str, examples: List[Example], adapter: ModelAdapter, batch_size=8) -> TaskResult:
    if not examples:
        return TaskResult(task=name, metric="none", score=0.0, correct=0, total=0)
    metric = examples[0].metric
    fn = METRICS[metric]
    correct = 0
    total = 0
    start = time.perf_counter()
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        outputs = adapter.generate([ex.prompt for ex in chunk])
        for ex, out in zip(chunk, outputs):
            correct += fn(out, ex.targets, ex.extras)
            total += 1
    return TaskResult(task=name, metric=metric, score=correct / total if total else 0.0,
                      correct=int(round(correct)), total=total,
                      latency_ms=(time.perf_counter() - start) * 1000)


def load_task_jsonl(path: Path) -> List[Example]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            examples.append(Example(id=str(obj.get("id", f"ex-{num}")), prompt=obj["prompt"],
                                    targets=list(obj["targets"]), metric=obj["metric"],
                                    extras=dict(obj.get("extras", {}))))
    return examples


def seed_fixtures(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "arithmetic": [Example(id="arith-00", prompt="compute: 2 + 2", targets=["4"], metric="exact_match"),
                       Example(id="arith-01", prompt="compute: 7 - 3", targets=["4"], metric="exact_match"),
                       Example(id="arith-02", prompt="compute: 6 * 4", targets=["24"], metric="exact_match")],
        "summary": [Example(id="sum-00", prompt="summarize: Cats are mammals. Mammals are warm blooded.",
                            targets=["cats are mammals"], metric="rouge_l"),
                    Example(id="sum-01", prompt="summarize: Python uses indentation. Indentation defines blocks.",
                            targets=["python uses indentation"], metric="rouge_l")],
        "code-exec": [Example(id="code-00", prompt="python: write a function f that doubles its input",
                              targets=["ok"], metric="code_exec", extras={"io_pairs": [[1, 2], [3, 6]]})],
    }
    for name, examples in fixtures.items():
        path = target_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in examples:
                data = {"id": ex.id, "prompt": ex.prompt, "targets": ex.targets, "metric": ex.metric}
                if ex.extras:
                    data["extras"] = ex.extras
                f.write(json.dumps(data) + "\n")


def main() -> int:
    if not TASKS_DIR.exists():
        seed_fixtures(TASKS_DIR)
    tasks = {p.stem: load_task_jsonl(p) for p in sorted(TASKS_DIR.glob("*.jsonl"))}
    print(f"加载了 {len(tasks)} 个任务: {sorted(tasks)}")
    adapter = ToyAdapter()
    results = [run_task(name, examples, adapter) for name, examples in sorted(tasks.items())]
    overall = sum(r.score for r in results) / len(results) if results else 0.0
    print(f"总体分数 = {overall:.3f}")
    for r in results:
        print(f"  {r.task:>16}  指标={r.metric:>18}  分数={r.score:0.3f}  ({r.correct}/{r.total})  延迟={r.latency_ms:.1f}ms")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board = {"schema": "leaderboard.v1", "overall_score": overall, "adapter": adapter.name,
             "tasks": [{"task": r.task, "metric": r.metric, "score": r.score, "correct": r.correct, "total": r.total} for r in results]}
    (OUT_DIR / "leaderboard.json").write_text(json.dumps(board, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
