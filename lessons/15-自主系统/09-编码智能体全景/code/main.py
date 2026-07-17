"""CodeAct vs JSON 工具调用脚手架对比——纯标准库。

两个脚手架使用相同的存根"模型"（确定性规则），
所以对比隔离了脚手架而非模型质量。指标：
  - 解决的任务数
  - 使用的轮数
  - 每次动作的爆炸半径（一个动作可以触及的文件数）

教学用途：脚手架是负载承载的。OpenHands（arXiv:2407.16741）
明确做了 CodeAct 的押注；JSON 工具调用在管理服务中保持主导。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ── 迷你仓库 ──────────────────────────────────────────────

INITIAL_REPO = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "util.py": "def lower(s):\n    return s.upper()\n",
    "cli.py": "VERSION = 'v0.0'\n",
}

TESTS = [
    ("app.py", "add(2, 3) == 5"),
    ("util.py", "lower('AB') == 'ab'"),
    ("cli.py", "VERSION == 'v1.0'"),
]

FIXES: dict[str, tuple[str, str]] = {
    "app.py": ("a - b", "a + b"),
    "util.py": ("s.upper()", "s.lower()"),
    "cli.py": ("v0.0", "v1.0"),
}


def run_tests(repo: dict[str, str]) -> list[bool]:
    """确定性存根：模拟测试套件。"""
    results = []
    for path, _expr in TESTS:
        src = repo.get(path, "")
        if path == "app.py":
            results.append("return a + b" in src)
        elif path == "util.py":
            results.append("return s.lower()" in src)
        elif path == "cli.py":
            results.append("VERSION = 'v1.0'" in src)
        else:
            results.append(False)
    return results


def _apply_fix(repo: dict[str, str], path: str) -> bool:
    """原地应用修复。返回是否应用了修复。"""
    rule = FIXES.get(path)
    if rule is None:
        return False
    old, new = rule
    repo[path] = repo[path].replace(old, new)
    return True


# ── JSON 工具调用脚手架 ──────────────────────────────────

@dataclass
class JsonScaffold:
    """每次动作一个 JSON 载荷——安全、可审计、组合性有限。"""
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0

    def step(self) -> str:
        """返回一个 JSON 动作。"""
        self.turns += 1
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                return json.dumps({"tool": "edit", "path": path})
        return json.dumps({"tool": "done"})

    def blast_radius(self) -> int:
        return 1  # 每个动作恰好触及一个文件

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if json.loads(action).get("tool") == "done":
                break
        return sum(run_tests(self.repo)), self.turns


# ── CodeAct 脚手架 ────────────────────────────────────────

@dataclass
class CodeActScaffold:
    """一个代码片段可以编辑多个文件——组合性高，需要加固沙箱。"""
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0
    worst_touched: int = 0

    def step(self) -> str:
        """返回一个可能编辑多个文件的 Python 片段。"""
        self.turns += 1
        snippet_lines = []
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                snippet_lines.append(f"fs.write('{path}', ...)")
        self.worst_touched = max(self.worst_touched, len(snippet_lines))
        if not snippet_lines:
            return "done()"
        return "; ".join(snippet_lines)

    def blast_radius(self) -> int:
        return self.worst_touched

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if action == "done()":
                break
        return sum(run_tests(self.repo)), self.turns


# ── 报告 ──────────────────────────────────────────────────

def report(name: str, passed: int, turns: int, blast: int) -> None:
    total = len(TESTS)
    print(f"  {name:<18}  通过 {passed}/{total}  轮数 {turns:>2}  "
          f"爆炸半径 {blast}")


def main() -> None:
    print("=" * 70)
    print("CodeAct vs JSON 工具调用脚手架（阶段 15，第 9 课）")
    print("=" * 70)
    print()
    print("同一存根模型，三 bug 仓库。纯脚手架对比。")
    print("-" * 70)

    js = JsonScaffold()
    passed, turns = js.run()
    report("JSON 工具调用", passed, turns, js.blast_radius())

    ca = CodeActScaffold()
    passed, turns = ca.run()
    report("CodeAct (存根)", passed, turns, ca.blast_radius())

    print()
    print("=" * 70)
    print("要点: 脚手架不是布景。它是产品。")
    print("-" * 70)
    print("  同一模型，两个脚手架，不同轮数。")
    print("  CodeAct 将多个编辑压缩为一个动作。")
    print("  代价是爆炸半径：CodeAct 需要加固沙箱")
    print("  隔离（OpenHands 使用 Docker）。JSON 工具调用通过构造获得安全性")
    print("  因为每个动作独立验证。")
    print("  两者都不严格更好；权衡是要审计什么。")


if __name__ == "__main__":
    main()
