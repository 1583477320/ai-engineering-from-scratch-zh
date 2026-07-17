"""对比纯提示词运行和工作台引导运行。

智能体是规则存根；重点是周围的面。
第一轮没有工作面，第二轮全部七个面都在。
对比两者的失败模式。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# 工作台的七个面
WORKBENCH_SURFACES = [
    "instructions",
    "state",
    "scope",
    "feedback",
    "verification",
    "review",
    "handoff",
]


@dataclass
class RepoTask:
    """一个仓库任务。"""
    description: str
    allowed_files: list[str]
    forbidden_files: list[str]
    acceptance: list[str]


@dataclass
class RunResult:
    """一次运行的结果。"""
    label: str
    surfaces_present: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tests_run: bool = False
    declared_success: bool = False
    actually_passing: bool = False
    notes: list[str] = field(default_factory=list)

    def missing_surfaces(self) -> list[str]:
        return [s for s in WORKBENCH_SURFACES if s not in self.surfaces_present]


def stub_agent(task: RepoTask, surfaces: list[str]) -> RunResult:
    """确定性存根智能体——模拟两种运行模式。"""
    result = RunResult(label="prompt-only" if not surfaces else "workbench")
    result.surfaces_present = list(surfaces)

    has_scope = "scope" in surfaces
    has_state = "state" in surfaces
    has_verification = "verification" in surfaces
    has_feedback = "feedback" in surfaces

    # 有范围约束时只修改允许的文件，否则跑偏
    if has_scope:
        result.files_touched = [f for f in task.allowed_files]
    else:
        result.files_touched = [*task.allowed_files, "README.md", "scripts/release.sh"]
        result.notes.append("touched unrelated files because scope was missing")

    # 有反馈捕获时真正运行测试，否则猜测输出
    if has_feedback:
        result.tests_run = True
        result.notes.append("captured stdout/stderr/exit code from the test run")
    else:
        result.notes.append("never ran the test command, guessed at output")

    # 有验证门控时声明成功前检查，否则虚构成功
    if has_verification:
        result.actually_passing = True
        result.declared_success = True
        result.notes.append("verification gate proved acceptance criteria met")
    else:
        result.declared_success = True
        result.actually_passing = False
        result.notes.append("declared success without running acceptance checks")

    # 没有状态文件时，下个会话从零开始
    if not has_state:
        result.notes.append("no state file written, next session restarts from zero")

    return result


def failure_report(result: RunResult) -> dict[str, object]:
    """生成失败模式报告。"""
    return {
        "label": result.label,
        "missing_surfaces": result.missing_surfaces(),
        "off_scope_writes": [
            f for f in result.files_touched if f not in {"app.py", "test_app.py"}
        ],
        "tests_run": result.tests_run,
        "declared_success": result.declared_success,
        "actually_passing": result.actually_passing,
        "notes": result.notes,
    }


def main() -> None:
    task = RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"],
    )

    prompt_only = stub_agent(task, surfaces=[])
    workbench = stub_agent(task, surfaces=WORKBENCH_SURFACES)

    print("=== 纯提示词运行 ===")
    for k, v in failure_report(prompt_only).items():
        print(f"  {k}: {v}")
    print()

    print("=== 工作台引导运行 ===")
    for k, v in failure_report(workbench).items():
        print(f"  {k}: {v}")

    # 写入失败模式报告
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "failure_modes.json"
    out_path.write_text(json.dumps(failure_report(prompt_only), indent=2) + "\n")
    print(f"\n已写入 {out_path.name}")

    # 总结
    print("\n=== 对比总结 ===")
    print(f"纯提示词：{len(prompt_only.missing_surfaces())} 个面缺失 → 声明成功但实际失败")
    print(f"工作台：{len(workbench.missing_surfaces())} 个面缺失 → 声明成功且实际通过")


if __name__ == "__main__":
    main()
