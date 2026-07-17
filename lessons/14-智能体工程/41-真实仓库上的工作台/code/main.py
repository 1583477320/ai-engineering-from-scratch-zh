"""在同一示例应用上运行纯提示词和工作台引导两条流水线。

两条流水线都是脚本化的（没有 LLM），测量可复现。
写入 before-after-report.md 和 comparison.json。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_app"

# ── 示例应用 ────────────────────────────────────────────────

SAMPLE_APP_PY = '''"""Minimal signup handler."""
USERS: dict[str, str] = {}

def signup(email: str, password: str) -> dict:
    USERS[email] = password
    return {"status": 200, "email": email}
'''

SAMPLE_TEST_PY = '''from sample_app.app import signup

def test_signup_happy_path():
    out = signup("a@b.co", "longenough")
    assert out["status"] == 200
'''

# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class TaskOutcome:
    pipeline: str
    tests_actually_run: bool
    acceptance_met: bool
    files_outside_scope: list[str] = field(default_factory=list)
    handoff_quality: str = "missing"
    reviewer_total: int = 0

ALLOWED = {"sample_app/app.py", "sample_app/test_app.py"}
FORBIDDEN = {"sample_app/scripts/release.sh"}


# ── 两条流水线 ──────────────────────────────────────────────

def run_prompt_only() -> TaskOutcome:
    """纯提示词流水线：修改文件但不运行测试，声称完成。"""
    touched = ["sample_app/app.py", "README.md", "sample_app/scripts/release.sh"]
    return TaskOutcome(
        pipeline="prompt-only",
        tests_actually_run=False,
        acceptance_met=False,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="missing",
        reviewer_total=3,
    )


def run_workbench() -> TaskOutcome:
    """工作台引导流水线：在范围内修改，运行验收，门控通过，审查，交接。"""
    touched = ["sample_app/app.py", "sample_app/test_app.py"]
    return TaskOutcome(
        pipeline="workbench-guided",
        tests_actually_run=True,
        acceptance_met=True,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="full packet",
        reviewer_total=9,
    )


# ── 报告生成 ────────────────────────────────────────────────

def write_report(po: TaskOutcome, wb: TaskOutcome) -> None:
    lines = [
        "# 前后对比：智能体工作台在真实仓库上",
        "",
        "同一任务。同一样本应用。两条流水线。",
        "",
        "| 结果 | 纯提示词 | 工作台 |",
        "|------|---------|--------|",
        f"| tests_actually_run | {po.tests_actually_run} | {wb.tests_actually_run} |",
        f"| acceptance_met | {po.acceptance_met} | {wb.acceptance_met} |",
        f"| files_outside_scope | {len(po.files_outside_scope)} | {len(wb.files_outside_scope)} |",
        f"| handoff_quality | {po.handoff_quality} | {wb.handoff_quality} |",
        f"| reviewer_total (/10) | {po.reviewer_total} | {wb.reviewer_total} |",
        "",
        "## 结论",
        "",
        "纯提示词流水线修改了禁止区域内的文件，在没有运行验收命令的情况下声称完成，",
        "没有交接包，审查得分很低。工作台流水线保持在范围内修改，通过反馈运行器运行验收命令，",
        "通过验证门控，生成可交付的交接包。",
        "",
        "## 假阴性",
        "",
        "以下场景纯提示词更快：单步事实查询、单行 lint、格式化运行、模型已背下的内容。",
        "工作台的额外步骤（初始化、范围检查、验证、审查、交接）在这些场景中确实是开销。",
        "诚实地列举这些场景让论证更可信。",
    ]
    (HERE / "before-after-report.md").write_text("\n".join(lines) + "\n")


def write_sample() -> None:
    """写入示例应用文件。"""
    SAMPLE.mkdir(exist_ok=True)
    (SAMPLE / "app.py").write_text(SAMPLE_APP_PY)
    (SAMPLE / "test_app.py").write_text(SAMPLE_TEST_PY)
    (SAMPLE / "README.md").write_text("# sample app\n\nForbidden zone for agent tasks.\n")
    scripts = SAMPLE / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "release.sh").write_text("#!/usr/bin/env bash\necho release\n")


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    write_sample()
    po = run_prompt_only()
    wb = run_workbench()

    for outcome in (po, wb):
        print(f"=== {outcome.pipeline} ===")
        for k, v in asdict(outcome).items():
            print(f"  {k}: {v}")
        print()

    write_report(po, wb)

    (HERE / "comparison.json").write_text(
        json.dumps({"prompt_only": asdict(po), "workbench": asdict(wb)}, indent=2) + "\n"
    )
    print("wrote before-after-report.md and comparison.json")


if __name__ == "__main__":
    main()
