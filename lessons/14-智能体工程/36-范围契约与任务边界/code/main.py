"""范围契约检查器：glob 匹配、违规预算、严重性级别、多契约合并。

加载任务级 scope_contract.json 和 RunSummary（修改的文件、运行的命令、
已用分钟数、网络请求），生成带严重性标签的 Finding 列表，
应用违规预算使门控在实践中可用，支持多契约（项目级 + 任务级）合并。

运行：python3 code/main.py
"""

from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent


# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class ScopeContract:
    """范围契约——约束一个任务的文件范围、时间预算和审批边界。"""
    task_id: str
    goal: str
    allowed_files: list[str]           # glob 模式，智能体可以修改的文件
    forbidden_files: list[str]         # glob 模式，智能体绝对不能碰的文件
    acceptance_criteria: list[str]     # 验收命令列表
    rollback_plan: str                 # 回滚计划
    approvals_required: list[str] = field(default_factory=list)
    time_budget_minutes: int | None = None
    network_egress: list[str] | None = None  # None=无限制, []=拒绝所有, [...]=白名单
    violation_budget: int = 0                # 违规预算
    docs_paths_soft: list[str] = field(default_factory=lambda: ["docs/**", "README.md", "**/*.md"])


@dataclass
class RunSummary:
    """一次智能体运行的摘要。"""
    touched_files: list[str]
    commands_run: list[str]
    elapsed_minutes: float = 0.0
    network_hosts: list[str] = field(default_factory=list)


@dataclass
class Finding:
    """检查发现——带严重性标签的违规记录。"""
    code: str        # 如 "scope.forbidden"
    severity: str    # block | warn | info
    detail: str


@dataclass
class ScopeReport:
    """范围检查报告。"""
    task_id: str
    in_scope_writes: list[str]
    off_scope_writes: list[str]
    forbidden_writes: list[str]
    soft_off_scope_writes: list[str]
    missing_acceptance: list[str]
    findings: list[Finding]
    over_budget: bool

    def passed(self) -> bool:
        """检查是否通过：没有超预算且没有 block 级违规。"""
        return not self.over_budget and not any(f.severity == "block" for f in self.findings)


# ── 工具函数 ────────────────────────────────────────────────

def matches_any(path: str, patterns: list[str]) -> bool:
    """检查路径是否匹配任意一个 glob 模式。"""
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _min_optional(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _merge_egress(a: list[str] | None, b: list[str] | None) -> list[str] | None:
    """合并网络出口白名单。None 延后到对方，两个列表取交集。"""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return sorted(set(a) & set(b))


# ── 多契约合并 ─────────────────────────────────────────────

def merge_contracts(parent: ScopeContract, child: ScopeContract) -> ScopeContract:
    """最小权限合并：交集（允许）、并集（禁止）、最严格（时间）。"""
    return ScopeContract(
        task_id=child.task_id,
        goal=child.goal or parent.goal,
        allowed_files=sorted(set(parent.allowed_files) & set(child.allowed_files)),
        forbidden_files=sorted(set(parent.forbidden_files) | set(child.forbidden_files)),
        acceptance_criteria=list(dict.fromkeys(parent.acceptance_criteria + child.acceptance_criteria)),
        rollback_plan=child.rollback_plan or parent.rollback_plan,
        approvals_required=list(dict.fromkeys(parent.approvals_required + child.approvals_required)),
        time_budget_minutes=_min_optional(parent.time_budget_minutes, child.time_budget_minutes),
        network_egress=_merge_egress(parent.network_egress, child.network_egress),
        violation_budget=min(parent.violation_budget, child.violation_budget),
        docs_paths_soft=sorted(set(parent.docs_paths_soft) | set(child.docs_paths_soft)),
    )


# ── 范围检查 ──────────────────────────────────────────────

def scope_check(contract: ScopeContract, run: RunSummary) -> ScopeReport:
    """对照范围契约检查一次运行的摘要。

    返回值包含：
    - in_scope / off_scope / forbidden / soft_off_scope：按严重性分组的路径
    - missing_acceptance：未运行的验收命令
    - findings：带严重性标签的发现
    - over_budget：警告级违规是否超出预算
    """
    in_scope: list[str] = []
    off_scope: list[str] = []
    soft_off_scope: list[str] = []
    forbidden: list[str] = []
    for path in run.touched_files:
        if matches_any(path, contract.forbidden_files):
            forbidden.append(path)
        elif matches_any(path, contract.allowed_files):
            in_scope.append(path)
        elif matches_any(path, contract.docs_paths_soft):
            soft_off_scope.append(path)   # 文档越界，通常是 warn/info
        else:
            off_scope.append(path)

    missing = [c for c in contract.acceptance_criteria if c not in run.commands_run]

    findings: list[Finding] = []
    if forbidden:
        findings.append(Finding("scope.forbidden", "block", f"禁止写入的文件：{forbidden}"))
    if off_scope:
        findings.append(Finding("scope.off_scope", "warn", f"越界写入的文件：{off_scope}"))
    if soft_off_scope:
        findings.append(Finding("scope.soft_off_scope", "info", f"文档越界：{soft_off_scope}"))
    if missing:
        findings.append(Finding("acceptance.missing", "block", f"验收命令未运行：{missing}"))
    if contract.time_budget_minutes is not None and run.elapsed_minutes > contract.time_budget_minutes:
        findings.append(Finding("time.over_budget", "block",
                                f"已用 {run.elapsed_minutes:.1f}m，超出预算 {contract.time_budget_minutes}m"))
    if contract.network_egress is not None and run.network_hosts:
        bad_hosts = [h for h in run.network_hosts if h not in contract.network_egress]
        if bad_hosts:
            findings.append(Finding("network.unallowed_host", "block",
                                    f"访问了未白名单的主机：{bad_hosts}"))

    warn_count = sum(1 for f in findings if f.severity == "warn")
    over_budget = warn_count > contract.violation_budget

    return ScopeReport(
        task_id=contract.task_id,
        in_scope_writes=in_scope,
        off_scope_writes=off_scope,
        forbidden_writes=forbidden,
        soft_off_scope_writes=soft_off_scope,
        missing_acceptance=missing,
        findings=findings,
        over_budget=over_budget,
    )


# ── 归档 ──────────────────────────────────────────────────

def archive(report: ScopeReport) -> Path:
    """将检查报告归档为 JSON。"""
    out = HERE / "closed" / f"{report.task_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "task_id": report.task_id,
            "findings": [asdict(f) for f in report.findings],
            "in_scope": report.in_scope_writes,
            "off_scope": report.off_scope_writes,
            "soft_off_scope": report.soft_off_scope_writes,
            "passed": report.passed(),
            "closed_at": time.time(),
        }, indent=2) + "\n"
    )
    return out


# ── 演示 ──────────────────────────────────────────────────

def main() -> None:
    # 项目级契约
    project_wide = ScopeContract(
        task_id="P-PROJECT",
        goal="项目级默认值",
        allowed_files=["app.py", "test_app.py", "lib/**/*.py"],
        forbidden_files=["scripts/release.sh", "config/prod.yaml"],
        acceptance_criteria=[],
        rollback_plan="回滚并重新部署",
        approvals_required=["任何新的运行时依赖"],
        time_budget_minutes=60,
        violation_budget=1,
        network_egress=["api.openai.com", "api.anthropic.com"],
    )

    # 任务级契约
    task = ScopeContract(
        task_id="T-001",
        goal="为 /signup 添加输入验证",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["migrations/**"],
        acceptance_criteria=["pytest -x test_app.py::test_signup_rejects_short_password"],
        rollback_plan="回滚提交并部署上一个构建标签",
        approvals_required=[],
        time_budget_minutes=30,
        violation_budget=0,
        network_egress=["api.anthropic.com"],
    )

    # 合并契约（最小权限）
    effective = merge_contracts(project_wide, task)

    # 合规运行
    clean = RunSummary(
        touched_files=["app.py", "test_app.py"],
        commands_run=["pytest -x test_app.py::test_signup_rejects_short_password"],
        elapsed_minutes=12.4,
        network_hosts=["api.anthropic.com"],
    )

    # 跑偏运行
    creep = RunSummary(
        touched_files=["app.py", "README.md", "scripts/release.sh", "migrations/001_init.sql"],
        commands_run=[],
        elapsed_minutes=42.1,
        network_hosts=["api.anthropic.com", "evil.example"],
    )

    clean_report = scope_check(effective, clean)
    creep_report = scope_check(effective, creep)

    print("effective contract:", json.dumps(asdict(effective), indent=2))
    print("\n合规运行结果：")
    for f in clean_report.findings:
        print(f"  [{f.severity}] {f.code}: {f.detail}")
    print(f"  passed={clean_report.passed()} over_budget={clean_report.over_budget}")

    print("\n跑偏运行结果：")
    for f in creep_report.findings:
        print(f"  [{f.severity}] {f.code}: {f.detail}")
    print(f"  passed={creep_report.passed()} over_budget={creep_report.over_budget}")

    archive(clean_report)
    archive(creep_report)
    print(f"\n已归档到 {HERE / 'closed'}/")


if __name__ == "__main__":
    main()
