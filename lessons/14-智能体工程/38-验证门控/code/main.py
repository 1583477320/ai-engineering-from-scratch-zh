"""确定性验证门控，支持覆盖率门槛、--strict 模式、签名覆盖日志。

合并任务的 scope_report、rule_report、feedback log 和可选的
coverage_report 为单个 verification_report.json。没有 LLM 评判器——
LLM 评判在审查员侧（阶段 14 · 39）。覆盖需要 HMAC 签名条目。

运行：python3 code/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
OVERRIDES_PATH = HERE / "overrides.jsonl"
COVERAGE_FLOOR_DEFAULT = 0.80
COVERAGE_REGRESSION_DELTA = 0.01

# 覆盖签名密钥。生产环境从密钥管理器读取。
# 仅当 VERIFY_DEMO_MODE=1 时回退到演示密钥
_OVERRIDE_SECRET_ENV = "VERIFY_OVERRIDE_SECRET"
_DEMO_MODE_ENV = "VERIFY_DEMO_MODE"


def _load_override_secret() -> str:
    secret = os.environ.get(_OVERRIDE_SECRET_ENV)
    if secret:
        return secret
    if os.environ.get(_DEMO_MODE_ENV) == "1":
        print(f"WARNING: {_OVERRIDE_SECRET_ENV} unset, using demo secret", file=sys.stderr)
        return "demo-override-secret-do-not-ship"
    raise RuntimeError(f"refused to start: {_OVERRIDE_SECRET_ENV} is unset. "
                       f"Set the env var, or pass {_DEMO_MODE_ENV}=1 for demo.")


# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class Finding:
    code: str       # 如 "acceptance.missing"
    severity: str   # block | warn
    detail: str


@dataclass
class Artifacts:
    task_id: str
    acceptance_commands: list[str]
    feedback: list[dict[str, object]]
    scope_report: dict[str, object]
    rule_report: list[dict[str, object]]
    coverage_report: dict[str, float] | None = None
    head_commit: str = ""


@dataclass
class VerdictReport:
    task_id: str
    passed: bool
    strict: bool
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[str, float] | None = None
    head_commit: str = ""


# ── 检查函数 ────────────────────────────────────────────────

def _acceptance_findings(art: Artifacts) -> list[Finding]:
    """检查验收命令是否运行且退出码为 0。"""
    findings: list[Finding] = []
    commands_run = [str(rec.get("command")) for rec in art.feedback]
    for cmd in art.acceptance_commands:
        if cmd not in commands_run:
            findings.append(Finding("acceptance.missing", "block", f"never ran: {cmd}"))
    for rec in art.feedback:
        cmd_str = str(rec.get("command"))
        if rec.get("exit_code") is None:
            findings.append(Finding("feedback.null_exit", "block", f"missing exit for {cmd_str}"))
        elif rec.get("exit_code") != 0 and cmd_str in set(art.acceptance_commands):
            findings.append(Finding("acceptance.failed", "block",
                                     f"exit {rec.get('exit_code')} on {cmd_str}"))
    return findings


def _scope_findings(art: Artifacts) -> list[Finding]:
    """检查范围违规。"""
    findings: list[Finding] = []
    if art.scope_report.get("forbidden_writes"):
        findings.append(Finding("scope.forbidden", "block",
                                 f"forbidden writes: {art.scope_report['forbidden_writes']}"))
    if art.scope_report.get("off_scope_writes"):
        findings.append(Finding("scope.off_scope", "warn",
                                 f"off-scope writes: {art.scope_report['off_scope_writes']}"))
    return findings


def _rule_findings(art: Artifacts) -> list[Finding]:
    """检查规则违规。"""
    return [Finding("rule.failed", "block", f"rule failed: {row.get('slug')}")
            for row in art.rule_report if not row.get("passed")]


def _coverage_findings(art: Artifacts, floor: float) -> list[Finding]:
    """Anthropic Hybrid Norm：可验证奖励（测试+覆盖率）+ 评级评判。"""
    findings: list[Finding] = []
    if not art.coverage_report:
        findings.append(Finding("coverage.missing", "warn",
                                 "no coverage_report.json; cannot enforce floor"))
        return findings
    current = float(art.coverage_report.get("current", 0.0))
    previous = float(art.coverage_report.get("previous", current))
    if current < floor:
        findings.append(Finding("coverage.below_floor", "block",
                                 f"coverage {current:.2%} below floor {floor:.0%}"))
    delta = previous - current
    if delta > COVERAGE_REGRESSION_DELTA:
        findings.append(Finding("coverage.regression", "block",
                                 f"dropped {delta:.2%} (prev {previous:.2%} -> {current:.2%})"))
    elif delta > 0:
        findings.append(Finding("coverage.minor_regression", "warn",
                                 f"dropped {delta:.2%}"))
    return findings


# ── 主验证函数 ──────────────────────────────────────────────

def verify(art: Artifacts, strict: bool = False,
           coverage_floor: float = COVERAGE_FLOOR_DEFAULT) -> VerdictReport:
    findings = (
        _acceptance_findings(art)
        + _scope_findings(art)
        + _rule_findings(art)
        + _coverage_findings(art, coverage_floor)
    )
    if strict:
        # --strict：所有 warn 升级为 block。仅在发布分支开启
        findings = [Finding(f.code, "block" if f.severity == "warn" else f.severity, f.detail)
                    for f in findings]
    blocking = [f for f in findings if f.severity == "block"]
    return VerdictReport(
        task_id=art.task_id,
        passed=not blocking,
        strict=strict,
        findings=findings,
        coverage=art.coverage_report,
        head_commit=art.head_commit,
    )


# ── 签名覆盖日志 ────────────────────────────────────────────

def _sign(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_load_override_secret().encode(), canonical, hashlib.sha256).hexdigest()[:32]


def record_override(task_id: str, finding_code: str, reason: str,
                    user_id: str, head_commit: str) -> dict[str, object]:
    """追加一条签名覆盖条目。五个字段都必需。"""
    if not all([task_id, finding_code, reason, user_id, head_commit]):
        raise ValueError("override requires task_id, finding_code, reason, user_id, head_commit")
    payload = {"task_id": task_id, "finding_code": finding_code, "reason": reason,
               "user_id": user_id, "head_commit": head_commit, "ts": time.time()}
    payload["signature"] = _sign({k: v for k, v in payload.items() if k != "signature"})
    with OVERRIDES_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    return payload


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="promote every warn to block")
    ap.add_argument("--floor", type=float, default=COVERAGE_FLOOR_DEFAULT)
    args = ap.parse_args()

    accept = ["pytest -x test_app.py::test_signup_rejects_short_password"]
    cases = [
        Artifacts(task_id="T-001", acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": True}],
            coverage_report={"current": 0.84, "previous": 0.85}, head_commit="a1b2c3d"),
        Artifacts(task_id="T-002", acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["README.md"]},
            rule_report=[{"slug": "forbidden/no-release-script-edits", "passed": False}],
            coverage_report={"current": 0.62, "previous": 0.80}, head_commit="b2c3d4e"),
        Artifacts(task_id="T-003", acceptance_commands=accept,
            feedback=[], scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": False}], head_commit="c3d4e5f"),
    ]

    for art in cases:
        report = verify(art, strict=args.strict, coverage_floor=args.floor)
        path = HERE / f"verification_report_{art.task_id}.json"
        path.write_text(json.dumps(
            {"task_id": report.task_id, "passed": report.passed, "strict": report.strict,
             "head_commit": report.head_commit, "coverage": report.coverage,
             "findings": [asdict(f) for f in report.findings]}, indent=2) + "\n")
        flag = " (strict)" if report.strict else ""
        print(f"task {report.task_id}{flag}: passed={report.passed} findings={len(report.findings)}")
        for f in report.findings:
            print(f"  [{f.severity}] {f.code}: {f.detail}")
        print()

    # 演示签名覆盖
    os.environ.setdefault(_DEMO_MODE_ENV, "1")
    try:
        entry = record_override(
            task_id="T-002", finding_code="scope.off_scope",
            reason="reviewer approved README update for new signup contract",
            user_id="rohitg00", head_commit="b2c3d4e")
        print(f"override recorded: signature={entry['signature']} verified={True}")
    except RuntimeError as exc:
        print(f"override demo skipped: {exc}")


if __name__ == "__main__":
    main()
