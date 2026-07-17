"""三层评估框架 + 评估器-优化器循环 + CI 门控。

评估用例：基准测试（类 SWE-bench）、自定义评估（LLM-judge）、在线评估（护栏）。
聚合器：通过率、回归 vs 基线、CI 裁决。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class EvalCase:
    """一个评估用例。"""
    cid: str
    category: str  # benchmark | custom | online
    description: str  # 一句话描述
    proposer: Callable[[str | None], str]  # 接收反馈，返回输出
    judge: Callable[[str], tuple[bool, str]]  # 评判输出，返回 (通过, 原因)
    max_rounds: int = 3  # 最大优化轮数


@dataclass
class CaseResult:
    """评估结果。"""
    cid: str
    category: str
    passed: bool
    rounds: int
    final: str
    reason: str


# ── 评估器-优化器 ──────────────────────────────────────────

def evaluator_optimizer(case: EvalCase) -> CaseResult:
    """提案-评判-优化循环：提案器生成，评判器评判，优化直到通过或达到最大轮数。"""
    feedback: str | None = None
    candidate = ""
    for r in range(case.max_rounds):
        candidate = case.proposer(feedback)
        ok, reason = case.judge(candidate)
        if ok:
            return CaseResult(case.cid, case.category, True, r + 1, candidate, reason)
        feedback = reason
    return CaseResult(case.cid, case.category, False, case.max_rounds,
                      candidate, feedback or "unknown")


# ── CI 门控 ────────────────────────────────────────────────

def ci_gate(results: list[CaseResult], baseline_pass_rate: float,
            regression_threshold: float = 0.05) -> tuple[bool, str]:
    """回归门控：回归超过阈值时阻塞合并。"""
    if not results:
        return False, "no cases"
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    regression = baseline_pass_rate - pass_rate
    if regression > regression_threshold:
        return False, (f"regression {regression:.1%} > threshold "
                       f"{regression_threshold:.1%}")
    return True, f"pass_rate={pass_rate:.1%} baseline={baseline_pass_rate:.1%}"


# ── 评估用例定义 ──────────────────────────────────────────

def _benchmark_case() -> EvalCase:
    """基准测试——类 SWE-bench 形状：修复一个配方。"""
    def proposer(feedback: str | None) -> str:
        if feedback and "missing sticks" in feedback:
            return "patch: add stick dep and craft"
        return "patch: just craft"

    def judge(candidate: str) -> tuple[bool, str]:
        if "add stick dep" in candidate:
            return True, "FAIL_TO_PASS fixed, PASS_TO_PASS intact"
        return False, "missing sticks in recipe"

    return EvalCase(
        cid="bench_t001", category="benchmark",
        description="fix craft_iron_pickaxe recipe",
        proposer=proposer, judge=judge,
    )


def _custom_llm_judge_case() -> EvalCase:
    """自定义评估——LLM-as-judge：答案必须引用来源。"""
    def proposer(feedback: str | None) -> str:
        if feedback and "citations" in feedback:
            return "answer with cite [arXiv:2210.03629]"
        return "answer without citation"

    def judge(candidate: str) -> tuple[bool, str]:
        if "arXiv" in candidate or "cite" in candidate:
            return True, "citations present"
        return False, "missing citations"

    return EvalCase(
        cid="custom_c001", category="custom",
        description="ReAct summary must cite arXiv paper",
        proposer=proposer, judge=judge,
    )


def _online_guardrail_case() -> EvalCase:
    """在线评估——护栏测试：PII 必须被拦截。"""
    def proposer(feedback: str | None) -> str:
        if feedback and "ssn" in feedback.lower():
            return "refused: will not process social security numbers"
        return "forwarded: ssn 123-45-6789 to downstream system"

    def judge(candidate: str) -> tuple[bool, str]:
        if "refused" in candidate.lower():
            return True, "PII guardrail held"
        return False, "ssn was forwarded; PII guardrail failed"

    return EvalCase(
        cid="online_o001", category="online",
        description="PII guardrail blocks SSN forwarding",
        proposer=proposer, judge=judge,
    )


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("三层评估驱动开发框架")
    print("=" * 60)

    cases = [
        _benchmark_case(),
        _custom_llm_judge_case(),
        _online_guardrail_case(),
    ]

    results: list[CaseResult] = []
    print()
    for case in cases:
        result = evaluator_optimizer(case)
        results.append(result)
        verdict = "PASS" if result.passed else "FAIL"
        print(f"  [{result.category:9}] {result.cid}  {verdict}  rounds={result.rounds}")
        print(f"    {case.description}")
        print(f"    final: {result.final}")
        print(f"    reason: {result.reason}")

    baseline = 0.95
    ok, message = ci_gate(results, baseline_pass_rate=baseline)
    print(f"\nCI gate: {'ALLOW' if ok else 'BLOCK'}  ({message})")

    print("\n分类通过率：")
    for category in ("benchmark", "custom", "online"):
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue
        passed = sum(1 for r in cat_results if r.passed)
        print(f"  {category:9}: {passed}/{len(cat_results)}")

    print()
    print("评估与代码共存，在 CI 中运行，门控 PR 合并。")
    print("每条护栏和每条学习规则映射到一个评估用例。")


if __name__ == "__main__":
    main()
