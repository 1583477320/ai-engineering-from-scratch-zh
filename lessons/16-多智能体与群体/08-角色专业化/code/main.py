"""角色专业化：规划者、执行者、批评者、验证者。

构建一个简单的 Python 函数。批评者（LLM 模拟）和验证者（代码）一起
捕获任何单独都可能漏过的 bug。

运行两次：一次执行者输出正确，一次有缺陷。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Spec:
    task_name: str
    signature: str
    description: str
    tests: list[tuple[tuple, int]]

@dataclass
class Artifact:
    code: str

@dataclass
class CriticReport:
    approved: bool
    notes: list[str] = field(default_factory=list)

@dataclass
class VerifierReport:
    passed: bool
    failures: list[str] = field(default_factory=list)


def planner(user_wish):
    """规划者：将愿望转为结构化规格。"""
    return Spec(
        task_name="add_two",
        signature="add_two(a: int, b: int) -> int",
        description=user_wish,
        tests=[((1, 2), 3), ((10, 20), 30), ((-5, 5), 0)],
    )


def executor_correct(spec):
    return Artifact(code="def add_two(a, b):\n    return a + b\n")


def executor_buggy(spec):
    return Artifact(code="def add_two(a, b):\n    return a * b\n")


def critic(spec, art):
    """LLM 批评者——可以被似是而非的代码愚弄。"""
    notes = []
    if "def" not in art.code:
        notes.append("缺少 def 语句")
    if "return" not in art.code:
        notes.append("缺少 return")
    if spec.task_name not in art.code:
        notes.append(f"函数名不匹配 '{spec.task_name}'")
    return CriticReport(approved=not notes, notes=notes)


def verifier(spec, art):
    """确定性验证者——在沙箱中执行代码并运行测试。"""
    ns = {}
    try:
        exec(art.code, ns, ns)
    except Exception as e:
        return VerifierReport(passed=False, failures=[f"执行错误: {e}"])
    fn = ns.get(spec.task_name)
    if not callable(fn):
        return VerifierReport(passed=False, failures=[f"未产生可调用的 '{spec.task_name}'"])
    failures = []
    for args, expected in spec.tests:
        try:
            got = fn(*args)
        except Exception as e:
            failures.append(f"调用 {args} 抛出 {e}")
            continue
        if got != expected:
            failures.append(f"调用 {args}: 期望 {expected}, 得到 {got}")
    return VerifierReport(passed=not failures, failures=failures)


def run_pipeline(user_wish, executor, label):
    print(f"\n=== {label} ===")
    spec = planner(user_wish)
    print(f"  [规划者] 规格: {spec.signature}, {len(spec.tests)} 个测试")
    art = executor(spec)
    print(f"  [执行者] 产出:\n    {art.code.replace(chr(10), chr(10)+'    ')}")
    crep = critic(spec, art)
    print(f"  [批评者] approved={crep.approved}, notes={crep.notes}")
    vrep = verifier(spec, art)
    print(f"  [验证者] passed={vrep.passed}, failures={vrep.failures}")
    if crep.approved and vrep.passed:
        print("  结论: 交付。")
    elif not vrep.passed:
        print("  结论: 验证者阻止交付（确定性捕获）。")
    elif not crep.approved:
        print("  结论: 批评者阻止交付（主观捕获）。")


def main():
    print("角色专业化流水线——规划者、执行者、批评者、验证者")
    print("-" * 70)

    run_pipeline("返回两个整数之和的函数。", executor_correct, "正确执行输出")

    run_pipeline("返回两个整数之和的函数。", executor_buggy, "有缺陷执行输出（看起来合理；运行时失败）")

    print("\n关键洞察: 批评者放过有缺陷代码因为它看起来没问题。")
    print("只有验证者——确定性测试执行——捕获了语义 bug。")
    print("全 LLM 管道（无验证者）会交付 bug。经典 MAST 失败模式。")


if __name__ == "__main__":
    main()
