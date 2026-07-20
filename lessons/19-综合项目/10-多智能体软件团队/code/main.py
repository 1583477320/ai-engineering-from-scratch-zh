"""多智能体软件团队——类型化任务板+交接会计脚手架。

核心架构原语是类型化消息任务板，协调架构师、N个并行编码员、
评审者和测试者，每个角色边界产生追踪span。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class MsgKind(Enum):
    PLAN_REQUEST = "plan_request"
    SUBTASK = "subtask"
    DIFF_READY = "diff_ready"
    REVIEW_NEEDED = "review_needed"
    REVIEW_FEEDBACK = "review_feedback"
    APPROVED = "approved"
    TEST_NEEDED = "test_needed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"


@dataclass
class Msg:
    kind: MsgKind
    by: str
    to: str
    payload: dict = field(default_factory=dict)
    tokens: int = 0


@dataclass
class Board:
    messages: list[Msg] = field(default_factory=list)
    tokens_by_role: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def post(self, m: Msg) -> None:
        self.messages.append(m)
        self.tokens_by_role[m.by] += m.tokens


@dataclass
class Subtask:
    name: str
    files: list[str]
    lines_changed: int = 0
    has_bug: bool = False


def architect_plan(issue: str, rng: random.Random) -> list[Subtask]:
    subs = [Subtask("parser", ["src/parser.py"]), Subtask("cache", ["src/cache.py"]),
            Subtask("api", ["src/api.py"]), Subtask("migration", ["src/migrate.py"])]
    subs[rng.randrange(len(subs))].has_bug = rng.random() < 0.3
    return subs


def coder_implement(sub: Subtask, rng: random.Random) -> dict:
    sub.lines_changed = rng.randint(15, 95)
    return {"subtask": sub.name, "lines": sub.lines_changed, "has_bug": sub.has_bug}


def reviewer_check(diffs: list[dict], rng: random.Random) -> tuple[bool, str]:
    buggy = [d for d in diffs if d["has_bug"]]
    if not buggy:
        return True, "lgtm"
    if rng.random() < 0.85:
        return False, f"在 {buggy[0]['subtask']} 中发现bug"
    return True, "lgtm (FALSE-APPROVE)"


def tester_run(diffs: list[dict], rng: random.Random) -> tuple[bool, str]:
    buggy = [d for d in diffs if d["has_bug"]]
    if buggy:
        return False, f"测试在 {buggy[0]['subtask']} 中失败"
    if rng.random() < 0.03:
        return False, "flaky test"
    return True, "412/412 通过"


def run_team(issue: str, n_coders: int = 4, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(0)
    board = Board()
    plan = architect_plan(issue, rng)
    board.post(Msg(MsgKind.PLAN_REQUEST, by="architect", to="board",
                   payload={"subtasks": [s.name for s in plan]}, tokens=4500))
    for i, sub in enumerate(plan[:n_coders]):
        coder = f"coder-{chr(65 + i)}"
        board.post(Msg(MsgKind.SUBTASK, by="architect", to=coder,
                       payload={"subtask": sub.name, "files": sub.files}, tokens=1200))
    diffs = []
    for i, sub in enumerate(plan[:n_coders]):
        result = coder_implement(sub, rng)
        diffs.append(result)
        board.post(Msg(MsgKind.DIFF_READY, by=f"coder-{chr(65+i)}", to="merge_coord",
                       payload=result, tokens=3200 + result["lines"] * 30))
    board.post(Msg(MsgKind.REVIEW_NEEDED, by="merge_coord", to="reviewer", payload={"diffs": diffs}, tokens=2000))
    approved, comment = reviewer_check(diffs, rng)
    if approved:
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester", payload={"comment": comment}, tokens=1800))
    else:
        board.post(Msg(MsgKind.REVIEW_FEEDBACK, by="reviewer", to="coder-A", payload={"comment": comment}, tokens=1800))
        board.post(Msg(MsgKind.DIFF_READY, by="coder-A", to="merge_coord", payload={"subtask": "parser", "lines": 52, "has_bug": False}, tokens=3100))
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester", payload={"comment": "now lgtm"}, tokens=1500))
        diffs = [{"subtask": d["subtask"], "lines": d["lines"], "has_bug": False} for d in diffs]
    passed, testmsg = tester_run(diffs, rng)
    board.post(Msg(MsgKind.TEST_PASSED if passed else MsgKind.TEST_FAILED, by="tester", to="pr_opener", payload={"msg": testmsg}, tokens=1200))
    return {"approved": approved, "tested_passed": passed, "total_tokens": sum(board.tokens_by_role.values()),
            "tokens_by_role": dict(board.tokens_by_role), "handoffs": sum(1 for m in board.messages if m.to != m.by)}


def main() -> None:
    rng = random.Random(11)
    r = run_team("fix widget parser race", n_coders=4, rng=rng)
    print(f"批准: {r['approved']} 测试通过: {r['tested_passed']} 交接: {r['handoffs']}")
    print(f"总Token: {r['total_tokens']:,}")
    for role, n in sorted(r['tokens_by_role'].items(), key=lambda x: -x[1]):
        print(f"  {role:14s} {n:>6,}")
    rng2 = random.Random(17)
    team_pass = sum(1 for i in range(10) if run_team(f"issue-{i}", n_coders=4, rng=rng2)['tested_passed'])
    print(f"\n10轮对比: 团队通过 {team_pass}/10")


if __name__ == "__main__":
    main()
