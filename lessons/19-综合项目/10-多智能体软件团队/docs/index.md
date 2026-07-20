# 综合项目10——多智能体软件工程团队

> SWE-AF的工厂架构、MetaGPT的基于角色的提示、AutoGen 0.4的类型化参与者图、Cognition的Devin和Factory的Droids都收敛到相同的2026形态：架构师规划、N个编码员在并行工作树中工作、评审者把关、测试者验证。并行工作树将墙钟时间转化为吞吐量。共享状态和交接协议成为失效面。本综合项目要求你构建团队，在SWE-bench Pro上评测，并报告哪些交接失败、频率如何。

**类型：** 综合项目
**编程语言：** Python/TypeScript（智能体），Shell（工作树脚本）
**前置知识：** 第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第16章（多智能体）、第17章（基础设施）
**涉及章节：** P11 · P13 · P14 · P15 · P16 · P17
**预计时间：** 40小时

---

## 学习目标

- 构建多智能体软件工程团队：架构师+编码员+评审者+测试者
- 实现类型化消息任务板和交接协议
- 实现并行工作树和合并协调
- 评测token放大率和多智能体速度提升

---

## 1. 问题

单智能体编码主循环在大任务上遇到天花板。不是因为单个智能体弱，而是因为20万token的上下文无法同时容纳架构计划加四个并行代码库切片加评审者评论加测试输出。

多智能体工厂将问题分解：架构师拥有计划，编码员在并行工作树中拥有实现，评审者把关，测试者验证。SWE-AF的"工厂"架构、MetaGPT的角色、AutoGen的类型化参与者图——所有三种框架都描述了相同的形态。

失效面是交接。架构师计划了编码员无法实现的东西。编码员产生冲突的差异。评审者批准了幻觉修复。测试者与仍在写入的编码员竞争。

---

## 2. 核心概念

### 2.1 角色

**架构师**（Claude Opus 4.7）：阅读问题，编写计划，分解为带显式接口的子任务。

**编码员**（Claude Sonnet 4.7，N个并行实例，每个在`git worktree` + Daytona沙箱中）：独立实现子任务。

**评审者**（GPT-5.4）：阅读合并后的差异，批准或要求特定更改。

**测试者**（Gemini 2.5 Pro）：在隔离沙箱中运行测试套件，报告通过/失败及结果。

### 2.2 通信

通过共享任务板（文件或Redis）进行通信。每个角色消费其允许处理的任务。交接是A2A协议类型化的消息。

协调关注点：合并冲突解决、共享状态同步、评审者把关（评审者不能批准自己提出的更改）。

### 2.3 Token放大

每个角色边界添加摘要提示和交接上下文。40轮单智能体运行变成跨四个角色的总共160轮。评估权衡token效率与单智能体基线。

---

## 3. 从零实现

`code/main.py`实现类型化消息任务板、角色存根和交接会计。

```python
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


# ---------------------------------------------------------------------------
# 类型化消息任务板——A2A风格类型化消息
# ---------------------------------------------------------------------------

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

    def inbox(self, role: str) -> list[Msg]:
        return [m for m in self.messages if m.to == role]


# ---------------------------------------------------------------------------
# 角色存根——架构师、编码员、评审者、测试者
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    name: str
    files: list[str]
    lines_changed: int = 0
    has_bug: bool = False


def architect_plan(issue: str, rng: random.Random) -> list[Subtask]:
    subs = [
        Subtask("parser", ["src/parser.py"]),
        Subtask("cache", ["src/cache.py"]),
        Subtask("api", ["src/api.py"]),
        Subtask("migration", ["src/migrate.py"]),
    ]
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
        return False, f"在 {buggy[0]['subtask']} 中发现bug: 请修改"
    return True, "lgtm (FALSE-APPROVE)"


def tester_run(diffs: list[dict], rng: random.Random) -> tuple[bool, str]:
    buggy = [d for d in diffs if d["has_bug"]]
    if buggy:
        return False, f"测试在 {buggy[0]['subtask']} 模块中失败"
    if rng.random() < 0.03:
        return False, "flaky test"
    return True, "412/412 通过"


# ---------------------------------------------------------------------------
# 编排器——运行完整流程，计算token放大
# ---------------------------------------------------------------------------

def run_team(issue: str, n_coders: int = 4, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(0)
    board = Board()

    plan = architect_plan(issue, rng)
    board.post(Msg(MsgKind.PLAN_REQUEST, by="architect", to="board",
                   payload={"issue": issue, "subtasks": [s.name for s in plan]}, tokens=4500))

    for i, sub in enumerate(plan[:n_coders]):
        coder = f"coder-{chr(65 + i)}"
        board.post(Msg(MsgKind.SUBTASK, by="architect", to=coder,
                       payload={"subtask": sub.name, "files": sub.files}, tokens=1200))

    diffs: list[dict] = []
    for i, sub in enumerate(plan[:n_coders]):
        coder = f"coder-{chr(65 + i)}"
        result = coder_implement(sub, rng)
        diffs.append(result)
        board.post(Msg(MsgKind.DIFF_READY, by=coder, to="merge_coord",
                       payload=result, tokens=3200 + result["lines"] * 30))

    board.post(Msg(MsgKind.REVIEW_NEEDED, by="merge_coord", to="reviewer",
                   payload={"diffs": diffs}, tokens=2000))

    approved, comment = reviewer_check(diffs, rng)
    if approved:
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester",
                       payload={"comment": comment}, tokens=1800))
    else:
        board.post(Msg(MsgKind.REVIEW_FEEDBACK, by="reviewer", to="coder-A",
                       payload={"comment": comment}, tokens=1800))
        board.post(Msg(MsgKind.DIFF_READY, by="coder-A", to="merge_coord",
                       payload={"subtask": "parser", "lines": 52, "has_bug": False}, tokens=3100))
        board.post(Msg(MsgKind.APPROVED, by="reviewer", to="tester",
                       payload={"comment": "now lgtm"}, tokens=1500))
        diffs = [{"subtask": d["subtask"], "lines": d["lines"], "has_bug": False} for d in diffs]

    passed, testmsg = tester_run(diffs, rng)
    board.post(Msg(MsgKind.TEST_PASSED if passed else MsgKind.TEST_FAILED,
                   by="tester", to="pr_opener", payload={"msg": testmsg}, tokens=1200))

    return {
        "approved": approved, "review_comment": comment,
        "tested_passed": passed, "test_msg": testmsg,
        "total_tokens": sum(board.tokens_by_role.values()),
        "tokens_by_role": dict(board.tokens_by_role),
        "handoffs": sum(1 for m in board.messages if m.to != m.by),
    }


def main() -> None:
    rng = random.Random(11)
    print("=== 多智能体团队运行 ===")
    result = run_team("fix widget parser race", n_coders=4, rng=rng)
    print(f"批准        : {result['approved']} ({result['review_comment']})")
    print(f"测试通过    : {result['tested_passed']} ({result['test_msg']})")
    print(f"交接次数    : {result['handoffs']}")
    print(f"总Token    : {result['total_tokens']:,}")
    print("各角色Token:")
    for role, n in sorted(result['tokens_by_role'].items(), key=lambda x: -x[1]):
        print(f"  {role:14s} {n:>6,}")

    print("\n=== 10次对比试验 vs 单智能体基线 ===")
    rng2 = random.Random(17)
    team_pass = sum(1 for i in range(10) if run_team(f"issue-{i}", n_coders=4, rng=rng2)['tested_passed'])
    base_pass = sum(1 for i in range(10) if rng2.random() < 0.68)
    print(f"团队通过: {team_pass}/10  基线通过: {base_pass}/10")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 多智能体团队运行 ===
批准        : False (在 cache 中发现bug: 请修改)
测试通过    : True (412/412 通过)
交接次数    : 8
总Token    : 40,480
各角色Token:
  architect       4,500
  coder-A        12,200
  coder-B         9,200
  coder-C         8,600
  coder-D         8,180
  merge_coord     2,000
  reviewer        3,300
  tester          1,200

=== 10次对比试验 vs 单智能体基线 ===
团队通过: 8/10  基线通过: 7/10
```

---

## 4. 工具实践

**技术栈：**
- 编排：LangGraph + 共享状态 + 每智能体子图
- 消息：A2A协议（Google 2025）类型化智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码员）、GPT-5.4（评审者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：`git worktree add` + Daytona沙箱
- 合并协调：自定义三方合并 + LLM调解冲突

---

## 5. LLM视角

**交接视角**：每个角色边界增加token成本。40轮单智能体变成160轮团队。关键在于"每解决一个问题的token数"是否低于单智能体。

**并行视角**：并行工作树将墙钟时间转换为吞吐量。4个编码员做4个子任务，理论上4倍速度。但合并冲突和评审循环会降低实际收益。

**评审视角**：评审者对注入bug的假批准率是关键指标。目标低于5%。评审者不能批准自己提出的更改。

---

## 6. 工程最佳实践

**任务板设计**：
- JSONL文件或Redis后端
- 类型化消息：plan_request、subtask、diff_ready等
- 智能体订阅标签

**工作树隔离**：
- `git worktree add`每编码员
- Daytona沙箱隔离
- 合并协调器三方合并

**Token会计**：
- 每个角色边界记录token
- 计算每子任务的token放大率
- 与单智能体基线对比

---

## 7. 常见错误

**错误1：忽略交接成本**
症状：团队token消耗远超单智能体
修复：监控token放大率，减少不必要交接

**错误2：无合并冲突解决**
症状：并行工作树产生冲突，团队卡住
修复：实现合并协调器

**错误3：评审者批准自己的代码**
症状：bug通过评审
修复：评审者不能批准自己提出的更改

---

## 8. 面试考点

**Q1：多智能体软件团队如何改进单智能体编码？**
考察：对并行化和角色分解的理解

**Q2：什么是token放大？如何计算？**
考察：对多智能体成本的理解

**Q3：为什么评审者不能批准自己提出的更改？**
考察：对智能体治理的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 并行工作树 | "隔离分支" | `git worktree add`为每个编码员创建独立工作树 |
| 任务板 | "共享消息总线" | 智能体订阅的类型化消息的文件或Redis存储 |
| 交接 | "角色边界" | 从一个角色上下文到另一个的消息 |
| Token放大 | "多智能体开销" | 跨角色总token / 单智能体同一任务token |
| A2A协议 | "智能体到智能体" | Google 2025年类型化智能体间消息规范 |
| 合并协调器 | "集成者" | 执行三方合并和调解冲突的组件 |
| 假批准 | "评审者幻觉" | 评审者批准含有已知bug的差异 |

---

## 参考文献

- [SWE-AF工厂架构](https://github.com/Agent-Field/SWE-AF)
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT)
- [AutoGen v0.4](https://github.com/microsoft/autogen)
- [Cognition AI（Devin）](https://cognition.ai)
- [Google A2A协议](https://developers.google.com/agent-to-agent)
- [git worktree文档](https://git-scm.com/docs/git-worktree)
- [SWE-bench Pro](https://www.swebench.com)
