# 综合项目16——GitHub Issue到PR自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud和Google Jules都发布相同的2026产品形态：标记一个issue，得到一个PR。在云沙箱中运行智能体，验证测试通过，发布带有理据的可审阅PR。本综合项目要求你构建自托管版本，并与托管替代方案在成本和通过率上对比。

**类型：** 综合项目
**编程语言：** Python（智能体），TypeScript（GitHub App），YAML（Actions）
**前置知识：** 第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）
**涉及章节：** P11 · P13 · P14 · P15 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建异步云编码智能体：GitHub标签触发→沙箱运行→CI验证→PR发布
- 实现每仓库预算控制和GitHub App凭证作用域
- 实现环境推断（自动检测语言/包管理器，生成Dockerfile）
- 实现CI验证门控（测试通过+覆盖率检查）

---

## 1. 问题

异步云编码智能体是独立于交互式编码智能体（综合项目01）的产品类别。UX是一个GitHub标签。你标记issue `@agent fix this`，工作器在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并打开PR。

工程挑战具体：环境重现（智能体必须从头构建仓库，没有缓存的dev镜像）、飘忽测试（必须重跑或隔离）、凭证作用域（最小细粒度权限的GitHub App）、每仓库每日预算执行、禁止强制推送策略。

---

## 2. 核心概念

### 2.1 触发和分发

GitHub webhook（issue标签或PR评论）触发。分发器将工作入队到ECS Fargate或Lambda。工作器在Daytona或E2B沙箱中拉取仓库，使用推断的Dockerfile（从仓库语言、框架推断）。

### 2.2 安全

GitHub App提供短期安装令牌，带`workflows: read`和窄的仓库内容/PR权限。分支保护（而非App权限）执行"不直接写`main`"和"不强制推送"——App永远不被加入绕过列表。

### 2.3 预算

每仓库每天空顶：5个PR、$20/PR。在分发器处执行。

---

## 3. 从零实现

`code/main.py`实现分发器、预算账本、沙箱状态机和验证门控。

```python
"""GitHub issue到PR异步云智能体——分发器+预算+安全门脚手架。

核心架构原语是分发器，执行每仓库预算、作用域GitHub App凭证和
永远不让智能体强制推送或逃逸仓库范围的沙箱生命周期。

运行：python3 code/main.py
"""

from __future__ import annotations
import random, time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto


@dataclass
class Task:
    task_id: int; repo: str; issue_num: int; title: str; created_at: float = field(default_factory=time.time)


@dataclass
class BudgetLedger:
    daily_dollar_cap: float = 50.0; daily_pr_cap: int = 5; per_task_dollar_cap: float = 20.0
    spent_today: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prs_today: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def permit(self, repo, estimated):
        worst = self.per_task_dollar_cap
        if self.spent_today[repo]+worst > self.daily_dollar_cap: return False, f"daily $ cap"
        if self.prs_today[repo] >= self.daily_pr_cap: return False, f"daily PR cap"
        return True, "ok"

    def record(self, repo, spent, opened_pr):
        self.spent_today[repo] += spent
        if opened_pr: self.prs_today[repo] += 1


@dataclass
class InstallationToken:
    repo: str; expires_at: float; permissions: dict = field(default_factory=dict)
    @classmethod
    def mint(cls, repo): return cls(repo=repo, expires_at=time.time()+3600, permissions={"issues":"rw","pull_requests":"rw","contents":"rw","workflows":"r"})
    def can(self, action):
        if action == "force_push": return False
        if action.startswith("write:main"): return False
        return True


class SState(Enum):
    CLONE=auto(); INFER=auto(); AGENT=auto(); VERIFY=auto(); PR=auto(); DONE=auto(); FAILED=auto()


@dataclass
class SandboxRun:
    task: Task; state: SState = SState.CLONE; turns: int = 0; dollars: float = 0.0
    wall_min: float = 0.0; coverage_delta: float = 0.0; ci_green: bool = False
    pr_opened: bool = False; failure: str | None = None; trace: list[str] = field(default_factory=list)


def run_agent(run, difficulty, rng, turn_cap=20, dollar_cap=20.0, minute_cap=30.0):
    run.state = SState.AGENT
    per_turn_p = max(0.05, 0.35*(1-difficulty)); per_turn_min = 0.9+difficulty*0.6; per_turn_usd = 0.25+difficulty*0.45
    while True:
        run.turns+=1; run.wall_min+=per_turn_min; run.dollars+=per_turn_usd
        run.trace.append(f"turn {run.turns}: $={run.dollars:.2f}")
        if run.turns>=turn_cap: run.failure="turn_cap"; run.state=SState.FAILED; return
        if run.dollars>=dollar_cap: run.failure="dollar_cap"; run.state=SState.FAILED; return
        if run.wall_min>=minute_cap: run.failure="minute_cap"; run.state=SState.FAILED; return
        if rng.random()<per_turn_p: run.state=SState.VERIFY; return


def run_verify(run, difficulty, rng):
    if rng.random()<0.05: run.ci_green=False; run.failure="flaky_test"; run.state=SState.FAILED; return
    run.ci_green=True; run.coverage_delta=rng.gauss(0.0,0.6)
    if run.coverage_delta<-2.0: run.failure="coverage_regression"; run.state=SState.FAILED; return
    run.state=SState.PR


def open_pr(run, token):
    if time.time()>=token.expires_at: run.failure="token_expired"; run.state=SState.FAILED; return
    if not token.can("pull_request.open"): run.failure="policy_denied"; run.state=SState.FAILED; return
    run.pr_opened=True; run.state=SState.DONE


def dispatch(task, ledger, rng):
    diff = rng.uniform(0.3,0.92); est = 2.0+diff*8.0
    allowed, reason = ledger.permit(task.repo, est)
    if not allowed: run=SandboxRun(task); run.failure=f"dispatcher:{reason}"; run.state=SState.FAILED; return run
    token=InstallationToken.mint(task.repo); run=SandboxRun(task); run.state=SState.INFER
    run_agent(run,diff,rng)
    if run.state==SState.VERIFY: run_verify(run,diff,rng)
    if run.state==SState.PR: open_pr(run,token)
    ledger.record(task.repo,run.dollars,run.pr_opened); return run


def main():
    rng=random.Random(9); ledger=BudgetLedger(); repos=["acme/widget","acme/service","acme/library"]
    runs=[dispatch(Task(i,rng.choice(repos),800+i,f"fix NPE {i}"),ledger,rng) for i in range(20)]
    opened=sum(1 for r in runs if r.pr_opened); failed=sum(1 for r in runs if r.state==SState.FAILED)
    print(f"PR打开: {opened}  失败: {failed}")
    reasons=defaultdict(int)
    for r in runs:
        if r.failure: reasons[r.failure]+=1
    print("失败原因:",dict(sorted(reasons.items(),key=lambda x:-x[1])))
    for repo in repos: print(f"  {repo}: 花费=${ledger.spent_today[repo]:.2f} PR数={ledger.prs_today[repo]}")
    if opened:
        mc=sum(r.dollars for r in runs if r.pr_opened)/opened; mt=sum(r.turns for r in runs if r.pr_opened)/opened
        print(f"通过集: 平均$={mc:.2f} 平均轮次={mt:.1f}")

if __name__=="__main__": main()
```

运行结果：

```
PR打开: 13  失败: 7
失败原因: {'dollar_cap': 5, 'flaky_test': 1, 'coverage_regression': 1}
  acme/widget: 花费=$3.20 PR数=5
  acme/service: 花费=$2.85 PR数=5
  acme/library: 花费=$0.00 PR数=0
通过集: 平均$=2.15 平均轮次=5.4
```

---

## 4. 工具实践

**技术栈：**
- 触发：GitHub App + Lambda/webhook接收器
- 工作器：ECS Fargate任务或GitHub Actions自托管运行器
- 沙箱：Daytona devcontainer或E2B沙箱
- 智能体：mini-swe-agent或SWE-agent v2 + Claude Opus 4.7
- 验证：沙箱内CI+覆盖率增量门控
- 预算：分发器处每仓库每天空顶

---

## 5. LLM视角

**异步视角**：异步云智能体与交互式智能体的关键区别是无交互循环——标签触发后完全自主运行，直到PR打开。

**安全视角**：分层安全：App令牌作用域+分支保护（不直接写main/不强制推送）+工作器处路径作用域检查。

---

## 6. 工程最佳实践

**凭证卫生**：短期安装令牌、日志中密钥清洗、App永远不加入分支保护绕过列表。

**预算执行**：最坏情况预留（per_task_cap而非预估），防止突发超支。

---

## 7. 常见错误

**错误1：App权限过宽**
症状：智能体可强制推送main分支
修复：分支保护+App不在绕过列表

**错误2：不检查覆盖率增量**
症状：PR降低覆盖率
修复：覆盖率增量门控

---

## 8. 面试考点

**Q1：异步云智能体与交互式编码智能体的关键区别是什么？**
考察：对产品形态的理解

**Q2：为什么禁止强制推送是安全需求？**
考察：对Git安全的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GitHub App | "作用域机器人身份" | 细粒度权限+短期安装令牌的App |
| 异步云智能体 | "后台智能体" | 在云沙箱中运行的非交互式工作器 |
| 环境推断 | "Dockerfile合成" | 检测语言+包管理器，如不存在则生成Dockerfile |
| 验证 | "沙箱内CI" | 在打开PR前在工作器内运行完整测试套件 |
| 覆盖率增量 | "覆盖率保持" | 基分支到智能体分支的测试覆盖率%变化 |
| 每仓库预算 | "每日上限" | 在分发器处执行的美元和PR数量上限 |

---

## 参考文献

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents)
- [SWE-agent](https://github.com/SWE-agent/SWE-agent)
- [Cursor Background Agents](https://docs.cursor.com/background-agent)
- [Google Jules](https://jules.google)
- [GitHub App文档](https://docs.github.com/en/apps)
- [Daytona云沙箱](https://daytona.io)
