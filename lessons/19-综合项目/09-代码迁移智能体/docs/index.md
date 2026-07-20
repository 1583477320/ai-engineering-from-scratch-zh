# 综合项目09——代码迁移智能体（仓库级语言/运行时升级）

> Amazon的MigrationBench（Java 8到17）和Google的App Engine Py2到Py3迁移工具设定了2026年的标准。Moderne的OpenRewrite以规模执行确定性AST重写。Grit使用codemod风格的DSL解决相同问题。生产模式结合两者：确定性基础层处理安全重写，智能体层处理模糊情况，沙箱处理每分支构建，测试工具在PR打开前确保绿灯。本综合项目要求你迁移50个真实仓库并发布带失败分类的通过率。

**类型：** 综合项目
**编程语言：** Python（智能体），Java/Python（目标语言），TypeScript（仪表盘）
**前置知识：** 第5章（NLP）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）
**涉及章节：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建两层代码迁移管道：确定性配方层+智能体循环层
- 实现沙箱化的构建和测试验证流程
- 实现失败分类法：将每个失败仓库标记到分类桶
- 在50仓库基准上评测通过率、成本、覆盖率保持

---

## 1. 问题

大规模代码迁移是2026年最清晰的编码智能体生产应用之一。真实结果显而易见（迁移后测试套件是否通过？），收益真实（Java 8机群迁移是人月级项目），基准公开（MigrationBench 50仓库子集）。

Moderne的OpenRewrite处理确定性侧。智能体层处理OpenRewrite配方无法处理的一切：模糊重写、构建系统漂移、长尾语法、传递依赖断裂。

---

## 2. 核心概念

### 2.1 两层结构

**确定性基础层**（Java的OpenRewrite、Python的libcst）安全地运行大量机械重写：导入、方法签名、空安全检查、try-with-resources、废弃API替换。快速且可审计。

**智能体层**（OpenAI Agents SDK或LangGraph over Claude Opus 4.7和GPT-5.4-Codex）处理配方无法处理的场景：构建文件升级、传递依赖冲突、测试飘忽、自定义注解。

### 2.2 失败分类法

50个仓库中哪些坏了？传递依赖？自定义注解？构建工具版本？与迁移无关的测试飘忽？每个类别获得计数和示例差异。

---

## 3. 从零实现

`code/main.py`实现两层迁移管道：确定性配方通行+智能体循环，含硬预算和失败分类。

```python
"""代码迁移智能体——确定性配方+智能体循环回退脚手架。

核心架构原语是两层结构：先确定性配方通行（快速、可审计、安全），
然后智能体循环处理剩余失败，含硬预算和失败分类步骤。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 仓库+失败分类法
# ---------------------------------------------------------------------------

FAILURE_CLASSES = [
    "dep_upgrade_required",
    "build_tool_drift",
    "custom_annotation",
    "test_flake",
    "syntax_edge_case",
    "budget_exhausted",
    "coverage_regression",
]


@dataclass
class Repo:
    name: str
    loc: int
    lang: str          # "java" | "python"
    hardness: float    # 0..1


@dataclass
class Attempt:
    repo: Repo
    recipe_applied: int = 0
    agent_turns: int = 0
    cost_usd: float = 0.0
    wall_min: float = 0.0
    status: str = "pending"  # "pass" | "fail"
    failure_class: str | None = None
    coverage_base: float = 80.0
    coverage_final: float = 80.0


# ---------------------------------------------------------------------------
# 确定性配方通行——替代OpenRewrite/libcst
# ---------------------------------------------------------------------------

def run_recipes(repo: Repo) -> int:
    """返回应用的重写次数"""
    base = 20 + int(repo.loc / 500)
    return int(base * (1 - 0.2 * repo.hardness))


# ---------------------------------------------------------------------------
# 智能体循环——分类失败，应用修复，重试；预算感知
# ---------------------------------------------------------------------------

BUDGET_MIN = 30.0
BUDGET_USD = 8.0
BUDGET_TURNS = 20


def agent_loop(attempt: Attempt, rng: random.Random) -> None:
    """模拟计划-行动循环直到通过或预算耗尽"""
    per_turn_min = 2.8 + attempt.repo.hardness * 2.0
    per_turn_usd = 0.45 + attempt.repo.hardness * 0.65
    turn_pass_p = max(0.02, 0.22 * (1 - attempt.repo.hardness * 0.95))

    while True:
        if attempt.agent_turns >= BUDGET_TURNS:
            attempt.status = "fail"
            attempt.failure_class = "budget_exhausted"
            return
        if attempt.wall_min >= BUDGET_MIN or attempt.cost_usd >= BUDGET_USD:
            attempt.status = "fail"
            attempt.failure_class = "budget_exhausted"
            return

        attempt.agent_turns += 1
        attempt.wall_min += per_turn_min
        attempt.cost_usd += per_turn_usd

        if rng.random() < turn_pass_p:
            delta = rng.gauss(0.0, 0.6)
            attempt.coverage_final = attempt.coverage_base + delta
            if attempt.coverage_final < attempt.coverage_base - 2.0:
                attempt.status = "fail"
                attempt.failure_class = "coverage_regression"
                return
            attempt.status = "pass"
            return


# ---------------------------------------------------------------------------
# 失败分类——将卡住的仓库桶进分类法
# ---------------------------------------------------------------------------

def classify_failure(rng: random.Random) -> str:
    """替代智能体的失败分类器。真实实现读取构建日志和测试输出。"""
    weights = {
        "dep_upgrade_required": 0.30,
        "build_tool_drift": 0.20,
        "custom_annotation": 0.18,
        "test_flake": 0.15,
        "syntax_edge_case": 0.17,
    }
    r = rng.random()
    acc = 0.0
    for cls, w in weights.items():
        acc += w
        if r <= acc:
            return cls
    return "syntax_edge_case"


# ---------------------------------------------------------------------------
# 管道——配方通行然后智能体然后PR/归档
# ---------------------------------------------------------------------------

def migrate(repo: Repo, rng: random.Random) -> Attempt:
    attempt = Attempt(repo=repo)
    attempt.recipe_applied = run_recipes(repo)

    straight_through_p = 0.55 * (1 - repo.hardness)
    if rng.random() < straight_through_p:
        delta = rng.gauss(0.0, 0.4)
        attempt.coverage_final = attempt.coverage_base + delta
        attempt.status = "pass"
        attempt.wall_min = 3.0 + rng.random() * 4
        attempt.cost_usd = 0.30
        return attempt

    agent_loop(attempt, rng)

    if attempt.status == "fail" and attempt.failure_class == "budget_exhausted":
        if rng.random() < 0.75:
            attempt.failure_class = classify_failure(rng)
    return attempt


# ---------------------------------------------------------------------------
# 50仓库模拟
# ---------------------------------------------------------------------------

def synth_bench(rng: random.Random) -> list[Repo]:
    bench: list[Repo] = []
    for i in range(50):
        lang = "java" if rng.random() < 0.6 else "python"
        hardness = min(0.95, max(0.05, rng.gauss(0.65, 0.18)))
        bench.append(Repo(name=f"repo-{i:02d}-{lang}",
                          loc=rng.randint(800, 40_000),
                          lang=lang,
                          hardness=hardness))
    return bench


def main() -> None:
    rng = random.Random(19)
    bench = synth_bench(rng)

    results: list[Attempt] = []
    for repo in bench:
        results.append(migrate(repo, rng))

    passed = [a for a in results if a.status == "pass"]
    failed = [a for a in results if a.status == "fail"]

    print(f"=== 迁移基准运行（50个仓库）===")
    print(f"通过 : {len(passed):2d}  ({len(passed) / 50:.1%})")
    print(f"失败 : {len(failed):2d}")

    print("\n失败分类法:")
    taxonomy: dict[str, int] = {}
    for a in failed:
        taxonomy[a.failure_class or "unknown"] = taxonomy.get(a.failure_class or "unknown", 0) + 1
    for cls, n in sorted(taxonomy.items(), key=lambda x: -x[1]):
        print(f"  {cls:24s} {n}")

    if passed:
        mean_cost = sum(a.cost_usd for a in passed) / len(passed)
        mean_min = sum(a.wall_min for a in passed) / len(passed)
        mean_turns = sum(a.agent_turns for a in passed) / len(passed)
        mean_cov_delta = sum(a.coverage_final - a.coverage_base for a in passed) / len(passed)
        print("\n通过仓库指标:")
        print(f"  平均$/仓库     : ${mean_cost:.2f}")
        print(f"  平均耗时(分)   : {mean_min:.1f}")
        print(f"  平均智能体轮次: {mean_turns:.1f}")
        print(f"  平均覆盖率变化: {mean_cov_delta:+.2f} 百分点")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 迁移基准运行（50个仓库）===
通过 : 30  (60.0%)
失败 : 20

失败分类法:
  dep_upgrade_required      6
  build_tool_drift          4
  custom_annotation         4
  syntax_edge_case          3
  test_flake                2
  budget_exhausted          1

通过仓库指标:
  平均$/仓库     : $1.52
  平均耗时(分)   : 5.8
  平均智能体轮次: 1.2
  平均覆盖率变化: -0.01 百分点
```

---

## 4. 工具实践

**技术栈：**
- 确定性基础：OpenRewrite（Java）或libcst（Python）
- 智能体：OpenAI Agents SDK或LangGraph + Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：Daytona devcontainer每分支，预装目标运行时
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50仓库子集（Java 8到17）
- 仪表盘：失败分类法仪表盘

---

## 5. LLM视角

**两层视角**：确定性配方覆盖70-80%的迁移工作。智能体覆盖剩余的20-30%，这些是真正的挑战——构建系统漂移、传递依赖、自定义注解。

**预算约束视角**：30分钟/$8/20轮的硬限制让智能体有选择地工作，而非无休止地尝试。这也是生产系统的关键要求。

**失败分类视角**：跨50个仓库归类失败原因，让配方作者可以针对top 3贡献。

---

## 6. 工程最佳实践

**管道设计**：
- 确定性配方先通行（70-80%的修复）
- 智能体循环处理剩余失败
- 测试+覆盖率门控

**预算控制**：
- 每仓库30分钟墙钟
- 每仓库$8成本上限
- 20轮智能体迭代上限

**失败分类法**：
- dep_upgrade_required
- build_tool_drift
- custom_annotation
- test_flake
- syntax_edge_case
- budget_exhausted
- coverage_regression

---

## 7. 常见错误

**错误1：跳过确定性配方层**
症状：智能体浪费token在机械重写上
修复：OpenRewrite/libcst配方先通行

**错误2：无预算限制**
症状：智能体在一个仓库上花$50
修复：硬预算上限

**错误3：不分类失败**
症状：不知道什么坏了，无法改进
修复：失败分类法仪表盘

---

## 8. 面试考点

**Q1：代码迁移的两层结构是什么？**
考察：对架构设计的理解

**Q2：为什么预算限制对迁移智能体很重要？**
考察：对成本控制的理解

**Q3：失败分类法如何指导改进？**
考察：对数据驱动改进的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 确定性基础层 | "配方引擎" | OpenRewrite/libcst：声明性AST重写，带安全保证 |
| Codemod | "代码修改程序" | 机械更改源代码的重写规则 |
| 构建漂移 | "工具版本偏差" | Maven/Gradle/uv主要版本间的微妙行为变化 |
| 失败类 | "分类桶" | 仓库未迁移的标记原因 |
| 覆盖率变化 | "覆盖率保持" | 从基分支到迁移分支的测试覆盖率%变化 |
| 智能体轮次 | "工具调用回合" | 智能体循环中的一个计划->行动->观察周期 |
| 预算耗尽 | "达到上限" | 仓库消耗完30分钟/$8/20轮限制而未通过 |

---

## 参考文献

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/)
- [Moderne.io OpenRewrite平台](https://www.moderne.io)
- [OpenRewrite文档](https://docs.openrewrite.org)
- [Grit.io](https://www.grit.io)
- [OpenAI沙箱迁移cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent)
- [Google App Engine Py2到Py3迁移工具](https://cloud.google.com/appengine)
- [libcst](https://github.com/Instagram/LibCST)
- [Daytona沙箱](https://daytona.io)
