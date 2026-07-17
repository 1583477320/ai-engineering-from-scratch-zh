# 评估驱动的智能体开发

> Anthropic 的指导："从简单的提示词开始，用全面的评估优化它们，只在需要时才添加多步智能体系统。" 评估不是最后一步。它是驱动第 14 章中每个其他选择的外循环。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 第 14 章全部
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 05（Self-Refine）— 评估器-优化器循环是 Self-Refine 的泛化

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名三个评估层——静态基准、自定义离线、在线生产——并解释各自的用途
- [ ] 实现评估器-优化器紧循环：提案 → 评判 → 优化，直到通过
- [ ] 描述 2026 年最佳实践：评估与代码共存、在 CI 中运行、门控 PR 合并
- [ ] 将第 14 章每节课映射到它生成的评估用例上

---

## 1. 问题

智能体通过演示。它在生产中以演示无法预测的方式失败。

基准测试回答"这个模型在广泛场景下是否有能力？"，而不是"这个智能体为我的产品生成了正确的补丁吗？"。一个在 SWE-bench 上得分 80% 的模型，在你的代码库上可能完全跑偏——因为你的代码库不在训练数据中，你的验收标准不是基准测试的问题格式。

问题的本质是：**演示是选择性的，生产是全量的。** 智能体在演示中选择最顺利的路径走，在生产中遇到所有边缘情况。

解决方案是三层评估——静态基准（跨模型对比）、自定义离线（产品特定测试）、在线生产（真实流量监控）——持续运行，每条护栏和每条学习规则都映射到一个评估用例。

---

## 2. 概念

### 2.1 三个评估层

```
静态基准（秒级）→ 通过？→ 自定义离线（分钟级）→ 通过？→ 部署 → 在线监控（实时）
       ↓失败            ↓失败                    ↓异常
   修改提示词        调试智能体               回滚/修复
```

| 层级 | 速度 | 成本 | 用途 | 工具 |
|------|------|------|------|------|
| **静态基准** | 秒 | 低 | 跨模型对比、回归门控 | SWE-bench Verified、GAIA、BFCL V4 |
| **自定义离线** | 分钟 | 中 | 产品功能验证 | LLM-as-judge、执行验证、轨迹对比 |
| **在线生产** | 实时 | 高 | 质量监控 | Langfuse 回放、护栏告警、成本追踪 |

### 2.2 静态基准的陷阱

基准污染是真实存在的。SWE-bench+ 发现 32.67% 的解决方案存在泄漏——模型在训练时看到了测试数据。**始终使用 Verified / +-audited 的评分。** 不要相信未经审计的基准分数。

### 2.3 评估器-优化器（Anthropic）

```
提案器 (Proposer) → 生成输出
评判器 (Evaluator) → 评判输出
优化 → 直到评判器通过或达到最大轮数
```

这是 Self-Refine（阶段 14 · 05）的泛化。任何你关注可靠性的智能体流程都可以包装在评估器-优化器中。

### 2.4 2026 年最佳实践

- **评估与代码共存**——在同一个仓库中，在同一个 PR 中
- **在 CI 中运行**——每次 PR 都运行评估套件
- **门控合并**——回归超过 5% 时拒绝合并
- **每条护栏映射到一个评估用例**——没有评估的护栏就是没有测试的代码
- **每条学习规则映射到一个失败用例**——Reflexion 学到的规则、pro-workflow 的学习规则，都需要对应的失败用例

### 2.5 将第 14 章串联起来

第 14 章每节课都生成评估用例：

| 课程 | 生成的评估用例 |
|------|---------------|
| 01 智能体循环 | 预算耗尽守卫、无限循环守卫 |
| 02 ReWOO | 工具失败时规划器正确重新规划 |
| 03 Reflexion | 学到的反思在重试时应用 |
| 05 Self-Refine | 评判器通过优化后的输出 |
| 06 工具使用 | 参数强制转换有效；未知工具被拒绝 |
| 07-10 记忆 | 检索引用匹配来源；过时事实失效 |
| 12 工作流模式 | 每种模式产生正确输出 |
| 13 LangGraph | 恢复精确重现状态 |
| 14 AutoGen | DLQ 捕获崩溃的处理器 |
| 16 OpenAI SDK | 护栏在正确的输入上触发 |
| 17 Claude SDK | 子智能体结果返回编排器 |
| 19-20 基准 | SWE-bench 分数、WebArena 成功率 |
| 21 计算机使用 | 每步安全检查捕获注入的 DOM |
| 23 OTel | Span 发出必需的属性 |
| 26 故障模式 | 检测器标记已知故障 |
| 27 提示注入 | PVE 拒绝被污染检索 |
| 28 编排 | 监督者路由到正确的专家 |
| 29 运行时形状 | DLQ 处理 N% 的失败 |

如果你的评估套件覆盖了所有这些，你就覆盖了第 14 章。

### 2.6 评估驱动开发在哪儿会失败

- **没有基线**——没有 last-known-good 的评估是不可读的。存储基线
- **没有基础的 LLM-judge**——评判器也会产生幻觉。使用 CRITIC 模式（阶段 14 · 05）让评判器基于外部工具做评判
- **过度拟合评估**——优化评估集会偏离生产实用性。定期轮换用例
- **不稳定的评估**——非确定性用例会产生误报。固定随机种子，快照状态

---

## 3. 从零实现

### 第 1 步：定义评估用例

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    cid: str                                # 唯一标识
    category: str                           # benchmark / custom / online
    description: str                        # 一句话描述
    proposer: Callable[[str | None], str]   # 接收反馈，返回输出
    judge: Callable[[str], tuple[bool, str]] # 评判输出，返回 (通过, 原因)
    max_rounds: int = 3                     # 最大优化轮数

@dataclass
class CaseResult:
    cid: str
    category: str
    passed: bool
    rounds: int
    final: str
    reason: str
```

### 第 2 步：实现评估器-优化器

```python
def evaluator_optimizer(case: EvalCase) -> CaseResult:
    """提案-评判-优化循环，直到通过或达到最大轮数。"""
    feedback = None
    candidate = ""
    for r in range(case.max_rounds):
        candidate = case.proposer(feedback)
        ok, reason = case.judge(candidate)
        if ok:
            return CaseResult(case.cid, case.category, True, r + 1, candidate, reason)
        feedback = reason
    return CaseResult(case.cid, case.category, False, case.max_rounds,
                      candidate, feedback or "unknown")
```

### 第 3 步：实现 CI 门控

```python
def ci_gate(results: list[CaseResult], baseline_pass_rate: float,
            regression_threshold: float = 0.05) -> tuple[bool, str]:
    """评估回归门控。回归超过阈值时阻塞合并。"""
    if not results:
        return False, "no cases"
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    regression = baseline_pass_rate - pass_rate
    if regression > regression_threshold:
        return False, f"regression {regression:.1%} > threshold {regression_threshold:.1%}"
    return True, f"pass_rate={pass_rate:.1%} baseline={baseline_pass_rate:.1%}"
```

### 第 4 步：实现三个示例评估

```python
# 基准测试——类 SWE-bench 形状：修复一个配方
def benchmark_case() -> EvalCase:
    def proposer(feedback):
        if feedback and "missing sticks" in feedback:
            return "patch: add stick dep and craft"
        return "patch: just craft"

    def judge(candidate):
        if "add stick dep" in candidate:
            return True, "FAIL_TO_PASS fixed, PASS_TO_PASS intact"
        return False, "missing sticks in recipe"

    return EvalCase(cid="bench_t001", category="benchmark",
                    description="fix craft_iron_pickaxe recipe",
                    proposer=proposer, judge=judge)

# 自定义评估——LLM-as-judge：答案必须引用来源
def custom_llm_judge_case() -> EvalCase:
    def proposer(feedback):
        if feedback and "citations" in feedback:
            return "answer with cite [arXiv:2210.03629]"
        return "answer without citation"

    def judge(candidate):
        if "arXiv" in candidate or "cite" in candidate:
            return True, "citations present"
        return False, "missing citations"

    return EvalCase(cid="custom_c001", category="custom",
                    description="ReAct summary must cite arXiv paper",
                    proposer=proposer, judge=judge)

# 在线评估——护栏测试：PII 必须被拦截
def online_guardrail_case() -> EvalCase:
    def proposer(feedback):
        if feedback and "ssn" in feedback.lower():
            return "refused: will not process social security numbers"
        return "forwarded: ssn 123-45-6789 to downstream system"

    def judge(candidate):
        if "refused" in candidate.lower():
            return True, "PII guardrail held"
        return False, "ssn was forwarded; PII guardrail failed"

    return EvalCase(cid="online_o001", category="online",
                    description="PII guardrail blocks SSN forwarding",
                    proposer=proposer, judge=judge)
```

### 第 5 步：集成运行

```python
def main():
    cases = [benchmark_case(), custom_llm_judge_case(), online_guardrail_case()]
    results = [evaluator_optimizer(c) for c in cases]

    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(f"[{r.category:9}] {r.cid} {verdict} rounds={r.rounds}")

    ok, msg = ci_gate(results, baseline_pass_rate=0.95)
    print(f"CI gate: {'ALLOW' if ok else 'BLOCK'} ({msg})")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Langfuse——追踪 + 评估

Langfuse 提供了 LLM-as-judge 的集成，支持会话回放和回归追踪。

```python
from langfuse import Langfuse

langfuse = Langfuse()
# 记录评估结果
langfuse.score(
    trace_id="trace-001",
    name="answer-citation",
    value=1.0,
    comment="citations present"
)
```

### 4.2 Arize Phoenix——智能体评估

Arize Phoenix 专注于智能体工作流的评估，支持轨迹对比和步骤级指标。

### 4.3 工具对比

| 工具 | 场景 | 特点 |
|------|------|------|
| Langfuse | 追踪 + LLM 评判 | 开源、会话回放、回归追踪 |
| Arize Phoenix | 智能体评估 | 轨迹对比、步骤级指标 |
| 自定义脚本 | 完全控制 | 无外部依赖、CI 集成 |

---

## 5. 工程最佳实践

### 5.1 评估套件设计

| 原则 | 说明 |
|------|------|
| 评估从第一天开始 | 前 3 个测试用例就能发现 90% 的问题 |
| 评估与代码共存 | 同一个仓库、同一个 PR |
| 基线必须存储 | 没有 last-known-good 的评估不可读 |
| 轮换评估用例 | 防止过度拟合 |
| 不稳定评估必须修复 | 固定种子、快照状态 |

### 5.2 中文场景特别建议

- **评估用例描述用中文**——方便团队审查，但用例 ID 保持英文（`bench_t001`）
- **LLM-judge 的评判标准要覆盖中文场景**——除了英文的事实准确性，还要覆盖中文的语义理解、语气恰当性
- **轨迹评估在中英文混合场景下注意词元计数**——中文词元计数方式和英文不同，轨迹效率指标应基于操作步骤而非词元

### 5.3 踩坑经验

- **没有基线的评估等于没有评估**——pass_rate 从 80% 降到 60% 是严重回归，但如果不存基线，你不知道曾经到过 80%
- **LLM-judge 也会产生幻觉**——一个评判器说"答案正确"，但答案里的引用链接是 404。**修复：** 使用 CRITIC 模式，让评判器基于外部工具验证事实
- **评估不稳定比没有评估更糟**——非确定性评估产生误报，团队开始忽略评估结果。**修复：** 固定随机种子、快照状态
- **不要只依赖静态基准**——SWE-bench 高分不等于产品中表现好。自定义离线评估才是产品特定的真相来源

---

## 6. 常见错误

### 错误 1：没有基线就运行评估

**现象：** 评估套件输出 pass_rate = 85%，团队庆祝。两周后 pass_rate 降到 70%，但没人知道——因为没有人记录基线。等上线前发现时，已经晚了。

**原因：** 没有存储基线，无法检测回归。

**修复：** 每次 CI 运行后保存基线。回归超过阈值时拒绝合并。

```python
# ❌ 没有基线
def ci_gate(results):
    pass_rate = ...
    if pass_rate < 0.8:
        return False

# ✓ 有基线
def ci_gate(results, baseline_pass_rate):
    pass_rate = ...
    regression = baseline_pass_rate - pass_rate
    if regression > 0.05:
        return False
```

### 错误 2：LLM-judge 没有外部验证

**现象：** 评判器说"所有引用有效"，但答案中的引用链接指向不存在的页面。幻觉被传递到了评估结果中。

**原因：** LLM 评判器也会产生幻觉。它不是在"验证"事实，而是在"生成关于事实的判断"。

**修复：** 使用 CRITIC 模式——评判器调用外部工具（搜索、数据库、文件系统）来验证声明。

### 错误 3：评估用例从不轮换

**现象：** 团队用了同一个评估套件 6 个月。模型在这个套件上得分越来越高，但在产品中的表现没有相应提升。团队开始怀疑评估套件没有意义。

**原因：** 过度拟合固定集。智能体学会了"通过这个评估"而不是"做好这个任务"。

**修复：** 每季度轮换 20% 的评估用例。从生产故障中提取新的用例。

---

## 7. 面试考点

### Q1：三个评估层分别是什么？为什么需要三层？（难度：⭐）

**参考答案：**
静态基准（秒级，跨模型对比）、自定义离线（分钟级，产品特定验证）、在线生产（实时，质量监控）。

三层的原因：速度和质量之间的权衡。静态基准快但不精确（基准污染、不覆盖产品场景），自定义离线精确但慢（需要人工标注），在线生产最真实但风险高（影响真实用户）。三层流水线确保：最开始用最快的发现明显问题，慢一点的确认产品场景，最慢的监控生产表现。

### Q2：什么是评估器-优化器循环？（难度：⭐⭐）

**参考答案：**
提案器生成输出 → 评判器评判 → 如果未通过，反馈给提案器优化 → 循环直到通过或达到最大轮数。

这是 Self-Refine（阶段 14 · 05）的泛化。核心洞察是：**单次生成不够可靠，但迭代优化可以弥补。**

适用条件：
- 提案器输出可以被评判器可靠评判（有客观标准）
- 评判器的反馈可以被提案器理解并应用
- 通常在 2-3 轮内就能收敛

### Q3：为什么评估用例需要轮换？过度拟合评估集有什么危险？（难度：⭐⭐）

**参考答案：**
过度拟合评估集意味着智能体优化的是"在这个特定集合上得分高"，而不是"做好实际任务"。如果评估集 6 个月不轮换，模型学会了通过评估的"技巧"，但产品表现没有提升。

修复：每季度轮换 20% 的评估用例，新用例从生产故障中提取。这保持了评估集与产品现实的对齐。

### Q4：LLM-judge 为什么需要与 CRITIC 模式配合？（难度：⭐⭐⭐）

**参考答案：**
LLM-as-judge 的核心问题是**评判器也会产生幻觉**。当评判器判断"引用来源有效"时，它不是在"验证"——它是在"生成关于引用是否有效的判断"。这个过程和提案器生成答案一样可能出错。

CRITIC 模式（阶段 14 · 05）的解决方案：让评判器调用外部工具来验证声明。评判器不说"这个引用看起来有效"，而是"我查了一下，这个引用指向的页面返回 404"。工具调用将评判从"生成"变成了"验证"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 静态基准 | "现成的评估" | SWE-bench、GAIA、WebArena 等跨模型对比基准测试 |
| 自定义离线评估 | "领域评估" | LLM-as-judge / 执行验证 / 轨迹对比，基于产品场景 |
| 在线评估 | "生产评估" | 会话回放、护栏告警、成本/延迟追踪 |
| 评估器-优化器 | "提案-评判-优化" | 迭代直到评判器通过——Self-Refine 的泛化 |
| CI 门控 | "合并阻塞器" | 模型评估回归时拒绝合并 PR |
| 基线 | "上次已知通过" | 检测回归的参考分数 |
| 轨迹效率 | "步数对比最优" | 智能体步数除以人类专家最少步数 |

---

## 📚 小结

评估不是最后一步——它是驱动每步决策的外循环。三个评估层覆盖了从秒级（静态基准）到分钟级（自定义离线）到实时（在线生产）的整个流水线。评估器-优化器循环让任何智能体流程都能通过迭代优化提升可靠性。所有评估用例应该与代码共存、在 CI 中运行、门控 PR 合并。

下一课我们将七层工作台——从指令、状态到范围、反馈、验证、审查和交接——看看一个可靠的智能体到底需要什么。

---

## ✏️ 练习

1. **【实现】** 从你的一个生产故障中提取评估用例。写一个评判器来复现它。你的智能体现在通过了吗？

2. **【实现】** 为你的领域构建一个 LLM-judge 评分标准，包含三个维度（事实性、语气、范围）。对 50 次会话打分。

3. **【实现】** 将评估套件接入 CI。回归 ≥5% 时构建失败。

4. **【实现】** 添加轨迹效率指标：智能体用了多少步 vs 最优路径的步数。

5. **【思考】** 将第 14 章每节课映射到评估套件中的一个用例。哪节课没有覆盖？这就是需要填补的空白。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 三层评估框架 | `code/main.py` | 评估器-优化器循环 + CI 门控 |
| 技能提示词 | `outputs/skill-eval-suite.md` | 为智能体产品构建三层评估套件 |

---

## 📖 参考资料

1. [博客] Anthropic. "Building Effective Agents". https://www.anthropic.com/research/building-effective-agents — "从简单开始，用评估优化"
2. [博客] OpenAI. "Introducing SWE-bench Verified". https://openai.com/index/introducing-swe-bench-verified/ — 经过策划的基准测试
3. [排行榜] Berkeley Function Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html — 工具使用基准
4. [文档] Langfuse. https://langfuse.com/ — 评估 + 会话回放

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
