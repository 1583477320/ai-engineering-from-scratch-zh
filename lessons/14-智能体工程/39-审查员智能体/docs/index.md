# 审查员智能体——构建者和评分者应该分开

> 写代码的智能体不能给它自己打分。审查员是第二个循环——不同的系统提示词、不同的目标、对构建者产出的一切只读。构建者和审查员之间的差距，正是大多数可靠性所在。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 38（验证门控）
**预计时间：** ~55 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 05（Self-Refine）— 单智能体自审查基线；阶段 14 · 30（评估驱动开发）— 校准集生成器

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么同一个智能体不能可靠地审查自己的作品
- [ ] 构建一个审查员智能体循环——消费构建者工件，生成结构化审查报告
- [ ] 编写审查员评分标准——在特定维度上打分，而不是凭感觉
- [ ] 将审查员接入工作台，使人工作步骤从真实工件开始
- [ ] 理解审查员偏差的四种来源及缓解措施

---

## 1. 问题

你让智能体修复一个 bug。它编辑了四个文件，运行了测试，报告完成。验证门控（阶段 14 · 38）确认验收命令运行了、范围守住了。门控说 `passed: true`。你合入。两天后你发现修复只解决了 bug 的一半。

**验收是必要的，但不充分。** 审查员问的是验收无法回答的问题：这个改动解决了正确的问题吗？它是否扩大了范围但没有标记？它是否记录了本应被质疑的假设？它是否让工作台处于下一个会话可以继续的状态？

---

## 2. 概念

### 2.1 审查员与构建者分离

```
构建者智能体 → 工件（差异 + 状态 + 反馈 + 裁决）
                  ↓
            审查员智能体 → 评分标准（reviewer_checklist.md）
                  ↓
            审查报告（review_report.json）
                  ↓
            人工签字
```

核心约束：**审查员不能修改差异。** 它读取差异、状态、反馈、裁决。它写报告。它不打补丁。如果报告说"修复这个"，下一个构建者轮次来做修复；审查员回去继续审查。混合角色会破坏这个差距。

### 2.2 审查员评分标准：五个维度

每个维度 0-2 分，满分 10 分。

| 维度 | 问题 |
|------|------|
| **问题匹配** | 这个改动解决的是任务描述的问题，还是附近的问题？ |
| **范围纪律** | 修改是否限于契约范围，还是刻意扩展了契约？ |
| **假设记录** | 所有隐含假设是否写在了可审查的地方？ |
| **验证质量** | 验收命令真正证明了目标，还是证明了一个弱化版本？ |
| **交接就绪** | 下一个会话能从当前状态干净地继续吗？ |

7 分以下为软失败（soft_fail），需要修改；5 分以下为硬失败（hard_fail），停止并上报人工。

### 2.3 审查员是不同角色，不是不同模型

你可以用和构建者相同的模型运行审查员。关键在于角色分离：不同的系统提示词、不同的输入、对差异没有写权限。姿态的变化就是信号的变化。

### 2.4 审查员评分标准 vs 验证门控

门控（阶段 14 · 38）检查确定性事实：验收命令运行了吗？规则通过了吗？范围守住了吗？审查员做定性判断：这是正确的工作吗？有文档吗？交接可用吗？

两者都需要。**验证门控 + 审查员 = 完整覆盖。**

### 2.5 审查员偏差的四种来源

LLM 评判器有四种可靠的偏差（Adnan Masood, 2026 年 4 月）：

| 偏差 | 现象 | 缓解措施 |
|------|------|---------|
| **位置偏差** | GPT-4 约 40% 在 (A,B) vs (B,A) 排序上不一致 | 两种排序都评判，只计一致结果 |
| **冗长偏差** | 长输出得分偏高约 15% | 使用 1-4 分制，明确奖励简洁 |
| **自偏好** | 评判器偏好同一模型家族的输出 | 跨模型家族轮换评判器 |
| **权威偏差** | 评判器给知名作者的引用过高评分 | 评分前去除作者名 |

### 2.6 校准集

一个 10-20 个历史任务的集合，每个带有已知正确裁决。每次提示词变更时对校准集运行审查员。如果与历史记录的一致性低于 80%，评分标准需要在发布前修订。

---

## 3. 从零实现

### 第 1 步：定义输入和评分维度

```python
from dataclasses import dataclass, field

@dataclass
class ReviewerInputs:
    task_id: str
    goal: str
    diff_summary: dict[str, list[str]]    # 差异摘要
    state: dict[str, object]              # 状态文件
    feedback: list[dict[str, object]]     # 反馈记录
    verdict: dict[str, object]            # 验证门控裁决

@dataclass
class DimensionScore:
    name: str          # 维度名
    score: int         # 0-2
    note: str          # 评判依据

@dataclass
class ReviewReport:
    task_id: str
    total: int         # 总分 (0-10)
    verdict: str       # pass | soft_fail | hard_fail
    dimensions: list[DimensionScore] = field(default_factory=list)
```

### 第 2 步：实现评分函数

```python
def score_problem_fit(inputs: ReviewerInputs) -> DimensionScore:
    """这个改动解决了正确的目标吗？"""
    files = inputs.diff_summary.get("touched", [])
    goal = inputs.goal.lower()
    # 从目标中提取关键概念，检查修改的文件是否覆盖
    keywords = [w for w in goal.split() if len(w) > 4]
    hits = sum(any(k in f.lower() for f in files) for k in keywords)
    score = min(2, hits)
    return DimensionScore("problem_fit", score, f"keyword hits across touched files: {hits}")

def score_scope_discipline(inputs: ReviewerInputs) -> DimensionScore:
    """修改是否限于契约范围？"""
    off = inputs.verdict.get("findings", [])
    block_scope = [f for f in off if f.get("code") == "scope.forbidden"]
    if block_scope:
        return DimensionScore("scope_discipline", 0, "forbidden writes present")
    warn_scope = [f for f in off if f.get("code") == "scope.off_scope"]
    return DimensionScore("scope_discipline", 1 if warn_scope else 2,
                           f"off-scope warnings: {len(warn_scope)}")

def score_assumptions(inputs: ReviewerInputs) -> DimensionScore:
    """假设是否被记录？"""
    assumptions = inputs.state.get("assumptions") or []
    if not assumptions:
        return DimensionScore("assumptions", 1,
                               "no assumptions recorded; either trivial or undocumented")
    return DimensionScore("assumptions", 2, f"{len(assumptions)} assumptions recorded")

def score_verification(inputs: ReviewerInputs) -> DimensionScore:
    """验收命令真正证明了目标吗？"""
    exits = [rec.get("exit_code") for rec in inputs.feedback]
    if any(code is None for code in exits):
        return DimensionScore("verification_quality", 0, "feedback has missing exit codes")
    if all(code == 0 for code in exits) and exits:
        return DimensionScore("verification_quality", 2, "all feedback exit zero")
    return DimensionScore("verification_quality", 1, "mixed exit codes")

def score_handoff(inputs: ReviewerInputs) -> DimensionScore:
    """下一个会话能干净地继续吗？"""
    if inputs.state.get("active_task_id"):
        return DimensionScore("handoff_readiness", 1, "active task not closed")
    if inputs.state.get("next_action"):
        return DimensionScore("handoff_readiness", 2, "next_action set, task closed")
    return DimensionScore("handoff_readiness", 0, "no next_action recorded")
```

### 第 3 步：实现审查主函数

```python
SCORERS = [score_problem_fit, score_scope_discipline,
           score_assumptions, score_verification, score_handoff]

def review(inputs: ReviewerInputs) -> ReviewReport:
    """运行审查员评分标准，返回审查报告。"""
    dims = [fn(inputs) for fn in SCORERS]
    total = sum(d.score for d in dims)
    has_zero = any(d.score == 0 for d in dims)

    if has_zero or total < 5:
        verdict = "hard_fail"
    elif total >= 7:
        verdict = "pass"
    else:
        verdict = "soft_fail"

    return ReviewReport(task_id=inputs.task_id, total=total,
                        verdict=verdict, dimensions=dims)
```

### 第 4 步：运行演示

```python
def main():
    # 干净通过
    clean = ReviewerInputs(task_id="T-001", goal="add input validation to signup",
        diff_summary={"touched": ["app/signup.py", "tests/test_signup.py"]},
        state={"active_task_id": None,
               "assumptions": ["users sign up with email + password only"],
               "next_action": "pick next task from board"},
        feedback=[{"command": "pytest", "exit_code": 0}],
        verdict={"passed": True, "findings": []})

    # 错误的问题
    wrong = ReviewerInputs(task_id="T-002", goal="add input validation to signup",
        diff_summary={"touched": ["docs/api.md"]},
        state={"active_task_id": "T-002", "assumptions": [], "next_action": ""},
        feedback=[{"command": "pytest", "exit_code": 0}],
        verdict={"passed": True, "findings": [{"code": "scope.off_scope", "severity": "warn"}]})

    for case in (clean, wrong):
        report = review(case)
        print(f"task {report.task_id}: total={report.total}/10 verdict={report.verdict}")
        for d in report.dimensions:
            print(f"  {d.name:22} {d.score}  {d.note}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Cloudflare 的专家池架构

Cloudflare 2026 年 4 月的 AI 代码审查系统：在 30 天内对 5,169 个仓库的 48,095 个合并请求进行了 131,246 次审查运行。中位审查完成时间 3 分 39 秒。最多 7 个专家审查员（安全、性能、代码质量、文档、发布管理、合规、Codex）在审查协调器下并行运行。协调器去重发现并判断严重性。前沿模型只用于协调器；专家运行在更便宜的模型上。

### 4.2 生产模式

| 模式 | 说明 |
|------|------|
| **专家池** | 一个审查员处理所有维度适用于单人仓库。当项目有安全、性能、文档多个关注面时，拆分为专家审查员 |
| **偏差缓解** | 位置偏差 → 两种排序都评判；冗长偏差 → 明确奖励简洁；自偏好 → 跨模型轮换；权威偏差 → 评分前去作者名 |
| **校准集** | 10-20 个历史任务，已知正确裁决。每次提示词变更时运行。一致性低于 80% 时评分标准需要修订 |
| **混合规范** | 门控做确定性检查，审查员做语义检查。Artthropic 2026 年指南明确了这个分工 |

### 4.3 双模型配对

构建者运行在更快更便宜的模型上。审查员运行在更强、上下文更小的模型上，专注于判断。

---

## 5. 工程最佳实践

### 5.1 审查员设计原则

| 原则 | 说明 |
|------|------|
| 角色分离 | 同一模型可以做两种角色，但系统提示词和输入必须不同。审查员对差异只读 |
| 维度评分 | 精确打分（0-2），不凭感觉 |
| 校准集 | 10-20 个历史任务——一致性 < 80% 时修订评分标准 |
| 偏差意识 | 位置偏差、冗长偏差、自偏好、权威偏差——每种都有缓解措施 |

### 5.2 中文场景特别建议

- **评分标准中的维度描述用中文**——方便中文团队理解和使用
- **校准集要覆盖中文场景**——英文校准集可能覆盖不到中文特有的语义问题
- **权威偏差在中英文混合场景下更明显**——中文团队可能对英文论文引用评分过高。评分前去除作者名

### 5.3 踩坑经验

- **审查员可以修改差异**——审查员直接修改了构建者的代码，"修复"了问题。混合角色破坏了审查的意义。**修复：** 审查员对差异只读
- **没有校准集**——新的评分标准上线后，发现所有任务都是 soft_fail。花了三天调整标准。**修复：** 上线前用 20 个历史任务校准
- **位置偏差导致评分不稳定**——同一个差异换了个顺序就从 pass 变成 fail。**修复：** 两种排序都评判，只计一致结果

---

## 6. 常见错误

### 错误 1：构建者和审查员是同一个人

**现象：** 审查员和构建者使用相同的系统提示词、相同的工作目录、对差异有写权限。审查员读完差异说"看起来不错"，然后顺手修正了一个注释拼写错误。

**原因：** 没有角色分离。审查员成了构建者的延伸，没有独立判断。

**修复：**
```
# ❌ 审查员与构建者相同
构建者：编辑代码 → 审查员：读取同一份差异，可以修改 → 误导性的"通过"

# ✓ 审查员与构建者分离
构建者：编辑代码 → 工件 → 审查员：只读取，写报告 → 独立判断
```

### 错误 2：评分标准没有维度，凭感觉打分

**现象：** 审查员的输出是"看起来不错"或"这个有问题"，没有具体的维度评分、没有评判依据。构建者不知道怎么改进。

**原因：** 没有结构化的评分标准。

**修复：** 五个维度，每个 0-2 分。每个维度有明确的问题。审查员必须给出每个维度的评分和依据。

### 错误 3：不用校准集

**现象：** 新的评分标准上线后，所有任务都是 hard_fail。或者所有任务都是 pass——审查员没有发现任何问题。都是没有校准的结果。

**原因：** 评分标准没有用历史数据验证过。

**修复：** 评分标准上线前，用 20 个历史任务（已知正确裁决）运行审查员。一致性 < 80% 时修订标准。

---

## 7. 面试考点

### Q1：为什么审查员必须和构建者分离？角色分离的核心是什么？（难度：⭐⭐）

**参考答案：**
同一个智能体不能可靠地审查自己的作品。角色分离的核心是：不同的系统提示词、不同的输入、对差异没有写权限。

即使使用同一个模型，改变系统提示词（从"生成代码"变成"审查代码"）和输入范围（只读）就能产生不同的信号。写代码的智能体关注"能否工作"，审查员关注"是否做了正确的工作"。这两个视角必须分开。

### Q2：审查员的五个评分维度是什么？为什么需要结构化评分？（难度：⭐）

**参考答案：**
问题匹配、范围纪律、假设记录、验证质量、交接就绪。每个 0-2 分，满分 10 分。

结构化评分的原因是：凭感觉打分的审查员输出"看起来不错"——构建者不知道怎么改进。维度评分让构建者知道具体在哪个维度不足，让审查员的判断是可操作、可追踪的。

### Q3：LLM 评判器的四种偏差是什么？如何缓解？（难度：⭐⭐⭐）

**参考答案：**
（1）**位置偏差**——GPT-4 约 40% 在 (A,B) vs (B,A) 排序上不一致。缓解：两种排序都评判，只计一致结果。

（2）**冗长偏差**——长输出得分偏高约 15%。缓解：使用 1-4 分制，明确奖励简洁。

（3）**自偏好**——评判器偏好同一模型家族的输出。缓解：跨模型家族轮换评判器。

（4）**权威偏差**——评判器给知名作者的引用过高评分。缓解：评分前去除作者名。

### Q4：什么是校准集？为什么需要它？（难度：⭐⭐）

**参考答案：**
校准集是 10-20 个历史任务的集合，每个带有已知正确裁决。每次评分标准变更时对校准集运行审查员。如果与历史记录的一致性低于 80%，评分标准需要在发布前修订。

没有校准集，评分标准的第一次上线就像盲飞。校准集确保审查员在真实数据上的表现可衡量、可验证。这也是阶段 14 · 30（评估驱动开发）在审查员场景中的具体应用。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 审查员评分标准 | "检查清单" | 五个维度 0-2 分，每个维度有明确的问题 |
| 软失败 (Soft Fail) | "需要修改" | 总分低于 7 分；构建者需要处理发现的问题 |
| 硬失败 (Hard Fail) | "拒绝" | 总分低于 5 分或任何维度为 0；停止并上报人工 |
| 角色分离 | "不同的提示词" | 同一模型可以扮演两种角色，关键是输入和姿态 |
| 校准集 | "历史对照" | 10-20 个已知正确裁决的历史任务——验证评分标准 |

---

## 📚 小结

验收是必要的，但不充分。验证门控回答"做对了吗"，审查员回答"做了对的事吗"。五个维度（问题匹配、范围纪律、假设记录、验证质量、交接就绪）结构化评分确保审查员的判断是可操作、可追踪的。LLM 评判器有四种偏差需要主动缓解，校准集确保评分标准在真实数据上的表现可衡量。

下一课我们将构建多会话交接机制——让下一个会话从停止的地方继续，而不是从零开始。

---

## ✏️ 练习

1. **【思考】** 为你的产品领域添加第六个维度。论证为什么它不能归入现有五个维度。

2. **【实现】** 用两个不同的系统提示词（简洁的、冗长的）运行审查员。哪个生成的报告人类更愿意阅读？

3. **【实现】** 为每个维度添加 `confidence` 字段。最低维度置信度低于 0.6 时拒绝交付报告。

4. **【实现】** 构建校准集：10 个已知正确裁决的历史任务。运行审查员查看其在哪些地方与历史记录不一致。

5. **【思考】** 添加"请求更多证据"功能：审查员可以在评分前要求构建者运行特定的测试。如何设计退避机制避免死循环？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 审查员智能体 | `code/main.py` | 五维度评分标准、软/硬失败裁决 |
| 技能提示词 | `outputs/skill-reviewer-agent.md` | 项目特定的审查员评分标准和与验证门控的集成 |

---

## 📖 参考资料

1. [官方文档] OpenAI Agents SDK Handoffs: https://platform.openai.com/docs/guides/agents-sdk/handoffs
2. [官方文档] Anthropic Claude Code Subagents: https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/sub-agents
3. [博客] Cloudflare. "Orchestrating AI Code Review at Scale". https://blog.cloudflare.com/ai-code-review/ — 7 专家 + 协调器架构，131k 次审查/30 天
4. [博客] Adnan Masood. "Rubric-Based Evaluations and LLM-as-a-Judge: Methodologies, Biases, Empirical Validation". https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80 — 4 种偏差及缓解措施
5. [论文] "Agent-as-a-Judge: Evaluating Agents with Agents". OpenReview / ICLR. https://openreview.net/forum?id=DeVm3YUnpj

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
