# 指令即约束规则——从"请小心"到机器可执行的规则

> 写成散文的指令是愿望。写成约束的规则是测试。工作台把每条规则变成智能体运行时可以检查、审查员事后可以验证的东西。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 32（最小工作台）
**预计时间：** ~50 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分路由散文和操作规则——哪些指令可以机器检查，哪些只是"请小心"
- [ ] 将启动规则、禁止操作、完成定义、不确定性处理和审批边界表达为机器可检查的约束
- [ ] 实现一个规则检查器，对照规则集对一次智能体运行进行评分
- [ ] 设计可差异化的规则集，使审查变更时能清楚看到每条规则的改动

---

## 1. 问题

你的 `AGENTS.md` 写得像一份新员工入职文档。它告诉智能体"要小心"、"要充分测试"、"不确定就问"。三天后，智能体交了一个没有测试的变更，编辑了禁止目录，而且从未问过——因为它根本不知道那条线在哪里。

指令在"可操作"时有力量，在"只是期许"时毫无作用。`"请小心处理生产数据"` 不能阻止智能体修改 `scripts/release.sh`。`"要充分测试"` 不能让智能体在提交前运行 `pytest`。问题不在于指令写得不够多，而在于它们没有被转化为可以被检查、被评分、被自动化的东西。

解决方案是把规则从散文变成约束。每条规则有一个名称、一个类别和一个检查函数。工作台在运行时检查它，审查员在代码审查时验证它，CI 在持续集成时报告它。

---

## 2. 概念

### 2.1 规则的五种类别

```
AGENTS.md（路由器）→ docs/agent-rules.md（完整规则集）→ rule_checker.py（检查器）→ rule_report.json（报告）→ 审查员
```

大多数规则可以归入五个类别：

| 类别 | 规则回答的问题 | 示例 |
|------|---------------|------|
| 启动 (Startup) | 工作开始前必须满足什么？ | "状态文件存在且是最新的" |
| 禁止 (Forbidden) | 绝对不能发生什么？ | "不要编辑 `scripts/release.sh`" |
| 完成定义 (Definition of Done) | 什么证明任务完成了？ | "pytest 退出码为 0 且验收命令通过" |
| 不确定性 (Uncertainty) | 智能体不确定时怎么做？ | "打开一个问答笔记，而不是猜测" |
| 审批 (Approval) | 什么需要人工审批？ | "任何新依赖、任何生产环境写操作" |

如果一条规则不适合这五个类别中的任何一个，它通常想成为两条规则。强制拆分。

### 2.2 操作规则 vs 期许规则

```
期许规则（❌）：    "请小心处理生产数据"
操作规则（✓）：    "任何写入 production/ 目录的操作必须经过审批检查"
```

区别很简单：**操作规则有一个检查函数**。`rule_checker.py` 中定义了一个 Python 函数，工作台可以在运行时调用它来判断这条规则是否被遵守。没有检查函数的规则要么删除，要么升级为可检查的形式。

### 2.3 渐进式披露：地图，不是百科全书

`AGENTS.md` 不断增长的原因是每次事故都添加一条规则，但没有事故会删除规则。一年后，文件变成两千行，智能体读完第一页就耗尽了注意力预算，只执行了被告诉的一小部分。

修复方案不是写更短的文件，而是分层：

```
AGENTS.md                  # 路由器，< 50 行：这个仓库是什么，去哪里找，5 条硬规则
docs/
  agent-rules.md           # 完整规则集（本课）
  architecture.md          # 任务涉及模块边界时加载
  testing.md               # 任务涉及测试时加载
  deploy.md                # 仅在发布任务时加载，受审批规则门控
```

| 层级 | 位置 | 何时读取 | 大小预算 |
|------|------|---------|---------|
| 路由器 | `AGENTS.md` | 每次会话，始终读取 | 不超过 ~50 行 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时读取 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅在任务涉及该主题时加载 | 按需深入 |

两个测试保证分层有效：

- **可达性测试**——智能体从路由器出发，最多两跳就能到达任何规则。路由器必须用路径链接每个主题文档，而不是用散文描述它
- **新鲜度测试**——路由器足够短，审查员每次 PR 都会重读。断裂的链接比缺失的规则更糟糕——路由器中的坏链接本身就是启动检查违规

### 2.4 规则 vs 框架护栏

框架护栏（如 OpenAI Agents SDK 的 guardrails、LangGraph 的 interrupts）在运行时层面强制执行规则。本课的规则集是人类可读、可审查的契约，护栏执行的就是这些规则。两者都需要：运行时在单次运行中捕获违规，规则集证明运行时在做正确的事情。

---

## 3. 从零实现

### 第 1 步：定义规则数据结构

每条规则有四个字段：slug（标识符）、category（类别）、check（检查函数名）、description（描述）。

```python
from dataclasses import dataclass

@dataclass
class Rule:
    slug: str          # 如 "startup/state-file-fresh"
    category: str      # startup, forbidden, definition_of_done, uncertainty, approval
    check: str         # rule_checker.py 中的函数名
    description: str   # 一句话描述
```

### 第 2 步：解析 agent-rules.md

规则文件使用 Markdown 格式，每条规则一个标题块，方便差异比较和人工审查。

```python
import re

def parse_rules(path) -> list[Rule]:
    text = path.read_text()
    rules = []
    # 按 "## " 分割为规则块
    for block in re.split(r"\n## ", text)[1:]:
        head, *rest = block.split("\n", 1)
        slug = head.strip()
        body = rest[0] if rest else ""
        # 提取 category 和 check 字段
        cat = re.search(r"-\s*category:\s*(\S+)", body)
        chk = re.search(r"-\s*check:\s*(\S+)", body)
        desc = [ln.strip() for ln in body.splitlines() if ln.strip()][-1]
        if cat and chk:
            rules.append(Rule(slug, cat.group(1), chk.group(1), desc))
    return rules
```

### 第 3 步：实现规则检查器

每条规则对应一个检查函数。函数接收运行轨迹（TurnTrace），返回 True/False。

```python
from dataclasses import dataclass, field

@dataclass
class TurnTrace:
    """一次智能体运行的轨迹记录。"""
    read_state_file: bool           # 是否读取了状态文件
    edited_files: list[str]         # 编辑了哪些文件
    confidence: float               # 智能体的置信度
    asked_for_help: bool            # 是否请求了人工帮助
    tests_exit_code: int | None     # 测试退出码
    added_dependencies: list[str]   # 新增的依赖
    approvals: list[str] = field(default_factory=list)  # 已审批的项

class RuleChecker:
    def state_file_fresh(self, trace: TurnTrace) -> bool:
        """启动规则：必须先读取状态文件。"""
        return trace.read_state_file

    def no_release_script_edits(self, trace: TurnTrace) -> bool:
        """禁止规则：不得编辑发布脚本。"""
        return "scripts/release.sh" not in trace.edited_files

    def tests_pass(self, trace: TurnTrace) -> bool:
        """完成定义：测试必须通过。"""
        return trace.tests_exit_code == 0

    def opened_question_when_unsure(self, trace: TurnTrace) -> bool:
        """不确定性规则：不确定时必须提问。"""
        return trace.confidence >= 0.7 or trace.asked_for_help

    def new_dependency_approved(self, trace: TurnTrace) -> bool:
        """审批规则：新依赖必须经过审批。"""
        if not trace.added_dependencies:
            return True
        return all(dep in trace.approvals for dep in trace.added_dependencies)
```

### 第 4 步：评分与报告

```python
def score(rules, checker, trace):
    """对照规则集对一次运行评分。"""
    results = []
    for rule in rules:
        check_fn = getattr(checker, rule.check, None)
        passed = bool(check_fn(trace)) if check_fn else False
        results.append({
            "slug": rule.slug,
            "category": rule.category,
            "passed": passed,
        })
    return results
```

### 第 5 步：运行演示

```python
# 模拟一次"坏"的运行：没读状态文件、编辑了发布脚本、测试失败、不确定但没问
bad_trace = TurnTrace(
    read_state_file=False,
    edited_files=["app.py", "scripts/release.sh"],
    confidence=0.4,
    asked_for_help=False,
    tests_exit_code=1,
    added_dependencies=["fastapi"],
)

# 模拟一次"好"的运行
good_trace = TurnTrace(
    read_state_file=True,
    edited_files=["app.py", "test_app.py"],
    confidence=0.9,
    asked_for_help=False,
    tests_exit_code=0,
    added_dependencies=[],
)

checker = RuleChecker()
bad_results = score(rules, checker, bad_trace)
good_results = score(rules, checker, good_trace)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Claude Code 的规则读取

Claude Code 在会话启动时读取 `AGENTS.md`，并在拒绝操作时引用规则。规则检查器在 CI 中重新运行，捕获静默漂移。你不需要额外的工具——`AGENTS.md` + `docs/agent-rules.md` 就是完整的规则系统。

### 4.2 OpenAI Agents SDK 的护栏

```python
from openai_agents import Agent, guardrail

# 将规则注册为输入和输出护栏
agent = Agent(
    name="coder",
    instructions="读取 agent-rules.md 并遵守所有规则",
    input_guardrails=[rule_check_guardrail],
    output_guardrails=[rule_check_guardrail],
)
```

Markdown 是文档层，SDK 是运行时层。两者读取同一套规则。

### 4.3 LangGraph 的中断机制

LangGraph 的 interrupts 在运行中的节点违反规则时触发。中断处理器读取规则，询问人工，然后恢复执行。规则集是跨三个框架可移植的——因为它只是 Markdown 加函数名。

### 4.4 实践模式对照

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 规则写入时标记严重性 | 所有场景 | `block`、`warn`、`info` 三级，写入时确定而非事后追加 |
| 规则过期机制 | 规则集超过 30 条 | 90 天无违规的规则进入季度审查，决定保留、降级或删除 |
| Markdown 源文件 + JSON 缓存 | CI 集成 | `agent-rules.md` 是编写文件，`agent-rules.lock.json` 是预提交钩子生成的缓存 |

---

## 5. 工程最佳实践

### 5.1 规则集的设计原则

| 原则 | 说明 |
|------|------|
| 每条规则必须有检查函数 | 没有检查函数的规则要么删除，要么升级 |
| 严重性在写入时确定 | 团队早期倾向于高估严重性，deadline 时悄悄降低；写入时强制校准 |
| 规则过期是强制函数 | 90 天无违规 → 季度审查；有数据支撑（Cloudflare 131,246 次审查运行） |
| 路由器保持精简 | `AGENTS.md` 不超过 50 行，只有指针，没有百科全书 |

### 5.2 中文场景特别建议

- **规则描述使用中文**——`agent-rules.md` 的 description 字段用中文写，方便团队审查
- **slug 使用英文**——标识符保持英文，便于跨工具兼容
- **分层文档的路径用英文**——`docs/agent-rules.md` 而不是 `docs/智能体规则.md`，避免路径编码问题

### 5.3 踩坑经验

- **规则集膨胀是最大的敌人**——没有过期机制的规则集会从 15 条增长到 80+ 条，其中大部分从未触发。设置 90 天过期，季度审查
- **不要把规则和教程混在一起**——`AGENTS.md` 是路由器，不是新手指南。保持 50 行以内
- **严重性标记要诚实**——`block` 意味着"违规时停止运行"。如果你的团队实际上会在 deadline 时绕过它，那就标记为 `warn`

---

## 6. 常见错误

### 错误 1：规则写成散文，没有检查函数

**现象：** `AGENTS.md` 写了 200 行指令，但智能体从未遵守过。每次事故后又加 10 行，文件越来越长，遵守率越来越低。

**原因：** `"请仔细测试"` 无法被机器检查。没有检查函数的规则等于没有规则。

**修复：**
```python
# ❌ 期许规则——无法检查
"Please test thoroughly before committing."

# ✓ 操作规则——有明确的检查函数
def tests_pass(trace: TurnTrace) -> bool:
    return trace.tests_exit_code == 0
```

### 错误 2：路由器膨胀成百科全书

**现象：** `AGENTS.md` 从 50 行增长到 2000 行。智能体每次会话都读取全文，但注意力预算在前 50 行就耗尽了。后面的规则从未被执行。

**原因：** 没有分层。所有内容都堆在路由器里。

**修复：** 将详细内容移到 `docs/<topic>.md`，路由器只保留指针。`AGENTS.md` 不超过 50 行。

### 错误 3：严重性事后标记

**现象：** 所有规则都标记为 `block`，但团队在 deadline 时悄悄绕过 `block` 规则。结果是 `block` 变成了摆设，真正重要的规则也被忽视。

**原因：** 严重性应该在写入时确定，而不是事后追加。`block` 是硬性停止，必须与 `overrides.jsonl` 审计日志配对。

**修复：** 写入时标记 `block`/`warn`/`info`，与验证门控（阶段 14 · 38）配对。`block` 规则的任何覆盖都必须记录在审计日志中。

### 错误 4：规则不设过期日期

**现象：** 两年前的规则仍然生效，但已经不再适用。新团队成员不敢删除任何规则，因为不知道哪条是"重要的"。

**原因：** 没有 `expires_at` 字段。没有过期机制的规则集会无限增长。

**修复：** 每条规则添加 `expires_at` 字段（默认 90 天）。90 天无违规的规则进入季度审查，决定保留、降级为 `info` 或删除。

---

## 7. 面试考点

### Q1：什么是操作规则？什么是期许规则？各举一例。（难度：⭐）

**参考答案：**
操作规则是可以被机器检查的规则——它有一个检查函数，工作台在运行时可以调用它。例如："测试退出码必须为 0" 对应 `tests_pass(trace) -> trace.tests_exit_code == 0`。

期许规则是无法被机器检查的指令——"请小心处理生产数据" 没有检查函数，智能体无法判断自己是否遵守了它。期许规则要么删除，要么升级为操作规则。

### Q2：为什么规则集需要分层？路由器应该包含什么？（难度：⭐⭐）

**参考答案：**
规则集不分层会导致路由器膨胀。智能体每次会话都读取全文，但注意力预算有限——2000 行的 `AGENTS.md` 只有前 50 行会被真正执行。

路由器（`AGENTS.md`）只包含：仓库简介、去哪里找什么信息、5 条硬规则。详细内容移到 `docs/<topic>.md`，智能体只在任务涉及该主题时加载。两个测试保证分层有效：可达性测试（任何规则最多两跳可达）和新鲜度测试（路由器足够短，审查员每次 PR 都会重读）。

### Q3：规则过期机制为什么重要？Cloudflare 的数据说明了什么？（难度：⭐⭐）

**参考答案：**
没有过期机制的规则集会无限增长。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，5,169 个仓库，131,246 次审查运行）显示：有过期机制的规则集保持在 30 条以内；没有的则增长到 80+ 条，其中大部分从未触发。

过期机制的工作方式：每条规则有 `expires_at` 字段（默认 90 天）。如果 90 天内没有触发任何违规，规则进入季度审查——要么有理由保留，要么降级为 `info`，要么删除。这是一种"用数据驱动规则清理"的强制函数。

### Q4：如何将规则集与 LangGraph 的中断机制集成？（难度：⭐⭐⭐）

**参考答案：**
规则集是人类可读的 Markdown，LangGraph 的中断是运行时层面的拦截。集成方式：在规则检查器中，当检测到违规时调用 `langgraph.interrupt()`，传入规则描述。中断处理器读取规则，询问人工决策，然后恢复或终止运行。

关键点：Markdown 是文档层（可审查），SDK 是运行时层（可执行）。两者读取同一套规则，但职责不同。规则集的可移植性来自于它只是 Markdown 加函数名——不依赖任何特定框架。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 操作规则 (Operational Rule) | "真正的指令" | 工作台可以在运行时检查的规则——有检查函数 |
| 期许规则 (Aspirational Rule) | "请小心" | 没有检查函数的规则——要么删除，要么升级 |
| 完成定义 (Definition of Done) | "验收通过" | 客观的、文件支持的任务完成证明 |
| 严重性 (Severity) | "硬规则" | `block`（违规时停止运行）/`warn`（警告）/`info`（记录） |
| 规则过期 (Rule Expiry) | "清理过期规则" | N 天无违规的规则进入审查——保留、降级或删除 |
| 渐进式披露 (Progressive Disclosure) | "分层文档" | 路由器保持精简，详细内容按需加载 |

---

## 📚 小结

"请小心"不是指令，是愿望。本课将智能体指令从散文转化为机器可执行的约束规则——每条规则有类别、有检查函数、有严重性级别。你实现了一个规则解析器和检查器，理解了分层文档设计的两个测试（可达性和新鲜度），以及规则过期机制如何防止规则集膨胀。

下一课我们将解决另一个关键问题：智能体的状态如何跨会话持久化——聊天记录是易失的，仓库才是持久的。

---

## ✏️ 练习

1. 【理解】找到一个真实的 `AGENTS.md` 文件，用本课的五种类别（启动、禁止、完成定义、不确定性、审批）对其进行分类。有多少行是操作规则？有多少行是期许规则？

2. 【实现】为 `RuleChecker` 添加严重性支持。每条规则携带 `severity`（`block`/`warn`/`info`），评分报告按严重性聚合。`block` 规则违规时输出特殊标记。

3. 【实现】将规则检查器集成到 CI 中。在 GitHub Actions 的 `agent-run` 步骤之后添加一个 `rule-check` 步骤，使用 `rule_report.json` 作为输入，任何 `block` 规则失败时构建失败。

4. 【实现】为每条规则添加 `expires_at` 字段。实现一个 `stale_rules` 函数，返回 60 天内无违规的规则列表。输出格式适合季度审查会议。

5. 【思考】框架护栏（如 OpenAI Agents SDK 的 guardrails）和本课的规则集有什么区别？如果两者都有，什么时候一个会失效而另一个不会？写 200 字以内的分析。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 规则集 | `code/agent-rules.md` | 五种类别的示例规则集，可直接作为模板 |
| 规则检查器 | `code/main.py` | 解析规则文件、运行检查、生成报告 |
| 技能提示词 | `outputs/skill-rule-set-builder.md` | 面谈项目负责人，将散文指令转化为五类别规则集 |

---

## 📖 参考资料

1. [官方文档] OpenAI Agents SDK Guardrails: https://platform.openai.com/docs/guides/agents-sdk/guardrails
2. [官方文档] LangGraph Interrupts: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/
3. [论文] Anthropic. "Building Effective Agents". https://www.anthropic.com/research/building-effective-agents
4. [博客] Cloudflare. "Orchestrating AI Code Review at Scale". https://blog.cloudflare.com/ai-code-review/ — 131,246 次审查运行，规则组合经验
5. [GitHub] logi-cmd/agent-guardrails. https://github.com/logi-cmd/agent-guardrails — 合并门控实现：作用域、变异测试、违规预算

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
