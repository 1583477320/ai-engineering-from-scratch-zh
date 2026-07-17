# 范围契约与任务边界——让"别跑偏"不再只是一句嘱咐

> 模型不知道工作在哪儿结束。范围契约是一个按任务编写的文件，说清工作在哪儿开始、在哪儿结束、越界了怎么回滚。它把"别跑偏"从愿望变成了可检查的规则。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 32（最小工作台）、阶段 14 · 33（规则即约束）
**预计时间：** ~50 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 编写一个智能体在任务启动时读取、验证器在任务结束时检查的范围契约
- [ ] 指定允许的文件、禁止的文件、验收条件、回滚计划和审批边界
- [ ] 实现一个范围检查器，将差异与契约对比并标记违规
- [ ] 设计违规预算机制，让门控在实践中可用而不被禁用

---

## 1. 问题

智能体总是在跑偏。任务是"修复登录 bug"。生成的差异却涉及登录路由、邮件辅助函数、数据库驱动、README 和发布脚本。每一次修改在当下都有个合理理由。但把它们合在一起，就和当初审查的那个变更完全不同了。

范围蔓延是智能体工作中最被低估的失败模式——因为智能体每一步都在真诚地叙述它正在做的事。问题不在于提示词不够严厉，而在于没有一个写在磁盘上的契约说清承诺了什么，以及一个检查来对比结果与承诺。

解决方案是**范围契约（Scope Contract）**：每个任务启动时写入一个 JSON 文件，列出允许的文件、禁止的文件、验收条件、回滚计划，以及需要审批才能越界的行为。智能体在任务结束时生成差异，检查器将差异与契约对比，任何越界都会被标记。

---

## 2. 概念

### 2.1 范围契约包含什么

```
任务 → scope_contract.json → 智能体循环 → 最终差异 → scope_checker.py → 裁决（在范围内/越界）
```

| 字段 | 用途 |
|------|------|
| `task_id` | 关联到任务板上的任务 |
| `goal` | 一句话目标，审查员可以验证 |
| `allowed_files` | 智能体可以写入的 glob 模式 |
| `forbidden_files` | 智能体绝对不可触碰的 glob 模式 |
| `acceptance_criteria` | 证明完成的测试命令或断言 |
| `rollback_plan` | 操作员可以执行的一段回滚步骤 |
| `approvals_required` | 需要人工审批才能越界的行为 |

没有 `forbidden_files` 的契约是不完整的。**否定空间（negative space）占了契约的一半。**

### 2.2 使用 Glob，不用原始路径

真实仓库会移动文件。将契约绑定到 glob（如 `app/**/*.py`、`tests/test_signup*.py`），这样跨会话的重构不会使契约失效。

### 2.3 回滚是范围的一部分

列出回滚方式会强迫契约作者思考可能出什么问题。一个无法回滚的契约不应该被批准。

### 2.4 范围检查是差异检查

智能体产生差异。检查器读取差异、允许的 glob、禁止的 glob，以及运行的验收命令列表。每次违规是一个带标签的发现，验证门控可以拒绝它。

### 2.5 两种范围层次：功能列表与任务契约

范围契约只约束**一个任务**。智能体可以完美地遵守登录修复的契约，然后在下一轮决定项目还需要一个设置页面、一个深色模式开关和重写路由器。契约从未被问过"什么工作在项目范围内"——只问了"哪些文件在任务范围内"。

第二种层次需要自己的原语：`feature_list.json`，智能体在会话启动时读取。它是项目待办事项的机器可读、有序文件：

```json
{
  "project": "知识库系统",
  "active": "import-pdf",
  "features": [
    { "id": "import-pdf",      "status": "in_progress", "goal": "导入 PDF 到知识库",          "done_when": "pytest tests/test_import.py && 示例 PDF 出现在知识库视图中" },
    { "id": "full-text-search", "status": "todo",        "goal": "搜索文档内容并排序结果",      "done_when": "查询返回排序结果包含片段" },
    { "id": "cite-answers",    "status": "todo",        "goal": "答案附带来源引用",             "done_when": "每个答案至少渲染一个可点击的引用" }
  ]
}
```

两个规则让功能列表实用而非装饰：

- **"最多一个 `in_progress`" 不变式本身是一条启动检查规则**（阶段 14 · 33）：如果列表中有两个 `in_progress`，会话拒绝启动，直到人工解决
- 功能列表是文件，不是聊天消息——聊天滚动出上下文，文件跨会话、跨智能体持久化

契约和列表通过最小权限组合：任务契约的 `allowed_files` 必须落在功能列表当前任务的影响范围内，不能超出它。

### 2.6 多契约合并语义（最小权限）

当两个范围契约同时适用（如项目级契约 + 任务级契约），合并规则是：

| 维度 | 合并方式 |
|------|---------|
| `allowed_files` | **交集**（两个契约都必须允许该路径）|
| `forbidden_files` | **并集**（任意一个契约禁止即禁止）|
| `time_budget_minutes` | **最小值**（最严格的生效）|
| `approvals_required` | **累积** |
| `network_egress` | `None` 无限制 / `[]` 拒绝所有 / `[...]` 允许列表；合并时 `None` 延后到对方，两个列表取交集 |

---

## 3. 从零实现

### 第 1 步：定义数据结构

```python
from dataclasses import dataclass, field

@dataclass
class ScopeContract:
    task_id: str
    goal: str
    allowed_files: list[str]          # glob 模式
    forbidden_files: list[str]        # glob 模式
    acceptance_criteria: list[str]    # 验收命令
    rollback_plan: str                # 回滚计划
    approvals_required: list[str] = field(default_factory=list)
    time_budget_minutes: int | None = None
    network_egress: list[str] | None = None  # None=无限制, []=拒绝所有, [...]=允许列表
    violation_budget: int = 0         # 违规预算
    docs_paths_soft: list[str] = field(default_factory=lambda: ["docs/**", "README.md", "**/*.md"])

@dataclass
class RunSummary:
    touched_files: list[str]        # 修改过的文件
    commands_run: list[str]         # 运行的命令
    elapsed_minutes: float = 0.0    # 已用时间
    network_hosts: list[str] = field(default_factory=list)

@dataclass
class Finding:
    code: str         # 如 "scope.forbidden"
    severity: str     # block / warn / info
    detail: str
```

### 第 2 步：实现 Glob 匹配

```python
import fnmatch

def matches_any(path: str, patterns: list[str]) -> bool:
    """检查路径是否匹配任意一个 glob 模式。"""
    return any(fnmatch.fnmatch(path, p) for p in patterns)
```

### 第 3 步：实现范围检查

```python
def scope_check(contract: ScopeContract, run: RunSummary):
    in_scope = []
    off_scope = []
    soft_off_scope = []
    forbidden = []

    for path in run.touched_files:
        if matches_any(path, contract.forbidden_files):
            forbidden.append(path)
        elif matches_any(path, contract.allowed_files):
            in_scope.append(path)
        elif matches_any(path, contract.docs_paths_soft):
            soft_off_scope.append(path)    # 文档越界，通常是 warn
        else:
            off_scope.append(path)

    missing = [c for c in contract.acceptance_criteria if c not in run.commands_run]

    findings = []
    if forbidden:
        findings.append(Finding("scope.forbidden", "block", f"forbidden writes: {forbidden}"))
    if off_scope:
        findings.append(Finding("scope.off_scope", "warn", f"off-scope writes: {off_scope}"))
    if soft_off_scope:
        findings.append(Finding("scope.soft_off_scope", "info", f"docs/markdown off-scope: {soft_off_scope}"))
    if missing:
        findings.append(Finding("acceptance.missing", "block", f"acceptance not run: {missing}"))
    if contract.time_budget_minutes is not None and run.elapsed_minutes > contract.time_budget_minutes:
        findings.append(Finding("time.over_budget", "block", f"elapsed {run.elapsed_minutes:.1f}m > budget {contract.time_budget_minutes}m"))

    warn_count = sum(1 for f in findings if f.severity == "warn")
    over_budget = warn_count > contract.violation_budget

    return {"in_scope": in_scope, "off_scope": off_scope, "forbidden": forbidden,
            "findings": findings, "over_budget": over_budget,
            "passed": not over_budget and not any(f.severity == "block" for f in findings)}
```

### 第 4 步：实现多契约合并

```python
def merge_contracts(parent: ScopeContract, child: ScopeContract) -> ScopeContract:
    """最小权限合并：交集（允许）、并集（禁止）、最严格（时间）。"""
    return ScopeContract(
        task_id=child.task_id,
        goal=child.goal or parent.goal,
        allowed_files=sorted(set(parent.allowed_files) & set(child.allowed_files)),
        forbidden_files=sorted(set(parent.forbidden_files) | set(child.forbidden_files)),
        acceptance_criteria=list(dict.fromkeys(parent.acceptance_criteria + child.acceptance_criteria)),
        rollback_plan=child.rollback_plan or parent.rollback_plan,
        approvals_required=list(dict.fromkeys(parent.approvals_required + child.approvals_required)),
        time_budget_minutes=min_optional(parent.time_budget_minutes, child.time_budget_minutes),
        violation_budget=min(parent.violation_budget, child.violation_budget),
    )
```

### 第 5 步：运行演示

```python
# 项目级契约
project_wide = ScopeContract(
    task_id="P-PROJECT",
    goal="项目级默认值",
    allowed_files=["app.py", "test_app.py", "lib/**/*.py"],
    forbidden_files=["scripts/release.sh", "config/prod.yaml"],
    acceptance_criteria=[], rollback_plan="回滚并重新部署",
    time_budget_minutes=60, violation_budget=1,
)

# 任务级契约
task = ScopeContract(
    task_id="T-001",
    goal="为 /signup 添加输入验证",
    allowed_files=["app.py", "test_app.py"],
    forbidden_files=["migrations/**"],
    acceptance_criteria=["pytest -x test_app.py::test_signup_rejects_short_password"],
    rollback_plan="回滚提交并部署上一个构建标签",
    time_budget_minutes=30, violation_budget=0,
)

effective = merge_contracts(project_wide, task)
# 合并后：allowed_files = app.py, test_app.py（交集）
#          forbidden_files = scripts/release.sh, config/prod.yaml, migrations/**（并集）
#          time_budget = 30（最小值）
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Claude Code 的斜杠命令

`/scope` 命令写入范围契约并将其固定为会话上下文。子智能体在执行前读取契约。

### 4.2 GitHub PRs 中的范围检查

将契约作为 JSON 文件推送到 PR 正文或作为检入的构件。CI 对合并差异运行范围检查器。

### 4.3 LangGraph 的中断机制

范围违规触发 interrupt；处理器询问人工——是扩展契约还是智能体回退。

### 4.4 实践模式对照

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 违规预算 | 日常使用 | 轻微越界作为警告，超出预算才拒绝——否则门控会被禁用 |
| 路径族严重性不对称 | 生产环境 | `docs/**` 越界 = warn，`scripts/**`、`migrations/**` 越界 = block |
| 时间预算 | 长期任务 | `time_budget_minutes` 字段，超期需要重新批准 |
| 网络出口白名单 | 敏感环境 | `network_egress` 白名单，防止智能体静默调用外部 API |

---

## 5. 工程最佳实践

### 5.1 范围契约设计原则

| 原则 | 说明 |
|------|------|
| 否定空间是契约的一半 | 没有 `forbidden_files` 的契约不完整 |
| Glob 而非原始路径 | `app/**/*.py` 比 [`app.py`, `app/utils.py`, `app/models.py`] 更能抵抗重构 |
| 回滚计划是必需字段 | 无法回滚的契约不应该被批准 |
| 违规预算防止门控被禁用 | 轻微越界 = warn，超预算 = block——否则团队会关掉门控 |

### 5.2 中文场景特别建议

- **功能列表的 goal 字段用中文写**——方便非英文母语的团队成员理解
- **契约的 slug 和 task_id 用英文**——`T-001` 而不是 `任务-001`
- **glob 模式中的中文路径注意编码**——某些工具 glob 不支持中文字符，建议用英文路径或拼音

### 5.3 踩坑经验

- **没有违规预算的门控会被禁用**——如果每次越界都 block，团队在 deadline 时直接关掉。设置 `violation_budget: 2`，2 次以内 warn，超过才 block
- **只做文件范围不够**——智能体可以在修改范围内文件的同时，访问恶意网站或运行超长任务。加上 `time_budget_minutes` 和 `network_egress`
- **契约不与功能列表配合时，范围蔓延在任务层面发生**——一个任务完美遵守契约，但下个任务又开始碰不该碰的东西。`feature_list.json` 的"最多一个 in_progress"解决了这个问题

---

## 6. 常见错误

### 错误 1：契约中不写 forbidden_files

**现象：** 智能体修改了 `scripts/release.sh`，检查器没有报错。契约只列出了允许的文件，没有包含禁止的文件。

**原因：** 否定空间是契约的一半。没有 `forbidden_files`，智能体不知道哪些文件绝对不能碰。

**修复：**
```json
# ❌ 只写了允许的文件
{ "allowed_files": ["app.py", "test_app.py"] }

# ✓ 明确列出禁止的文件
{ "allowed_files": ["app.py", "test_app.py"],
  "forbidden_files": ["scripts/release.sh", "config/prod.yaml", "migrations/**"] }
```

### 错误 2：使用原始路径而非 glob

**现象：** 重构后 `test_app.py` 移到了 `tests/test_app.py`，契约失效。智能体可以随意修改新路径下的文件。

**原因：** 原始路径绑定到具体文件名，重构后契约不适用。

**修复：** 使用 glob 模式：`tests/**/*.py` 而非 `tests/test_app.py`。`app/**/*.py` 而非逐一路径枚举。

### 错误 3：没有违规预算

**现象：** 每次越界都 block。团队在 deadline 时直接关掉了门控检查。几周后没人记得曾经有过契约。

**原因：** 全有或全无的门控在压力下总是"全无"。

**修复：** 设置 `violation_budget: 2`。轻微越界第一次 warn，第二次 warn，第三次才 block。门控存活的关键是它有用但不烦人。

---

## 7. 面试考点

### Q1：什么是范围契约？它解决了什么问题？（难度：⭐）

**参考答案：**
范围契约是一个按任务编写的 JSON 文件，说清工作在哪里开始、在哪里结束、越界了怎么回滚。它解决的是智能体的**范围蔓延**问题——智能体每一步都在真诚地叙述，但所有修改合在一起构成了一个与当初审查完全不同的变更。

契约的核心字段是 `allowed_files`、`forbidden_files`、`acceptance_criteria` 和 `rollback_plan`。没有 `forbidden_files` 的契约是不完整的——否定空间占了契约的一半。

### Q2：为什么需要两个层次的范围约束？（难度：⭐⭐）

**参考答案：**
范围契约只约束一个任务。智能体可以完美遵守登录修复的契约，然后在下一轮决定项目还需要一个设置页面、一个深色模式开关——契约从未被问过"什么工作在项目范围内"。

第二个层次是 `feature_list.json`，一个机器可读的项目待办事项。功能列表的"最多一个 `in_progress`"不变式确保智能体一次只能做一个功能。契约和功能列表通过最小权限组合：契约的 `allowed_files` 必须落在功能列表当前功能的范围内。

### Q3：违规预算是做什么的？为什么它比二进制门控更好？（难度：⭐⭐）

**参考答案：**
违规预算允许一定次数的轻微越界作为警告（warn），超出预算才拒绝（block）。这解决了二进制门控（全有或全无）的根本问题：团队在 deadline 时总是关掉全有的门控。

`agent-guardrails` 的实现设置了 `violationBudget` 参数。轻微范围越界在预算内只做警告；只有超出预算时合并门控才拒绝。预算金额是门控存活——即既能交付又不会被禁用——的关键。

### Q4：多契约合并的最小权限语义是什么？（难度：⭐⭐⭐）

**参考答案：**
当项目级契约和任务级契约同时适用时：
- `allowed_files`：**交集**（两个契约都必须允许该路径）
- `forbidden_files`：**并集**（任意一个契约禁止即禁止）
- `time_budget_minutes`：**最小值**（最严格的生效）
- `approvals_required`：**累积**
- `network_egress`：`None` 延后到对方，两个列表取交集，空列表拒绝所有

这确保了合并后的契约不会比任何一个单独契约更宽松——最小权限的核心原则。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 范围契约 (Scope Contract) | "任务简报" | 任务级 JSON，列出允许/禁止的文件、验收条件、回滚计划 |
| 范围蔓延 (Scope Creep) | "它又碰了别的文件" | 同一个任务中修改了契约之外的文件 |
| 回滚计划 (Rollback Plan) | "可以回退" | 操作员执行的一段回滚步骤 |
| 违规预算 (Violation Budget) | "可以容忍几次" | 轻微越界在预算内只警告，超出预算才拒绝 |
| 差异检查 (Diff Check) | "路径审计" | 对比修改过的文件与契约 glob 模式 |

---

## 📚 小结

"别跑偏"是愿望，不是指令。范围契约把愿望变成了可检查的约束——每个任务启动时写入 JSON，列出允许的文件、禁止的文件、验收条件和回滚计划。你实现了一个范围检查器，理解了 glob 模式为什么比原始路径更好、违规预算为什么比二进制门控更实用，以及多契约合并的最小权限语义。

下一课我们解决另一个智能体循环中的关键问题：智能体在"看过真实输出"和"猜测输出"之间差距巨大——运行时反馈循环让你相信的不是智能体的预测，而是它实际运行的结果。

---

## ✏️ 练习

1. 【实现】在契约中添加 `network_egress` 字段列出允许的外部主机。拒绝访问其他主机的运行。

2. 【实现】扩展检查器：对 `docs/**` 越界使用 warn，对 `scripts/**` 越界使用 block。论证这种不对称的合理性。

3. 【实现】让契约从 `goal` 字段使用静态规则（不用 LLM）推导出 `allowed_files`。在第一个边界案例上会出什么问题？

4. 【实现】添加 `time_budget_minutes`，实际时间超出预算时拒绝继续运行。

5. 【思考】两个契约同时作用于同一个差异时，正确的合并语义是什么？如果用"第一个契约优先"会有什么问题？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 范围契约检查器 | `code/main.py` | glob 匹配、违规检查、多契约合并、归档 |
| 技能提示词 | `outputs/skill-scope-contract.md` | 为任务描述生成范围契约和检查器 |

---

## 📖 参考资料

1. [官方文档] LangGraph Human-in-the-Loop Interrupts: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
2. [官方文档] OpenAI Agents SDK Tool Approval Policies: https://platform.openai.com/docs/guides/agents-sdk
3. [GitHub] logi-cmd/agent-guardrails: https://github.com/logi-cmd/agent-guardrails — 合并门控和范围验证、违规预算、严重性级别
4. [博客] Augment Code. "AI Spec Template". https://www.augmentcode.com/guides/ai-spec-template — 三层边界系统（must/ask/never）
5. [博客] Dev|Journal. "Preventing AI Agent Configuration Drift with Agent Contract Testing". https://earezki.com/ai-news/2026-05-05-i-built-a-tiny-ci-tool-to-keep-ai-agent-configs-from-drifting-in-my-repo/ — `--strict` 模式无外部依赖

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
