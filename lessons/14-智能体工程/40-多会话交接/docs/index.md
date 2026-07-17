# 多会话交接——让下一个会话从停下的地方继续

> 会话会结束。工作不会。交接包是将"智能体工作了一小时"变成"下一个会话第一分钟就高效"的工件。有目的地构建它，而不是作为事后思考。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 34（仓库记忆）、阶段 14 · 38（验证门控）、阶段 14 · 39（审查员）
**预计时间：** ~50 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 32（最小工作台）— 三个文件的起点；阶段 14 · 34（持久状态）— 状态文件的基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别每个交接包需要的七个字段
- [ ] 从工作台工件自动生成交接包，无需手写散文
- [ ] 将大型反馈日志修剪为适合交接的大小
- [ ] 让下一个会话的第一步变得确定性
- [ ] 理解会话结束前清理工作台的重要性

---

## 1. 问题

会话结束了。智能体说"很好，我们取得了进展"。下一个会话打开。下一个智能体问"我们停在哪了？"前一个智能体的答案已经消失。下一个智能体重新发现、重新运行相同的命令、重新问相同的问题——花了三十分钟恢复上一个会话最后三十秒的状态。

坏交接的成本在任务的整个生命周期中——每个会话都在支付。修复方案是在会话结束时自动生成一个交接包：改了啥、为什么、尝试过什么、什么失败了、还差什么、下一步先做什么。

---

## 2. 概念

### 2.1 交接包的七个字段

```
状态文件 → generate_handoff.py → handoff.md（给人读）+ handoff.json（给智能体读）
验证报告 ↗                                     ↓
审查报告 ↗                                 下一个会话启动
反馈记录 ↗
```

| 字段 | 回答的问题 |
|------|----------|
| `summary` | 一段话总结做了什么 |
| `changed_files` | 差异概览 |
| `commands_run` | 实际执行了哪些命令 |
| `failed_attempts` | 尝试过什么以及为什么不工作 |
| `open_risks` | 可能影响下一个会话的风险以及严重性 |
| `next_action` | 下一个会话的第一步具体做什么 |
| `verdict_pointer` | 验证 + 审查报告的路径 |

**`next_action` 是承载性字段。** 一个除了 `next_action` 之外什么都有交接包是状态报告，不是交接包。

### 2.2 交接是生成的，不是手写的

手写交接是在累的时候会被跳过的交接。生成器读取工作台工件并发出包。智能体的任务是让工作台处于生成器可以总结的状态，而不是写总结本身。

### 2.3 两种形式：人类可读和机器可读

`handoff.md` 给人读。`handoff.json` 给下一个智能体加载。两者来自同样的源工件。如果它们不一致，JSON 胜出。

### 2.4 反馈日志修剪

完整的 `feedback_record.jsonl` 可能有数百条记录。交接只携带最后 K 条加上所有非零退出码的记录。下一个会话如果需要可以加载完整日志，但包保持轻量。

### 2.5 离开干净的工作台

交接描述工作。干净的工作台让工作可恢复。两者不是一回事。一个完美的 `handoff.md` 如果下一个会话打开的是半应用的差异、一个智能体忘了的临时文件、一个漂移的分支和运行前就报错的测试，那完全没有价值。

| 检查项 | 干净意味着 | 脏的代价 |
|-------|-----------|---------|
| 工作树 | 每个变更已提交或有注释的 stash | 半应用差异看起来像有意的修改 |
| 临时文件 | 没有 `*.tmp`、临时目录、调试打印、注释块 | 残留文件污染下一个会话的心智模型 |
| 测试 | 绿色，或红色且在 `open_risks` 中标注了失败原因 | 无声的红色测试是陷阱 |
| 任务板 | `feature_list.json` 状态反映现实 | 过时任务板引向已完成的工作 |
| 分支 | 在预期分支上，不是 detached HEAD | 错误分支意味着下一个提交落在错误的地方 |

清理阶段在生成交接包之前运行。它发出 `clean_state.json`，阻塞项列表为空时才是生成器开始写包的前提条件。

---

## 3. 从零实现

### 第 1 步：定义数据结构和修剪函数

```python
from dataclasses import dataclass, field

TAIL_K = 5

@dataclass
class WorkbenchSnapshot:
    task_id: str
    state: dict
    verdict: dict
    review: dict
    feedback: list[dict]
    diff_summary: dict

@dataclass
class HandoffPayload:
    task_id: str
    summary: str
    changed_files: list[str]
    commands_run: list[str]
    failed_attempts: list[str]
    open_risks: list[dict]
    next_action: str
    verdict_pointer: dict
    feedback_tail: list[dict] = field(default_factory=list)

def trim_feedback(records: list[dict]) -> list[dict]:
    """只保留最后 K 条 + 所有非零退出码的记录。"""
    tail = records[-TAIL_K:]
    nonzero = [r for r in records if r.get("exit_code") not in (0, None)]
    seen = set()
    result = []
    for r in tail + nonzero:
        key = id(r)
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result
```

### 第 2 步：推导风险

```python
def derive_risks(snapshot: WorkbenchSnapshot) -> list[dict]:
    """从验证发现、状态阻塞项、审查总分推导风险。"""
    risks = []
    for f in snapshot.verdict.get("findings", []) or []:
        if isinstance(f, dict) and f.get("severity") in ("warn", "block"):
            risks.append({"severity": str(f.get("severity")),
                          "detail": str(f.get("detail"))})
    for blocker in snapshot.state.get("blockers") or []:
        risks.append({"severity": "warn", "detail": f"open blocker: {blocker}"})
    raw_total = snapshot.review.get("total", 10)
    try:
        safe_total = int(raw_total)
    except (TypeError, ValueError):
        safe_total = 10
    if safe_total < 7:
        risks.append({"severity": "warn",
                      "detail": f"review total {raw_total} below 7"})
    return risks
```

### 第 3 步：生成交接包

```python
def generate_handoff(snapshot: WorkbenchSnapshot) -> tuple[str, HandoffPayload]:
    """生成 Markdown 和 JSON 两种格式的交接包。"""
    next_action = str(snapshot.state.get("next_action") or "no next_action recorded; needs human")
    payload = HandoffPayload(
        task_id=snapshot.task_id,
        summary=f"task {snapshot.task_id}: review={snapshot.review.get('verdict')}, gate={snapshot.verdict.get('passed')}",
        changed_files=snapshot.diff_summary.get("touched", []),
        commands_run=[str(r.get("command")) for r in snapshot.feedback],
        failed_attempts=[
            f"{r.get('command')} -> exit {r.get('exit_code')}"
            for r in snapshot.feedback if r.get("exit_code") not in (0, None)
        ],
        open_risks=derive_risks(snapshot),
        next_action=next_action,
        verdict_pointer={
            "verdict": f"outputs/verification/{snapshot.task_id}.json",
            "review": f"outputs/review/{snapshot.task_id}.json",
        },
        feedback_tail=trim_feedback(snapshot.feedback),
    )

    # 生成 Markdown
    md = f"""# Handoff: {payload.task_id}

**Summary.** {payload.summary}

## Changed files
{chr(10).join(f'- `{{f}}`' for f in payload.changed_files) or '- none'}

## Commands run
{chr(10).join(f'- `{{c}}`' for c in payload.commands_run) or '- none'}

## Failed attempts
{chr(10).join(f'- {{f}}' for f in payload.failed_attempts) or '- none'}

## Open risks
{chr(10).join(f'- [{{r['severity']}}] {{r['detail']}}' for r in payload.open_risks) or '- none'}

## Next action
{payload.next_action}

## Receipts
- verdict: `{payload.verdict_pointer['verdict']}`
- review:  `{payload.verdict_pointer['review']}`
"""
    return md, payload
```

### 第 4 步：运行演示

```python
def main():
    snapshot = WorkbenchSnapshot(
        task_id="T-001",
        state={"active_task_id": None,
               "blockers": ["awaiting decision on rate-limit window"],
               "next_action": "open PR with current diff and request review"},
        verdict={"passed": True, "findings": [{"severity": "warn", "detail": "off-scope: README.md"}]},
        review={"verdict": "pass", "total": 8},
        feedback=[{"command": "pytest", "exit_code": 0},
                  {"command": "ruff check .", "exit_code": 0},
                  {"command": "pytest test_signup.py", "exit_code": 1},
                  {"command": "pytest test_signup.py", "exit_code": 0}],
        diff_summary={"touched": ["app/signup.py", "tests/test_signup.py", "README.md"]})

    md, payload = generate_handoff(snapshot)
    print(md)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 现有工具的压缩策略

| 产品 | 压缩策略 | 交接方式 |
|------|---------|---------|
| Codex CLI | POST /v1/responses/compact（AES 加密 blob）+ 本地 handoff summary | 隐式（_summary 消息） |
| Claude Code | 五阶段渐进式压缩，95% 上下文时触发 | 显式（交接包） |
| OpenCode | 基于时间戳的消息隐藏 + LLM 总结 | 隐式 |

**压缩扩展会话；交接关闭一个会话并干净地启动下一个。** Hermes Issue #20372（2026 年 4 月）明确了这个区分：当就地压缩开始退化时，智能体应该写一个紧凑的交接包、结束会话、在干净的上下文中恢复。

### 4.2 生产模式

| 模式 | 说明 |
|------|------|
| **压缩 vs 交接** | 压缩扩展现有会话；交接关闭一个、启动下一个。压缩退化时应使用交接 |
| **每个分支一个活跃交接** | 总是包含 `branch`、`last_known_good_commit`、`status`（active/superseded/archived） |
| **在 50-75% 上下文预算时结束，不要撞墙** | CLAUDE.md + HANDOVER.md 模式报告：在 50-75% 上下文预算时结束效果最好 |

### 4.3 跨产品交接

交接包可以用作跨产品协作的通用语言——用 Claude Code 构建，用 Codex 继续。

---

## 5. 工程最佳实践

### 5.1 交接包设计原则

| 原则 | 说明 |
|------|------|
| 生成而非手写 | 手写交接在累的时候会被跳过。生成器读取工件自动生成 |
| `next_action` 是承载性字段 | 没有 `next_action` 的是状态报告，不是交接包 |
| 离开干净的工作台 | 清理阶段在交接之前运行。脏状态 → 不生成交接 |
| 在 50-75% 上下文时结束 | 压缩退化前主动交接，不要在撞墙时被迫交接 |

### 5.2 中文场景特别建议

- **交接包的 summary 用中文写**——方便中文团队阅读。但字段名保持英文（`next_action`、`changed_files`）
- **`handoff.md` 给人看用中文，`handoff.json` 给智能体用英文**——保持 JSON 字段名跨工具兼容
- **中文文件名的差异在交接包中注意编码**——`handoff.md` 中的中文文件名确保使用正确编码

### 5.3 踩坑经验

- **交接时没有清理工作台**——智能体结束会话时留下一个半应用的 diff、一个临时文件、一个漂移的分支。下一个会话花了 10 分钟清理而不是继续构建。**修复：** 交接前运行清理检查
- **`next_action` 缺失**——交接包写得很详细，但没有说下一步做什么。下一个智能体不知道从哪里开始。**修复：** `next_action` 是必填字段
- **压缩 vs 交接混为一谈**——不断压缩直到上下文质量崩溃，而不是主动结束并交接。**修复：** 在 50-75% 上下文预算时主动交接

---

## 6. 常见错误

### 错误 1：手写交接而非生成

**现象：** 交接包是智能体临会话结束时手写的一段话。写得好的时候很有用。但累的时候智能体不写，或者写得很潦草。"今天修复了一个 bug，还有点状态问题"——没有文件列表、没有风险分析、没有下一步。

**原因：** 手写交接没有强制性。它依赖智能体在会话结束时"做好事"的意愿。

**修复：** 交接包从工件自动生成。智能体的任务不是写总结，而是让工作台处于生成器可以总结的状态。

### 错误 2：没有清理就交接

**现象：** 交接包写得很完美，但工作台是脏的。一个未提交的修改、一个临时文件、一个分离的头指针。下一个智能体从错误的状态开始。

**原因：** 交接描述工作和工作台状态不是一回事。交接可以描述"完成了什么"，但工作台可能处于"不可恢复"的状态。

**修复：** 交接前运行清理检查（clean_state.json）。任何阻塞项都阻止交接生成。

### 错误 3：把压缩当交接用

**现象：** 不断压缩上下文直到 98% 预算。智能体开始迷失、忘记目标、重复工作。

**原因：** 压缩扩展现有会话，交接关闭一个并干净地启动下一个。在压缩开始退化时应该主动交接，而不是继续压缩。

**修复：** 在 50-75% 上下文预算时主动结束会话、生成交接包、启动新会话。

---

## 7. 面试考点

### Q1：交接包的七个字段是什么？哪个最关键？（难度：⭐）

**参考答案：**
summary、changed_files、commands_run、failed_attempts、open_risks、next_action、verdict_pointer。

**`next_action` 最关键。** 一个除了 `next_action` 之外什么都有交接包是状态报告，不是交接包。状态报告告诉"我们做了什么"，交接包还告诉"下一步做什么"。下一个会话需要 `next_action` 才能在第一分钟就高效。

### Q2：为什么交接应该生成而不是手写？（难度：⭐⭐）

**参考答案：**
手写交接在累的时候会被跳过。智能体会话结束时可能已经耗尽了上下文预算和注意力——让它手写交接的质量不可靠。

生成器从工作台工件（状态文件、验证报告、审查报告、反馈记录）自动生成包。智能体的任务不是写总结，而是让工作台处于生成器可以总结的状态。**交接是架构问题，不是文档问题。**

### Q3：压缩和交接有什么区别？什么时候应该用交接代替压缩？（难度：⭐⭐⭐）

**参考答案：**
**压缩（Compaction）** 扩展现有会话——在同一个上下文中压缩历史以腾出空间。**交接（Handoff）** 关闭一个会话并干净地启动下一个。

Hermes Issue #20372 区分：当就地压缩开始退化时（模型开始忘记目标、重复工作），智能体应该写一个紧凑的交接包、结束会话、在干净的上下文中恢复。CLAUDE.md + HANDOVER.md 模式报告在 50-75% 上下文预算时主动结束效果最好。

**关键区别：** 压缩在质量开始下降时已经太晚了。交接在质量下降前主动结束。

### Q4：为什么离开干净的工作台和交接同样重要？（难度：⭐⭐）

**参考答案：**
交接包描述工作。干净的工作台让工作可恢复。两者不是一回事。

一个完美的 `handoff.md` 如果下一个会话打开的是半应用的 diff、一个残留的临时文件、一个漂移的分支和运行前就报错的测试，那完全没有价值。下一个会话的前十分钟花在清理上一个会话的混乱上。

修复：交接前运行清理检查（clean_state.json）。阻塞项列表为空时生成器才开始写交接包。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 交接包 (Handoff Packet) | "会话总结" | 从工件生成的包——Markdown 给人、JSON 给智能体 |
| 下一步 (Next Action) | "先做什么" | 启动下一个会话的具体步骤 |
| 反馈修剪 (Feedback Trim) | "日志摘要" | 最后 K 条 + 所有非零退出码 |
| 状态报告 (Status Report) | "做了什么" | 缺少 `next_action` 的文档——有用但不是交接 |
| 清理检查 (Clean State) | "工作台就绪" | 写交接前验证工作台状态——脏状态不生成 |

---

## 📚 小结

交接包将"智能体工作了一小时"变成"下一个会话第一分钟就高效"。七个字段中 `next_action` 最关键。交接是生成的，不是手写的。压缩扩展会话，交接关闭一个并干净地启动下一个。在 50-75% 上下文预算时主动交接，撞墙时已经太晚了。交接前运行清理检查——脏状态不能生成交接。

至此第 14 章（智能体工程）的全部核心内容完成：从工作台的七个面到规则、状态、范围、反馈、验证、审查和交接——你构建了一个完整的智能体工作台。

---

## ✏️ 练习

1. **【实现】** 添加 `assumptions_to_validate` 字段——列出所有构建者记录但审查员评分不超过 1 的假设。

2. **【实现】** 对失败的运行和通过的运行使用不同的反馈摘要策略。论证这种不对称。

3. **【实现】** 包含"给人类的问题"列表。什么问题应该进入包，什么应该直接聊天询问？

4. **【实现】** 使生成器幂等：运行两次产生相同的包。什么需要稳定才能做到？

5. **【思考】** 添加"下一个会话前置条件"部分——列出下一个会话在执行前必须加载的工件。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 交接包生成器 | `code/main.py` | 七字段交接包生成——Markdown + JSON |
| 技能提示词 | `outputs/skill-handoff-generator.md` | 项目特定的交接包生成器和会话结束钩子 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Effective Harnesses for Long-Running Agents". https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
2. [官方文档] OpenAI Agents SDK Handoffs: https://platform.openai.com/docs/guides/agents-sdk/handoffs
3. [博客] Codex Blog. "Codex CLI Context Compaction". https://codex.danielvaughan.com/2026/03/31/codex-cli-context-compaction-architecture/ — POST /v1/responses/compact 和本地回退
4. [博客] Justin3go. "Shedding Heavy Memories: Context Compaction in Codex, Claude Code, OpenCode". https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode — 三家压缩对比
5. [博客] JD Hodges. "Claude Handoff Prompt: How to Keep Context Across Sessions (2026)". https://www.jdhodges.com/blog/ai-session-handoffs-keep-context-across-conversations/ — CLAUDE.md + HANDOVER.md，50-75% 上下文预算

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
