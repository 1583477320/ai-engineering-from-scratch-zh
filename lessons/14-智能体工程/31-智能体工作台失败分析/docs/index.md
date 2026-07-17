# 智能体工作台工程：为什么强大模型仍然失败

> 强大的模型是不够的。可靠的智能体需要一个工作台：指令、状态、范围、反馈、验证、审查和交接。剥离这些，即使是前沿模型也会产生不安全的产品。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 01（智能体循环）、阶段 14 · 26（故障模式）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 29（生产运行时）— 工作台原语在生产环境中的部署形态

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分模型能力与执行可靠性——知道什么时候是模型问题，什么时候是工作台问题
- [ ] 命名工作台的七个面（surfaces）以及每个缺失时的故障模式
- [ ] 在小型仓库任务上比较纯提示词运行和工作台引导运行的结果
- [ ] 将行业流行模式翻译回分布式系统原语——函数、工作者、触发器、运行时、队列、持久化、策略

---

## 1. 问题

你把一个前沿模型放进真实仓库，让它添加输入验证。它打开四个文件，写出看似合理的代码，声明成功，然后停止。你运行测试——两个失败。还有第三个文件被修改了，和验证完全没有关系。没有记录智能体做了什么假设、先尝试了什么、还有什么没做。

不是模型的 Python 写错了。是模型**对工作理解错了**。它不知道什么算完成、哪些文件可以写、哪个测试是权威的、下一个会话应该从哪里继续。

这不是模型 bug。**这是工作台 bug。** 智能体周围的面（surfaces）缺少了将一次性生成转变成可靠、可恢复的工程工作的部分。

---

## 2. 概念

### 2.1 工作台的七个面

```
任务 → 范围契约 → 仓库记忆 → 智能体循环 → 运行时反馈 → 验证门控 → 审查员 → 交接
                                                                              ↓
                                                                        仓库记忆（闭环）
```

| 面 (Surface) | 承载什么 | 缺失时的故障模式 |
|-------------|---------|----------------|
| **指令** | 启动规则、禁止操作、完成定义 | 智能体猜测"完成"是什么意思 |
| **状态** | 当前任务、修改的文件、阻塞项、下一步 | 每次会话从零开始 |
| **范围** | 允许的文件、禁止的文件、验收条件 | 修改泄漏到无关代码中 |
| **反馈** | 真实的命令输出 | 智能体在 400 错误上声明成功 |
| **验证** | 测试、lint、冒烟测试、范围检查 | "看起来不错"直达 main |
| **审查** | 不同角色的第二轮检查 | 构建者给自己的作业打分 |
| **交接** | 改了啥、为什么、还差什么 | 下一个会话重新发现一切 |

工作台与模型无关。你可以换模型，保留这些面。你不能换掉这些面保留可靠性。

### 2.2 工作台 vs 提示词工程

提示词告诉模型**本轮**你想要什么。工作台告诉模型**跨轮次、跨会话**怎么做工作。大多数智能体失败故事穿着提示词工程的衣服，实际上是工作台失败。

```
提示词工程："请在 /signup 中添加输入验证"
工作台工程：   范围契约（能写哪些文件）→ 状态文件（读到哪了）→ 
              反馈运行器（捕获命令输出）→ 验证门控（检查测试是否通过）
```

### 2.3 工作台 vs 框架

框架给你运行时（LangGraph、AutoGen、Agents SDK）。工作台给智能体在那个运行时中**工作的地方**。你需要两者。本章后半部分是关于第二件事的。

### 2.4 从原语出发，不从供应商分类出发

去掉"智能体"标签一会儿。一次智能体运行就是一次跨越时间、进程和机器的计算。要让它可靠，你需要任何生产系统都需要的那组原语。

| 原语 | 是什么 | 对智能体的含义 |
|------|-------|--------------|
| **函数 (Function)** | 类型化处理器。纯函数优先。拥有自己的输入和输出 | 一次工具调用、一次规则检查、一次验证步骤、一次模型调用 |
| **工作者 (Worker)** | 拥有一个或多个函数和生命周期的大型进程 | 构建者、审查员、验证器、MCP 服务器 |
| **触发器 (Trigger)** | 调用函数的事件源 | 智能体循环 tick、HTTP 请求、队列消息、定时任务、文件变更、钩子 |
| **运行时 (Runtime)** | 决定什么在哪里运行、超时和资源限制的边界 | Claude Code 的进程、LangGraph 的运行时、工作容器 |
| **HTTP / RPC** | 调用者和工作者之间的线路 | 工具调用协议、MCP 请求、模型 API |
| **队列 (Queue)** | 触发器和工作者之间的持久缓冲区；背压、重试、幂等性 | 任务板、反馈日志、审查收件箱 |
| **会话持久化 (Session Persistence)** | 在崩溃、重启、模型替换后存活的状态 | `agent_state.json`、检查点、KV 存储、仓库本身 |
| **授权策略 (Authorization Policy)** | 谁可以调用什么函数、在什么范围内 | 允许/禁止的文件、审批边界、MCP 能力列表 |

现在把七个工作台面映射到这些原语上：

- **指令** = 策略 + 函数元数据。规则是检查（函数）。`AGENTS.md` 是附加到运行时启动的策略
- **状态** = 会话持久化。运行时每步读取的键值存储。文件、KV 或 DB；持久化语义重要，存储后端不重要
- **范围** = 每个任务的授权策略。允许/禁止的 glob 是 ACL。审批要求是权限格
- **反馈** = 写入队列的调用日志。每次 shell 调用是一条记录，持久、可重放
- **验证** = 一个函数。输入确定性。任务关闭时触发。失败即关闭
- **审查** = 独立的工作者，对构建者工件只读，对审查报告只写
- **交接** = 会话结束触发器发出的持久记录。下一个会话的启动触发器读取

### 2.5 行业模式翻译成原语

每个流行的"工作台模式"都可以用八个原语来描述：

| 供应商/社区模式 | 它实际上是什么 |
|----------------|--------------|
| Ralph Loop（Claude Code、Codex）——智能体提前停止时，将原始意图重新注入新上下文窗口 | 一个触发器，用干净的上下文重新入队任务；会话持久化将目标带过去 |
| 计划/执行/验证（PEV） | 三个工作者，每种角色一个，通过状态和队列通信 |
| Harness-compute 分离（OpenAI Agents SDK，2026 年 4 月）——将控制面与执行面分开 | 重新陈述控制面/数据面。比"智能体"这个标签早几十年 |
| Open Agent Passport（OAP，2026 年 3 月）——在执行前对每个工具调用签名和审计 | 一个由执行前工作者强制执行的授权策略，带签名审计队列 |
| Guide 和 Sensor（Birgitta Böckeler / Thoughtworks）——前馈规则 + 反馈可观测性 | 授权策略 + 验证函数 + 可观测性追踪 |
| 渐进式压缩（Claude Code 逆向工程，2026 年 4 月） | 一个状态管理工作器，像定时任务一样在会话持久化上运行，保持其在预算内 |
| 钩子/中间件（LangChain、Claude Code）——拦截模型和工具调用 | 触发器 + 函数，包裹在运行时的调用路径周围 |
| 技能作为 Markdown 渐进式披露（Anthropic、Flue） | 一个函数注册表，函数的元数据在需要时才加载到上下文中 |
| MCP 服务器 | 工作者通过稳定的 RPC 暴露函数，能力列表作为授权 |

上表中的每一项都是智能体社区在重新发现一个在分布式系统中早已有名字的原语，然后给它起了一个新名字。用于营销时是有用的标签；作为工程词汇就没那么有用了。

### 2.6 数据证据

工作台 > 模型 这一说法现在有数字支撑了：

- **Terminal Bench 2.0**——同一个模型，改变工作台后，编码智能体从 30 名开外上升到第 5 名（LangChain, *Anatomy of an Agent Harness*）
- **Vercel**——删除了 80% 的智能体工具，成功率从 80% 跳到 100%（MongoDB）
- **Harvey**——仅通过工作台优化，法律智能体准确率翻倍（MongoDB）
- **88% 的企业 AI 智能体项目未能投产**——失败集中在运行时，而非推理（preprints.org, 2026 年 3 月）
- **2025 年基准研究**——三个流行开源框架的 ~50% 任务完成率；WebAgent 在长上下文条件下从 40-50% 崩溃到 10% 以下，主要来自无限循环和目标丢失

关键不是"工作台永远胜过模型"。模型确实会随着时间的推移吸收工作台的技巧。关键是：**在今天，起承载作用的工程在模型周围，而不是模型内部。** 承载这些工作的原语正是每个生产系统一直需要的那些。

### 2.7 供应商文章止步的地方

这是不需要客气的部分：

- **LangChain 的 *Anatomy of an Agent Harness***——列举了 11 个组件（提示词、工具、钩子、沙箱、编排、记忆、技能、子智能体、运行时循环），但没有命名队列、工作者作为部署单元、触发器语义、会话持久化作为一个独立关注点、授权策略。它把工作台当作一个可以配置的对象，而不是一个需要部署的系统
- **Addy Osmani 的 *Agent Harness Engineering***——提出了 `Agent = Model + Harness` 框架和棘轮模式，但没有说明工作台是由什么构建的。它读起来像立场声明，不是规格说明
- **Anthropic 和 OpenAI**——对"面"的讨论最深入，但停留在各自的运行时范围内。2026 年 4 月 Agents SDK 的 "harness-compute 分离"公告是第一个明确支持控制面/数据面分离的供应商文章。这是一个原语思想，不是一个新思想
- **agentic_harness 一书**——把工作台当作配置对象（Jaymin West 的 *Agentic Engineering* 第 6 章），最强的一句话是"工作台是智能体系统中的主要安全边界"，这不过是在重新陈述授权策略

你不需要不同意这些文章就能注意到这个差距。他们写的是**已经存在的系统的 UX 描述**。我们在写的是**系统本身**。系统构建正确时，七个面从原语中自然产生。系统构建错误时，再多的 `AGENTS.md` 润色也修复不了缺失的队列。

---

## 3. 从零实现

### 第 1 步：定义仓库任务和工作台面

```python
from dataclasses import dataclass, field

WORKBENCH_SURFACES = [
    "instructions", "state", "scope", "feedback",
    "verification", "review", "handoff",
]

@dataclass
class RepoTask:
    description: str
    allowed_files: list[str]
    forbidden_files: list[str]
    acceptance: list[str]

@dataclass
class RunResult:
    label: str
    surfaces_present: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tests_run: bool = False
    declared_success: bool = False
    actually_passing: bool = False
    notes: list[str] = field(default_factory=list)

    def missing_surfaces(self) -> list[str]:
        return [s for s in WORKBENCH_SURFACES if s not in self.surfaces_present]
```

### 第 2 步：实现存根智能体对比

```python
def stub_agent(task: RepoTask, surfaces: list[str]) -> RunResult:
    """确定性存根智能体——模拟两种运行模式。"""
    result = RunResult(label="prompt-only" if not surfaces else "workbench")
    result.surfaces_present = list(surfaces)

    has_scope = "scope" in surfaces
    has_feedback = "feedback" in surfaces
    has_verification = "verification" in surfaces
    has_state = "state" in surfaces

    # 范围检查
    if has_scope:
        result.files_touched = [f for f in task.allowed_files]
    else:
        result.files_touched = [*task.allowed_files, "README.md", "scripts/release.sh"]
        result.notes.append("touched unrelated files because scope was missing")

    # 反馈捕获
    if has_feedback:
        result.tests_run = True
        result.notes.append("captured stdout/stderr/exit code from the test run")
    else:
        result.notes.append("never ran the test command, guessed at output")

    # 验证门控
    if has_verification:
        result.actually_passing = True
        result.declared_success = True
        result.notes.append("verification gate proved acceptance criteria met")
    else:
        result.declared_success = True
        result.actually_passing = False
        result.notes.append("declared success without running acceptance checks")

    # 状态持久化
    if not has_state:
        result.notes.append("no state file written, next session restarts from zero")

    return result
```

### 第 3 步：生成失败报告

```python
def failure_report(result: RunResult) -> dict:
    return {
        "label": result.label,
        "missing_surfaces": result.missing_surfaces(),
        "off_scope_writes": [f for f in result.files_touched
                             if f not in {"app.py", "test_app.py"}],
        "tests_run": result.tests_run,
        "declared_success": result.declared_success,
        "actually_passing": result.actually_passing,
        "notes": result.notes,
    }
```

### 第 4 步：运行对比

```python
def main():
    task = RepoTask(
        description="add input validation to /signup and a passing test",
        allowed_files=["app.py", "test_app.py"],
        forbidden_files=["README.md", "scripts/release.sh"],
        acceptance=["test_app.py::test_signup_rejects_short_password passes"],
    )

    prompt_only = stub_agent(task, surfaces=[])
    workbench = stub_agent(task, surfaces=WORKBENCH_SURFACES)

    print("=== prompt only ===")
    for k, v in failure_report(prompt_only).items():
        print(f"  {k}: {v}")

    print("\n=== workbench ===")
    for k, v in failure_report(workbench).items():
        print(f"  {k}: {v}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 现有产品的七个面

| 产品 | 指令 | 状态 | 范围 | 反馈 | 验证 | 审查 | 交接 |
|------|------|------|------|------|------|------|------|
| Claude Code | AGENTS.md | Session Store | 斜杠命令 | Bash 工具 | 钩子 | CODEOWNERS | 交接包 |
| Codex/Cursor | CLAUDE.md | 工作区规则 | 范围配置 | 输出捕获 | CI | PR 审查 | 会话摘要 |
| LangGraph | 系统提示词 | 检查点 | Interrupts | 节点输出 | 验证节点 | — | 状态快照 |

### 4.2 工作台审计技能

`outputs/skill-workbench-audit.md` 是一个可移植的技能，它审计现有仓库的七个工作台面，报告哪些缺失、哪些部分、哪些健康。放在任何智能体设置旁边；它告诉你先要修复什么。

---

## 5. 工程最佳实践

### 5.1 工作台设计原则

| 原则 | 说明 |
|------|------|
| 工作台不是提示词 | 提示词告诉模型本轮做什么；工作台告诉模型跨轮次、跨会话怎么做 |
| 模型可换，工作台不可换 | 你可以换模型并保留工作台。你不能换掉工作台并保留可靠性 |
| 从原语出发，不从供应商分类出发 | 函数、工作者、触发器、运行时、队列、持久化、策略——这些比任何供应商框架更持久 |
| 七个面必须全部存在 | 缺少任何一个，工作台就有洞 |

### 5.2 中文场景特别建议

- **中文团队的工作台文档用中文写**——`AGENTS.md`、`agent-rules.md` 的 description 用中文，让团队能审查和修改
- **原语思维对中文开发者特别重要**——中文智能体生态系统以英文框架为主，理解背后的原语让你不被特定框架绑定
- **用"面"（surface）而不是"层"来思考**——七个面是并行的，不是分层的。每个面都在每个轮次中起作用

### 5.3 踩坑经验

- **把提示词工程当工作台工程**——在提示词中写 2000 行指令，但没有任何状态、范围、反馈机制。模型读完前 50 行就没注意力了，后面的从不执行
- **供应商锁定**——深度绑定 LangGraph 的检查点机制，迁移到 OpenAI SDK 时需要重写整个状态管理。**修复：** 坚持原语——状态就是键值存储，不管用什么框架
- **混淆演示与生产**——演示时选择最顺利的路径走，生产中遇到所有边缘情况。工作台的七个面在演示中显得"多余"，但在生产中每条都救过场

---

## 6. 常见错误

### 错误 1：把 AGENTS.md 写成百科全书

**现象：** AGENTS.md 3000 行，包含完整的历史决策记录、团队规范、架构说明。智能体加载后读了前 50 行，耗尽了注意力预算，忽略了关键规则。

**原因：** 分不清路由和参考文档。AGENTS.md 应该是指南针（指向去哪找信息），不是百科全书（包含所有信息）。

**修复：** AGENTS.md 不超过 50 行。详细规则放在 `docs/agent-rules.md`，架构放在 `docs/architecture.md`，仅当任务涉及时才加载。

### 错误 2：没有反馈回路

**现象：** 智能体说"正在运行测试"，下一条消息说"所有测试通过"。但测试从未真正运行过——智能体想象了输出。更糟的是，基于不存在的测试结果做了后续决策。

**原因：** 缺少反馈面。智能体没有捕获命令输出的机制，只能靠自己"猜"。

**修复：** 所有命令通过 `run_with_feedback()` 封装，输出、退出码、持续时间被结构化捕获。退出码为 `null` 时拒绝推进循环。

### 错误 3：没有交接机制

**现象：** 上一个会话完成了 80% 的工作。新的会话打开后说"让我检查一下文件"，找到的是过时的笔记，从头开始重新做了已经完成的工作——甚至重写了完成的文件。

**原因：** 缺少交接面。会话结束时的状态没有以持久形式传递给下一个会话。

**修复：** 每个会话结束时写交接包（改了啥、为什么、还差什么）。新会话启动时先读交接包，再读状态文件。

---

## 7. 面试考点

### Q1：工作台的七个面是什么？每个缺失时的故障模式是什么？（难度：⭐⭐）

**参考答案：**
指令（智能体猜完成标准）、状态（会话从零开始）、范围（修改泄漏到无关代码）、反馈（在错误上声明成功）、验证（未检查就上线）、审查（无人检查）、交接（下个会话重新发现一切）。

七个面必须同时存在。模型可以换，七个面不能换。

### Q2："工作台失败穿着提示词工程的衣服"是什么意思？（难度：⭐⭐）

**参考答案：**
大多数智能体失败故事看起来像模型问题——"模型不懂我的代码库"、"模型走偏了"。但实际上它们是工作台问题：没有状态文件所以模型从零开始、没有范围契约所以模型修改了不该碰的文件、没有反馈回路所以模型基于想象做决策。

提示词可以影响单次输出质量，但无法解决跨会话的持久化、跨轮次的反馈、跨角色的审查问题。这些问题需要工作台来解决，不是提示词。

### Q3：工作台工程和框架工程的区别是什么？（难度：⭐⭐⭐）

**参考答案：**
框架给你运行时（LangGraph、AutoGen、Agents SDK）——它们决定模型如何被调用、工具如何被执行、图如何被遍历。工作台给智能体在那个运行时中工作的**环境**——指令路由、状态持久化、范围约束、反馈捕获、验证门控。

你需要两者。框架解决"如何运行"，工作台解决"如何可靠地运行"。框架可以换（从 LangGraph 迁移到 OpenAI SDK），工作台应该保持（状态文件、范围契约、反馈运行器是框架无关的）。

### Q4：如何将任意供应商的"工作台模式"翻译成原语？（难度：⭐⭐⭐）

**参考答案：**
去掉"智能体"这个标签。一次智能体运行是跨越时间、进程和机器的计算。

八个原语：函数（Function）、工作者（Worker）、触发器（Trigger）、运行时（Runtime）、HTTP/RPC、队列（Queue）、会话持久化（Session Persistence）、授权策略（Authorization Policy）。

例如："Ralph Loop" → 一个触发器重新入队任务 + 会话持久化携带目标。"PEV" → 三个工作者通过状态和队列通信。"MCP 服务器" → 工作者通过 RPC 暴露函数，能力列表作为授权。

翻译到原语后，你可以无视供应商的品牌，构建框架无关的工作台。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作台 (Workbench) | "设置" | 模型周围的面——使工作可靠的工程化环境 |
| 面 (Surface) | "文档"或"脚本" | 智能体每轮读写的有名字的、机器可读的输入/输出 |
| 系统真实来源 (System of Record) | "笔记" | 聊天记录消失后，智能体当作事实来源的文件 |
| 完成定义 (Definition of Done) | "验收" | 客观的、文件支持的检查清单——智能体无法伪造 |
| 工作台审计 (Workbench Audit) | "仓库就绪度检查" | 对七个面进行检查，在工作开始前标记缺失的部分 |
| 原语 (Primitive) | "基础构建块" | 函数、工作者、触发器、运行时、队列、持久化、策略 |

---

## 📚 小结

不是模型不理解 Python，是模型不理解工作。本课定义了工作台的七个面（指令、状态、范围、反馈、验证、审查、交接），解释了每个缺失时的故障模式，并将行业流行模式翻译回了八个分布式系统原语。关键结论：模型可以换，七个面不能换。大多数智能体失败故事穿着提示词工程的衣服，实际上是工作台失败。

下一课我们将构建最小工作台——三个文件启动一个可靠的智能体工作环境。

---

## ✏️ 练习

1. **【理解】** 找一个你运行智能体的仓库。给七个面打分，从 0（缺失）到 2（健康）。你的最弱的面是什么？

2. **【实现】** 扩展 `main.py`，让纯提示词运行也产生虚构的"成功"声明。验证验证门控是否能捕获它。

3. **【思考】** 为你的产品添加第八个面。论证为什么它不能归入现有七个面中的任何一个。

4. **【实现】** 用不同的存根智能体重跑脚本——假设它产生一个额外的文件写入。哪个面最先捕获它？

5. **【思考】** 将阶段 14 · 26 中提到的五个行业常见故障模式映射到七个面上。每个面设计来吸收哪种模式？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 工作台对比 | `code/main.py` | 纯提示词 vs 工作台引导，生成失败模式报告 |
| 技能提示词 | `outputs/skill-workbench-audit.md` | 审计现有仓库的七个工作台面 |

---

## 📖 参考资料

1. [博客] Anthropic. "Effective Harnesses for Long-Running Agents". https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
2. [博客] LangChain. "The Anatomy of an Agent Harness". https://blog.langchain.com/the-anatomy-of-an-agent-harness/
3. [博客] MongoDB. "The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System". https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system
4. [博客] Addy Osmani. "Agent Harness Engineering". https://addyosmani.com/blog/agent-harness-engineering/
5. [论文] "Harness Engineering for Language Agents". preprints.org, 2026 年 3 月. https://www.preprints.org/manuscript/202603.1756

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
