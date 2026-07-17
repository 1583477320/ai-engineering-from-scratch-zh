# 最小智能体工作台

> 最小的有用工作台是三个文件：一个根指令路由、一个状态文件、一个任务板。其他一切都在此之上分层构建。如果一个仓库无法携带这三个文件，没有模型能拯救它。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 31（为什么强大模型仍然失败）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 34（仓库记忆与持久状态）— 本章的 `agent_state.json` 将在那里升级为 Schema 优先的状态管理

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义最小可行工作台的三个文件——指令路由、状态文件、任务板
- [ ] 解释为什么短的根路由器胜过长的单体 `AGENTS.md`
- [ ] 构建一个智能体每轮读取、结束时写入的状态文件
- [ ] 构建一个无需聊天记录即可跨会话存活的任务板

---

## 1. 问题

大多数团队搭建工作台的方式是写一个 3000 行的 `AGENTS.md` 然后宣布完成。模型加载它，忽略无法总结的部分，然后在和之前完全一样的面上失败。

你需要的是相反的东西。一个微小的根文件，只在需要时将智能体路由到更深的文件中。持久的状态，智能体在执行前读取、执行后写入。一个任务板，说清什么在进行中、什么被阻塞了、什么在待办。

**三个文件。** 每个有明确的职责。每个都足够机器可读，以后可以演变成真正的系统。

---

## 2. 概念

### 2.1 三个文件

```
任务 → 状态文件（agent_state.json） → 智能体循环 → 更新状态
                                      ↑
指令路由（AGENTS.md） → 指向状态文件 + 任务板 + 规则文档
                                      ↓
任务板（task_board.json） → todo → in_progress → done
```

| 文件 | 用途 | 类比 |
|------|------|------|
| **指令路由** `AGENTS.md` | 任务描述、规则、边界 | 项目 README + 路由表 |
| **状态文件** `agent_state.json` | 当前进度、已完成任务 | 检查点 |
| **任务板** `task_board.json` | 待办事项、优先级、状态 | 队列 |

### 2.2 AGENTS.md 是路由器，不是手册

好的 `AGENTS.md` 很短。它指向：

- 状态文件（"你在哪里"）
- 任务板（"还差什么"）
- 深度规则文档（`docs/agent-rules.md`，按需加载）
- 验证命令（"怎么知道它能用"）

更长的内容放在深度文档中，仅当需要时加载。长手册被忽略。短路由器被遵循。

```
# 好的 AGENTS.md（< 50 行）
当前会话开始前读取：
1. agent_state.json —— 上一个会话停在哪
2. task_board.json —— 什么在进行中、什么在待办
3. docs/agent-rules.md —— 启动规则、范围、完成定义（按需加载）
```

```
# 差的 AGENTS.md（3000 行）
包含了完整的历史决策、团队规范、架构说明、编码规范……
智能体读完前 50 行就耗尽了注意力预算
```

### 2.3 agent_state.json 是系统的真实来源

状态携带：活动任务 ID、修改过的文件、做的假设、阻塞项、下一步操作。

智能体每轮读取。下一个会话读取它而不是重放聊天记录。

状态存在文件中，因为聊天记录不可靠。会话会关闭。对话会被截断。文件不会。

```json
{
  "active_task_id": "T-001",
  "touched_files": ["app.py", "test_app.py"],
  "assumptions": ["验证逻辑和现有路由兼容"],
  "blockers": [],
  "next_action": "run verification command"
}
```

### 2.4 task_board.json 是队列

任务板携带每个任务及其状态 `todo | in_progress | done | blocked`。它是智能体在状态为空时的拉取源，也是你想知道智能体是否在轨道上时的检查源。

```json
[
  {
    "id": "T-001",
    "goal": "为 /signup 添加输入验证",
    "owner": "builder",
    "acceptance": ["pytest test_app.py::test_signup_rejects_short_password"],
    "status": "todo"
  }
]
```

任务板故意保持轻量。当它超过一屏时，你有了规划问题，不是任务板问题。

### 2.5 三个文件是底线，不是天花板

后续课程（如下表）在这三个文件之上分层构建：

| 后续课程 | 在三个文件之上添加 |
|---------|----------------|
| 范围契约（36） | `scope_contract.json`——允许/禁止文件、验收条件 |
| 反馈运行器（37） | `feedback_record.jsonl`——命令输出的结构化捕获 |
| 验证门控（38） | 范围检查 + 测试验证 |
| 审查员（39） | 审查检查清单 |
| 交接（40） | `handoff_packet.json`——改了啥、为什么、还差什么 |

---

## 3. 从零实现

### 第 1 步：定义数据结构

```python
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class AgentState:
    """智能体状态——当前会话的进度记录。"""
    active_task_id: str | None
    touched_files: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_action: str = ""

@dataclass
class Task:
    """任务板上的一个任务。"""
    id: str
    goal: str
    owner: str                # builder | reviewer | human
    acceptance: list[str]     # 验收命令
    status: str = "todo"      # todo | in_progress | done | blocked
```

### 第 2 步：实现读写函数

```python
def load_state(path: Path) -> AgentState:
    raw = json.loads(path.read_text())
    return AgentState(**raw)

def save_state(path: Path, state: AgentState) -> None:
    path.write_text(json.dumps(asdict(state), indent=2) + "\n")

def load_board(path: Path) -> list[Task]:
    return [Task(**t) for t in json.loads(path.read_text())]

def save_board(path: Path, board: list[Task]) -> None:
    path.write_text(json.dumps([asdict(t) for t in board], indent=2) + "\n")
```

### 第 3 步：实现 AGENTS.md

```python
AGENTS_MD = """# AGENTS.md

This repo runs with a workbench. Read these before acting:

1. `agent_state.json` — where the last session stopped.
2. `task_board.json` — what is in flight, what is next.
3. `docs/agent-rules.md` — startup, scope, definition of done (load on demand).

Definition of done: the task referenced by `agent_state.active_task_id` has
`status == "done"` on `task_board.json` and the verification command listed in
its `acceptance` has exited 0.

Verification command: `python3 -m pytest -x`
"""
```

### 第 4 步：实现一轮智能体运行

```python
def run_one_turn(state: AgentState, board: list[Task]) -> tuple[AgentState, list[Task]]:
    """执行一轮智能体循环：领取任务 → 修改文件 → 运行验证 → 完成任务。"""
    # 如果当前没有活动任务，从任务板领一个新的
    if state.active_task_id is None:
        nxt = next((t for t in board if t.status == "todo"), None)
        if nxt is None:
            state.next_action = "no work on the board, idle"
            return state, board
        nxt.status = "in_progress"
        state.active_task_id = nxt.id
        state.next_action = f"start work on {nxt.id}: {nxt.goal}"
        return state, board

    # 如果活动任务已不存在
    active = next((t for t in board if t.id == state.active_task_id), None)
    if active is None:
        state.active_task_id = None
        state.next_action = f"active task missing from board; resetting"
        return state, board

    # 第 1 子轮：修改源文件
    if "app.py" not in state.touched_files:
        state.touched_files.append("app.py")
        state.next_action = f"add test for {active.id} acceptance"
        return state, board

    # 第 2 子轮：添加测试
    if "test_app.py" not in state.touched_files:
        state.touched_files.append("test_app.py")
        state.next_action = f"run verification command for {active.id}"
        return state, board

    # 完成
    active.status = "done"
    state.active_task_id = None
    state.touched_files = []
    state.next_action = "pick next task from board"
    return state, board
```

### 第 5 步：主流程

```python
def main():
    # 首次运行写入初始文件
    write_initial(state_path, board_path, agents_path)
    state = load_state(state_path)
    board = load_board(board_path)

    print("before turn:")
    print(f"  active task : {state.active_task_id}")
    print(f"  next action : {state.next_action!r}")
    print(f"  todo on board: {[t.id for t in board if t.status == 'todo']}")

    state, board = run_one_turn(state, board)
    save_state(state_path, state)
    save_board(board_path, board)

    print("\nafter turn:")
    print(f"  active task : {state.active_task_id}")
    print(f"  touched     : {state.touched_files}")
    print(f"  next action : {state.next_action!r}")
    print(f"  board status: {[(t.id, t.status) for t in board]}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 三个文件在不同产品中的体现

| 产品 | 路由器 | 状态 | 任务板 |
|------|--------|------|--------|
| Claude Code | AGENTS.md / CLAUDE.md | `.claude/state.json` 风格存储 | 钩子 |
| Codex / Cursor | 工作区规则 | 会话记忆 | 聊天侧边栏待办 |
| 自定义 Python 智能体 | AGENTS.md | agent_state.json | task_board.json |

名字不同，形状不变。

### 4.2 嵌套 AGENTS.md（最近优先）

OpenAI 在其主仓库中提供了 88 个 `AGENTS.md` 文件，每个子组件一个。Codex、Cursor、Claude Code 和 Copilot 都会从工作目录向仓库根目录遍历，合并沿途找到的所有 `AGENTS.md`。

子目录文件扩展根文件（extend）。Codex 还支持 `AGENTS.override.md` 来替换而非扩展——不过这是 Codex 特有的，跨工具时不要依赖。

Augment Code 的测量最有价值：**好的 AGENTS.md 带来的质量提升相当于从 Haiku 升级到 Opus；差的 AGENTS.md 比没有还糟。**

### 4.3 跨工具符号链接

一个根文件加上符号链接可以让每个编码智能体使用同一个事实来源：

```bash
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md .github/copilot-instructions.md
ln -s AGENTS.md .cursorrules
```

Nx 的 `nx ai-setup` 从单个配置自动完成这个操作，覆盖 Claude Code、Cursor、Copilot、Gemini、Codex 和 OpenCode。

### 4.4 必须拒绝的反模式

- **指令冲突**——多条指令静默地将智能体从交互模式降级为贪心模式（ICLR 2026 AMBIG-SWE：48.8% → 28% 解决率）。**修复：** 用数字优先级，不要平铺冲突指令
- **不可验证的风格规则**——"遵循 Google Python 风格指南"没有强制命令。智能体只能想象其含义。**修复：** 每条风格规则配精确的 lint 命令
- **为人类而不是为智能体写文档**——浪费上下文预算。简洁是特性，不是缺陷

---

## 5. 工程最佳实践

### 5.1 三文件设计原则

| 原则 | 说明 |
|------|------|
| 路由器不超过 50 行 | 让审查员每次 PR 都能重读 |
| 状态文件每轮写入 | 崩溃时最多丢一轮的数据 |
| 任务板不超过一屏 | 超过一屏意味着规划问题不是板子问题 |
| 三个文件从第一天开始 | 即使只有一个任务也要有这三个文件 |

### 5.2 中文场景特别建议

- **AGENTS.md 中的注释用中文**——方便团队审查，但字段名和路径保持英文
- **如果团队使用中文代码库**（中文函数名、中文文件名），状态文件中的 `touched_files` 字段名仍用英文，值可以用中文路径
- **跨工具符号链接在中文路径下注意编码**——某些工具链接中文字符路径会有问题。建议根文件使用英文名 `AGENTS.md`

### 5.3 踩坑经验

- **路由器太长 = 没有路由器**——3000 行的 AGENTS.md 等于没有。修复：砍到 50 行以内，详细内容放子文档
- **状态文件不写 = 不可恢复**——会话结束时不写状态文件，下一次会话从零开始。修复：每轮结束都写
- **没有任务板 = 随机的智能体**——智能体不知道下一步做什么，随机做点"看起来有用"的事。修复：总是有一个任务板，即使只有一个待办

---

## 6. 常见错误

### 错误 1：路由器膨胀

**现象：** AGENTS.md 从开始到结束长了 3000 行。智能体每次都加载全部内容，但注意力预算只有前 50 行。后面 2950 行从未被执行。新团队成员不敢删除任何内容，因为每条记录看起来都很重要。

**原因：** 分不清路由和参考。AGENTS.md 应该是指南针（指向信息），不是百科全书（包含信息）。

**修复：**
```
# ❌ AGENTS.md 包含所有信息
这个仓库的架构是……编码规范是……2023 年 3 月做了一个决策……
# 3000 行后……
还忘记说了，运行时需要 OPENAI_API_KEY

# ✓ AGENTS.md 只是路由器
读取 agent_state.json → 检查 task_board.json → 按需加载 docs/*.md
```

### 错误 2：指令冲突

**现象：** AGENTS.md 中说"只修改指定的文件"，又说"确保所有测试通过"。当修改测试辅助文件时，智能体认为自己在做正确的事——它在遵守第二条指令。但第一条指令被违反了。

**原因：** 两条指令隐含冲突，没有优先级。

**修复：** 明确优先级。`如果指令冲突，编号小的优先。规则：1. 禁止文件优先级最高 2. 范围次之 3. 风格指南最低。`

### 错误 3：为人类写文档，不是为智能体

**现象：** AGENTS.md 读起来像人类团队的入职手册——"我们团队遵循敏捷开发方法论，每个迭代两周……" 智能体不需要知道方法论，它需要知道文件结构、验证命令、禁止操作。

**修复：** 为智能体写文档。简洁是特性。只包含智能体需要知道的信息——角色、规则、工具、验证方法。

---

## 7. 面试考点

### Q1：最小工作台的三个文件是什么？为什么是三个？（难度：⭐⭐）

**参考答案：**
（1）**指令路由**（AGENTS.md）——做什么、规则、边界
（2）**状态文件**（agent_state.json）——做到哪一步了
（3）**任务板**（task_board.json）——下一步做什么

三个文件对应三个核心问题：**做什么 → 做到哪了 → 下一步做什么**。有了这三个，即使是最简单的工作台也能回答智能体需要的基本问题。没有它们，任何前沿模型也无法可靠工作。

### Q2：为什么路由器（AGENTS.md）应该保持简短？（难度：⭐⭐）

**参考答案：**
两个原因。第一，智能体的注意力预算有限——每次会话都读取路由器，但只能有效处理前 50 行左右的内容。3000 行的路由器等于没有路由器。第二，路由器足够短时，审查员每次 PR 都会重读它，这是防止它悄悄膨胀回长篇大论的唯一机制。

路由器的职责是**指向**信息，不是**包含**信息。详细规则放 `docs/agent-rules.md`，架构放 `docs/architecture.md`，按需加载。

### Q3：什么是嵌套 AGENTS.md 与最近优先策略？（难度：⭐⭐⭐）

**参考答案：**
嵌套 AGENTS.md 是指仓库每个子目录都有自己的 AGENTS.md。运行时从工作目录向根目录遍历，合并沿途找到的所有 AGENTS.md 文件。子目录文件扩展（extend）根文件。

优势：每个子组件可以声明自己的规则，不影响其他组件。OpenAI 在其主仓库中使用了 88 个 AGENTS.md 文件。

注意事项：（1）关于 Codex 的 `AGENTS.override.md`（替换而非扩展）是 Codex 特有的，跨工具不要依赖（2）Augment Code 的测量表明：好的 AGENTS.md 带来的质量提升相当于从 Haiku 升级到 Opus；坏的比没有更糟

### Q4：状态文件为什么存在文件中而不是聊天记录中？（难度：⭐⭐）

**参考答案：**
聊天记录是不可靠的。会话会关闭。对话会被截断。模型切换时之前的上下文丢失。状态文件在仓库中，**版本化、可审查、跨会话持久**。

智能体每轮读取状态文件，每轮结束时写入。即使模型换了（从 GPT-4 换到 Claude），状态文件还在，工作不中断。下一个会话读取它，而不是重放聊天记录——这意味着下一个智能体可以从上一个停下的地方继续。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 路由器 (Router) | "AGENTS.md" | 短的根文件——指向深度文档和文件 |
| 状态文件 (State File) | "笔记" | 机器可读的智能体位置记录——每轮写入 |
| 任务板 (Task Board) | "待办列表" | JSON 队列——含状态、负责人、验收条件 |
| 系统真实来源 (System of Record) | "事实来源" | 聊天记录消失后工作台视为权威的文件 |
| 嵌套 AGENTS.md | "多个指令文件" | 每个子目录有独立的指令文件——从工作目录向根合并 |

---

## 📚 小结

最小工作台只需要三个文件：指令路由（做什么）、状态文件（做到哪了）、任务板（下一步）。三个文件从第一天开始——即使只有一个任务。路由器保持 50 行以内，详细内容放子文档按需加载。状态文件每轮写入，任务板不超过一屏。

下一课将在这个基础上添加规则引擎，把"请小心"变成可检查的约束。

---

## ✏️ 练习

1. **【实现】** 给 `agent_state.json` 添加 `last_run` 时间戳。文件超过 24 小时且无人确认时拒绝运行。

2. **【实现】** 给任务板添加 `priority` 字段，修改拉取逻辑使其总是选最高优先级 `todo`。

3. **【实现】** 将 `task_board.json` 迁移为 JSON Lines（每行一个任务），使 diff 在版本控制中更清晰。

4. **【实现】** 写一个 `lint_workbench.py`——如果 `AGENTS.md` 超过 80 行或引用了不存在的文件则报错。

5. **【思考】** 三个文件中，失去哪一个的伤害最大？论证你的选择。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 最小工作台 | `code/main.py` | 三文件工作台实现——路由器 + 状态 + 任务板，演示跨轮次运行 |

---

## 📖 参考资料

1. [官方文档] agents.md — 开放规范. https://agents.md/ — 被 Cursor、Codex、Claude Code、Copilot、Gemini、OpenCode 采用
2. [博客] Augment Code. "A Good AGENTS.md Is a Model Upgrade. A Bad One Is Worse Than No Docs At All". https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files — 有测量数据的质量跃升
3. [博客] Blake Crosley. "AGENTS.md Patterns: What Actually Changes Agent Behavior". https://blakecrosley.com/blog/agents-md-patterns — 经验上有用的和没有用的
4. [博客] Nx. "Teach Your AI Agent How to Work in a Monorepo". https://nx.dev/blog/nx-ai-agent-skills — 跨六个工具的单一来源生成
5. [博客] Anthropic. "Claude Code Subagents and Session Store". https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/sub-agents

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
