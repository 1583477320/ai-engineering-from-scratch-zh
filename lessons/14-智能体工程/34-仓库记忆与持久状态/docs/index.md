# 仓库记忆与持久状态——让智能体的状态跨会话存活

> 聊天记录是易失的。仓库是持久的。工作台把智能体状态存储在版本化的文件中，让下一个会话、下一个智能体、下一个审查员都从同一个事实来源读取。

**类型：** 实现课
**语言：** Python（标准库 + 可选 `jsonschema`）
**前置知识：** 阶段 14 · 32（最小工作台）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分哪些信息属于仓库记忆，哪些属于聊天记录——用"三个月后是否有用"作为判断标准
- [ ] 为 `agent_state.json` 和 `task_board.json` 编写 JSON Schema
- [ ] 构建一个状态管理器，支持加载、验证、修改和原子性持久化状态
- [ ] 使用 Schema 拒绝错误写入，防止损坏工作台

---

## 1. 问题

智能体完成了一个会话。聊天关闭了。下一个会话打开，问从哪里开始。模型说"让我检查一下文件"，读了过时的笔记，然后重新做了已经完成的工作。更糟的是，它重写了一个已完成的文件——因为没有人告诉它那个文件已经完成了。

问题的根源是：**聊天记录是易失的**。会话结束，上下文就消失了。智能体没有跨会话的记忆，每次都要从头开始"发现"项目状态。

解决方案是仓库记忆：状态存储在仓库的 JSON 文件中，在 Schema 下编写，原子性持久化，在代码审查中可差异比较。聊天是临时的信息流；仓库才是系统的真实来源。

---

## 2. 概念

### 2.1 什么属于仓库记忆

```
智能体循环 → StateManager → agent_state.schema.json → 验证 → agent_state.json
                                     │
                                     ↓ 不通过
                                  拒绝写入 + 抛出异常
```

| 属于仓库记忆 | 不属于仓库记忆 |
|-------------|--------------|
| 当前任务 ID | 原始聊天记录 |
| 本次会话修改的文件 | 词元级别的推理轨迹 |
| 智能体做的假设 | "用户好像很沮丧" |
| 未解决的阻塞项 | 采样的补全结果 |
| 下一步操作 | 供应商特定的模型 ID |

判断标准是**持久性**：三个月后在 CI 重跑时是否有用？如果是，存仓库。如果不是，那是遥测数据。

### 2.2 Schema 优先的状态管理

JSON Schema 是契约。没有 Schema，每个智能体会发明新字段，每个审查员要学习新的数据结构，每个 CI 脚本都要特判历史版本。有了 Schema，错误的写入就是被拒绝的写入。

Schema 覆盖：

- 必需字段
- 允许的 `status` 值
- 禁止的值（如数组不能为 `null`）
- 模式约束（任务 ID 匹配 `T-\d{3,}`）
- 版本字段（用于迁移）

### 2.3 原子性写入

状态写入必须承受部分失败：先写临时文件，fsync，再重命名覆盖目标文件。状态文件是事实来源；一个写了一半的文件比没有文件更糟。

```
tempfile.mkstemp（同一目录）→ 写入 → fsync → os.replace（原子重命名）
```

### 2.4 迁移

当 Schema 变更时，在 Schema 升级旁边放一个迁移脚本。状态文件携带 `schema_version` 字段；管理器拒绝加载它无法迁移的版本。

---

## 3. 从零实现

### 第 1 步：定义 Schema

```python
STATE_SCHEMA = {
    "$id": "agent_state.schema.json",
    "type": "object",
    "required": ["schema_version", "active_task_id", "touched_files", "next_action"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "active_task_id": {"type": ["string", "null"], "pattern": r"^(T-\d{3,}|)$"},
        "touched_files": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
    },
}

BOARD_SCHEMA = {
    "$id": "task_board.schema.json",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "goal", "owner", "acceptance", "status"],
        "properties": {
            "id": {"type": "string", "pattern": r"^T-\d{3,}$"},
            "goal": {"type": "string"},
            "owner": {"type": "string", "enum": ["builder", "reviewer", "human"]},
            "acceptance": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["todo", "in_progress", "done", "blocked"]},
        },
    },
}
```

### 第 2 步：实现 Schema 验证器

```python
import re
from typing import Any

class SchemaError(Exception):
    pass

def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """验证值是否符合 Schema。支持 type、enum、pattern、required、items。"""
    if "type" in schema and not _check_type(value, schema["type"]):
        raise SchemaError(f"{path}: expected {schema['type']}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and not re.match(schema["pattern"], value):
        raise SchemaError(f"{path}: {value!r} does not match /{schema['pattern']}/")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required field {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for idx, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{idx}]")
```

### 第 3 步：原子性写入

```python
import os
import tempfile
from pathlib import Path

def atomic_write(path: Path, content: str) -> None:
    """原子性写入：先写临时文件，fsync，再重命名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)  # 原子重命名
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
```

### 第 4 步：状态管理器

```python
import json

class StateManager:
    def __init__(self, state_path: Path, schema: dict[str, Any]):
        self.state_path = state_path
        self.schema = schema

    def load(self) -> Any:
        """加载并验证状态文件。"""
        raw = json.loads(self.state_path.read_text())
        validate(raw, self.schema)
        return raw

    def commit(self, state: Any) -> None:
        """验证并原子性写入状态。"""
        validate(state, self.schema)
        atomic_write(self.state_path, json.dumps(state, indent=2) + "\n")
```

### 第 5 步：运行演示

```python
# 初始化状态和任务板
initial_state = {
    "schema_version": 1,
    "active_task_id": None,
    "touched_files": [],
    "assumptions": [],
    "blockers": [],
    "next_action": "pick next task",
}
mgr = StateManager(state_path, STATE_SCHEMA)
mgr.commit(initial_state)

# 修改状态
state = mgr.load()
state["active_task_id"] = "T-001"
state["next_action"] = "read existing handler"
mgr.commit(state)

# 验证错误写入被拒绝
bad = dict(state)
bad["active_task_id"] = "T-bogus"  # 不匹配 T-\d{3,} 模式
try:
    mgr.commit(bad)
except SchemaError as exc:
    print("rejected:", exc)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 LangGraph 的检查点器

LangGraph 的 checkpointer 将图状态持久化到 SQLite、Postgres 或自定义后端。本课教的 Schema 是当检查点器失效时，你手动读取状态的基础。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 将状态持久化到 SQLite
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
# 图的每次状态变更都会被自动保存
```

### 4.2 Letta 的记忆块

Letta 使用结构化 Schema 的持久化记忆块。同样的纪律，但作用域限定在长期运行的角色上。

### 4.3 OpenAI Agents SDK 的会话存储

可插拔后端，Schema 感知。本课的状态文件就是本地文件后端的实现。

### 4.4 实践模式对照

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 原子性 temp-and-rename | 所有状态写入 | POSIX 和 Windows 都支持原子重命名 |
| 幂等键 | 非幂等工具调用 | 崩溃恢复时跳过已执行的调用 |
| 大文件与状态分离 | 包含 CSV/长文本的场景 | 状态文件只存路径，不存内容 |
| 事件溯源 + 快照 | 需要审计的长期运行 | 追加事件日志，定期快照，恢复时重放 |

---

## 5. 工程最佳实践

### 5.1 状态文件设计原则

| 原则 | 说明 |
|------|------|
| Schema 优先 | 先定义契约，再写入。错误的写入被拒绝，而不是静默损坏 |
| 原子性写入 | 写临时文件 → fsync → 重命名。部分失败不会损坏目标文件 |
| 大文件分离 | CSV、长文本、生成文件存为独立文件，状态只存路径 |
| 版本化 | `schema_version` 字段 + 迁移脚本，确保 Schema 升级不破坏工作台 |

### 5.2 中文场景特别建议

- **状态文件使用 UTF-8 编码**——JSON 标准要求 UTF-8，不要用其他编码
- **任务 ID 使用英文格式**——`T-001` 而不是 `任务-001`，避免跨工具兼容问题
- **假设和阻塞项用中文写**——方便团队审查，但保持 slug 和 ID 为英文

### 5.3 踩坑经验

- **非原子性写入是生产事故的常见原因**——`write_text()` + 异常捕获 = 部分写入静默损坏。始终使用 `tempfile.mkstemp` + `fsync` + `os.replace`
- **不要在状态文件中存储大文件内容**——CSV、长文本、生成文件会膨胀状态文件。保存路径，不保存内容
- **Schema 迁移必须有脚本**——`schema_version` 不匹配时拒绝加载，而不是静默升级。迁移脚本放在 `tools/migrate_state.py` 中

---

## 6. 常见错误

### 错误 1：使用 write_text() 直接写入状态文件

**现象：** 写入过程中进程崩溃或异常被静默捕获。状态文件变成半写状态，下一次加载时 JSON 解析失败，或者更糟——加载了损坏的数据。

**原因：** `write_text()` 不是原子性的。写入过程中断会留下部分数据。

**修复：**
```python
# ❌ 非原子性写入
path.write_text(json.dumps(state))

# ✓ 原子性写入：临时文件 + fsync + 重命名
fd, tmp = tempfile.mkstemp(dir=path.parent)
with os.fdopen(fd, "w") as f:
    f.write(json.dumps(state))
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, path)
```

### 错误 2：状态文件中存储大文件内容

**现象：** `agent_state.json` 从几 KB 膨胀到几 MB，因为里面存了 CSV 数据和长文本。每次加载和验证都变慢，Git 历史中充满了无意义的 diff。

**原因：** 大文件内容不适合放在状态文件中。状态应该是轻量级的指针。

**修复：** 大文件存为独立文件（或上传到对象存储），状态只存路径。

### 错误 3：Schema 版本不匹配时静默升级

**现象：** 智能体加载了旧版本的状态文件，自动添加了新字段，但旧字段的语义已经变了。结果是状态数据不一致，但没有报错。

**原因：** 没有版本检查。管理器应该在版本不匹配时拒绝加载。

**修复：** 每个状态文件携带 `schema_version` 字段。管理器检查版本，不匹配时拒绝加载并提示运行迁移脚本。

---

## 7. 面试考点

### Q1：什么信息应该存在仓库记忆中？什么不应该？判断标准是什么？（难度：⭐）

**参考答案：**
判断标准是**持久性**：三个月后在 CI 重跑时是否有用。

应该存的：当前任务 ID、修改的文件列表、智能体的假设、未解决的阻塞项、下一步操作。不应该存的：原始聊天记录、词元级推理轨迹、用户情绪判断、采样的补全结果。

核心原则：聊天是临时的信息流，仓库才是系统的真实来源。

### Q2：原子性写入为什么重要？它解决了什么问题？（难度：⭐⭐）

**参考答案：**
原子性写入解决的是"部分写入损坏"问题。如果直接用 `write_text()` 写入状态文件，写入过程中进程崩溃或异常被静默捕获，文件会处于半写状态。下一次加载时可能 JSON 解析失败，或者更糟——加载了损坏的数据。

原子性写入的流程：`tempfile.mkstemp`（在同一目录创建临时文件）→ 写入 → `fsync`（确保数据落盘）→ `os.replace`（原子重命名覆盖目标）。POSIX 和 Windows 都支持原子重命名，所以这个模式是跨平台的。

### Q3：如何处理 Schema 变更？为什么不能静默升级？（难度：⭐⭐）

**参考答案：**
每个状态文件携带 `schema_version` 字段。当管理器加载的文件版本与当前 Schema 版本不匹配时，拒绝加载并提示运行迁移脚本（`tools/migrate_state.py`）。

不能静默升级的原因：Schema 变更可能涉及语义变化——比如将 `blockers` 重命名为 `risks`，或者改变 `status` 的允许值。静默升级会导致新旧版本的数据混在一起，产生不一致。迁移脚本是显式的、可审查的、可测试的。

### Q4：事件溯源和快照如何配合工作？什么时候需要这种模式？（难度：⭐⭐⭐）

**参考答案：**
事件溯源：每次状态变更追加到事件日志（`state.events.jsonl`）。快照：定期将完整状态写入 `state.json`。恢复时：读取快照，然后重放快照时间戳之后的所有事件。

这种模式适合需要审计的长期运行智能体。它比纯快照多花了磁盘空间，但能逐字重放智能体的决策过程——这在调试长期运行任务时至关重要。Postgres 内部的 WAL（预写日志）使用的是同样的形状。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 仓库记忆 (Repo Memory) | "笔记文件" | 存储在仓库中版本化文件里的状态，在 Schema 下编写 |
| Schema 优先 (Schema-first) | "验证输入" | 先定义契约，再写入。拒绝偏离 |
| 原子性写入 (Atomic Write) | "就是重命名" | 写临时文件 → fsync → 重命名，部分失败不会损坏 |
| 迁移 (Migration) | "Schema 升级" | 将 vN 状态转换为 v(N+1) 状态的脚本 |
| 系统真实来源 (System of Record) | "事实来源" | 工作台视为权威的文件 |

---

## 📚 小结

聊天记录是易失的，仓库才是持久的。本课构建了一个 Schema 优先的状态管理器——用 JSON Schema 定义契约，用原子性写入防止部分失败损坏，用版本字段支持 Schema 迁移。你理解了什么信息属于仓库记忆（持久的、三个月后有用的），什么属于聊天记录（临时的、词元级的）。

下一课我们将解决另一个启动效率问题：智能体每次会话都要重新探测运行环境——初始化脚本可以把这个"启动税"只交一次。

---

## ✏️ 练习

1. 【实现】为 `StateManager` 添加 `last_human_touch` 时间戳。当智能体写入距离上一次人工编辑不到 5 秒时，拒绝写入。

2. 【实现】扩展验证器以支持 `oneOf`，使任务可以是构建任务或审查任务（不同的必需字段）。

3. 【实现】添加 `schema_version` 字段，编写从 v1 到 v2 的迁移脚本（将 `blockers` 重命名为 `risks`）。

4. 【实现】将存储后端从本地文件改为 SQLite。保持 `StateManager` API 不变。

5. 【实验】让两个智能体同时对同一个状态文件进行 50 毫秒写入竞争。会发生什么？原子性重命名如何保护你？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 状态管理器 | `code/main.py` | Schema 验证 + 原子性写入的状态管理器 |
| Schema 定义 | `code/main.py` 中的常量 | `agent_state.schema.json` 和 `task_board.schema.json` |
| 技能提示词 | `outputs/skill-state-schema.md` | 为项目生成 Schema 和状态管理器 |

---

## 📖 参考资料

1. [官方文档] JSON Schema Specification: https://json-schema.org/specification.html
2. [官方文档] LangGraph Checkpointers: https://langchain-ai.github.io/langgraph/concepts/persistence/
3. [官方文档] Letta Memory Blocks: https://docs.letta.com/concepts/memory
4. [GitHub] Hive Issue #6263 — non-atomic state.json writes: https://github.com/aden-hive/hive/issues/6263 — 真实项目中的部分写入损坏案例
5. [博客] Indium. "7 State Persistence Strategies for Long-Running AI Agents in 2026": https://www.indium.tech/blog/7-state-persistence-strategies-ai-agents-2026/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
