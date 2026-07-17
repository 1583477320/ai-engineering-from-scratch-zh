# 毕业设计：打包可复用的智能体工作台

> 本章以一个可 `cp -r` 的工作台包收尾。十一课的面压缩成一个目录——第二天早上就能让智能体在新仓库中可靠工作。这个包就是本章的核心产物。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 31 到 14 · 41（全部工作台课程）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 41（真实仓库上的工作台）— 本课的包是 41 课前后对比中工作台侧的实现

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 将七个工作台面打包成一个可投放的目录
- [ ] 固定 Schema、脚本和模板，让新仓库获得已知良好的基线
- [ ] 添加一个安装脚本——幂等地投放工作台包
- [ ] 判断什么留在包内、什么放在包外，并论证每次取舍

---

## 1. 问题

住在 Google Doc、聊天记录和三个半记半忘脚本中的工作台，每个季度都会被重建。治愈方案是一个版本化的工作台包：一个包含所有面、Schema、脚本和一键安装器的仓库或目录。

本课结束时你会得到：
- `outputs/agent-workbench-pack/`——磁盘上打包好的工作台
- `bin/install.sh`——将其投放到任何目标仓库的安装器

---

## 2. 概念

### 2.1 工作台包的布局

```
agent-workbench-pack/
├── AGENTS.md                          # 路由器（≤50 行）
├── docs/
│   ├── agent-rules.md                 # 规则集（阶段 14 · 33）
│   ├── reliability-policy.md          # 可靠性策略
│   ├── handoff-protocol.md            # 交接协议
│   └── reviewer-rubric.md             # 审查员评分标准（阶段 14 · 39）
├── schemas/
│   ├── agent_state.schema.json        # 状态 Schema（阶段 14 · 34）
│   ├── task_board.schema.json         # 任务板 Schema
│   └── scope_contract.schema.json     # 范围契约 Schema（阶段 14 · 36）
├── scripts/
│   ├── init_agent.py                  # 初始化脚本（阶段 14 · 35）
│   ├── run_with_feedback.py           # 反馈运行器（阶段 14 · 37）
│   ├── verify_agent.py                # 验证门控（阶段 14 · 38）
│   └── generate_handoff.py            # 交接包生成器（阶段 14 · 40）
├── bin/
│   └── install.sh                     # 幂等安装器
├── VERSION                            # 包版本
└── README.md
```

### 2.2 什么留在包内，什么放在包外

**留在包内：**
- 面 Schema（它们是契约）
- 四个脚本（它们是运行时）
- 四个文档（它们是规则和评分标准）

**放在包外：**
- 项目特定的任务（任务属于目标仓库的任务板，不属于包）
- 供应商 SDK 调用（包是框架无关的）
- 入职散文（包应该放在团队现有入职旁边，而不是嵌入其中）

### 2.3 安装器

一个简短的 `bin/install.sh`：

1. 没有 `--force` 时拒绝在已有包上安装
2. 将包复制到目标仓库
3. 如果存在 `.github/workflows/` 则接入 CI
4. 打印下一步：填充任务板、设置验收命令、运行初始化脚本

### 2.4 版本化

包携带 `VERSION` 文件。Schema 变更和需要迁移的脚本变更 bump 主版本号。仅文档变更 bump 补丁版本号。目标仓库的 `agent_state.json` 记录初始化时使用的包版本。

### 2.5 渐进式披露文档结构

```
AGENTS.md                  # 路由器，< 50 行
docs/
  agent-rules.md           # 完整规则集
  reliability-policy.md    # 可靠性策略
  handoff-protocol.md      # 交接协议
  reviewer-rubric.md       # 审查员评分标准
```

路由器只保留指针，详细内容按需加载——这是阶段 14 · 33 的渐进式披露原则在包级别的应用。

---

## 3. 从零实现

### 第 1 步：定义包目录结构

```python
from pathlib import Path
import shutil

PACK_DIR = Path(__file__).parent / "agent-workbench-pack"

# 包结构定义
PACK_STRUCTURE = {
    "AGENTS.md": _AGENTS_MD,
    "docs/agent-rules.md": _AGENT_RULES,
    "docs/reliability-policy.md": _RELIABILITY_POLICY,
    "docs/handoff-protocol.md": _HANDOFF_PROTOCOL,
    "docs/reviewer-rubric.md": _REVIEWER_RUBRIC,
    "schemas/agent_state.schema.json": _STATE_SCHEMA,
    "schemas/task_board.schema.json": _BOARD_SCHEMA,
    "schemas/scope_contract.schema.json": _SCOPE_SCHEMA,
    "VERSION": "1.0.0",
    "README.md": _README,
}
```

### 第 2 步：定义安装器

```bash
#!/usr/bin/env bash
# bin/install.sh — 幂等安装工作台包到目标仓库
set -euo pipefail

TARGET="${1:-.}"
PACK_VERSION="1.0.0"

# 检查是否已有包
if [ -f "$TARGET/.workbench-version" ] && [ "$FORCE" != "1" ]; then
    echo "ERROR: 工作台包已存在。使用 FORCE=1 覆盖。"
    exit 1
fi

# 复制包内容
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp -r "$SCRIPT_DIR/AGENTS.md" "$TARGET/"
cp -r "$SCRIPT_DIR/docs" "$TARGET/"
cp -r "$SCRIPT_DIR/schemas" "$TARGET/"
cp -r "$SCRIPT_DIR/scripts" "$TARGET/"

# 记录版本
echo "$PACK_VERSION" > "$TARGET/.workbench-version"

# 接入 CI（如果存在 .github/workflows/）
if [ -d "$TARGET/.github/workflows" ]; then
    echo "CI 目录已存在。请手动添加 verify_agent.py 步骤。"
fi

echo "✅ 工作台包安装完成 (v$PACK_VERSION)"
echo "下一步：填充任务板，设置验收命令，运行 scripts/init_agent.py"
```

### 第 3 步：定义包内容

```python
# AGENTS.md — 路由器，不超过 50 行
_AGENTS_MD = """# AGENTS.md

读取以下文件：
1. `agent_state.json` — 上一个会话停在哪
2. `task_board.json` — 什么在进行中、什么在待办
3. `docs/agent-rules.md` — 规则（按需加载）

完成定义：活动任务的验收命令退出码为 0。
验证命令：`python3 -m pytest -x`
"""

# agent_state Schema
_STATE_SCHEMA = {
    "$id": "agent_state.schema.json",
    "type": "object",
    "required": ["schema_version", "active_task_id", "touched_files", "next_action"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "active_task_id": {"type": ["string", "null"], "pattern": r"^(T-\\d{3,}|)$"},
        "touched_files": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
    },
}
```

### 第 4 步：实现安装函数

```python
def install(target_dir: Path, force: bool = False) -> None:
    """幂等地将工作台包安装到目标目录。"""
    version_file = target_dir / ".workbench-version"
    if version_file.exists() and not force:
        raise RuntimeError(f"工作台包已存在 ({version_file.read_text().strip()})。使用 force=True 覆盖。")

    for rel_path, content in PACK_STRUCTURE.items():
        dest = target_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    version_file.write_text("1.0.0\n")
    print(f"✅ 工作台包安装完成 (v1.0.0)")
    print(f"下一步：填充任务板，设置验收命令，运行 scripts/init_agent.py")
```

### 第 5 步：运行演示

```python
def main():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "my-project"
        target.mkdir()
        install(target)

        # 显示安装结果
        for root, dirs, files in sorted(target.walk()):
            for f in files:
                rel = (root / f).relative_to(target)
                print(f"  {rel}")

        # 幂等性测试
        install(target)  # 应该不报错
        print("\n幂等安装成功")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 跨工具分发

| 模式 | 说明 |
|------|------|
| 目录投放 | `cp -r agent-workbench-pack /path/to/repo` |
| GitHub 模板仓库 | Fork-and-customize，VERSION 控制漂移 |
| 符号链接分发 | `ln -s AGENTS.md CLAUDE.md` 从单一来源分发到多个编码智能体 |
| Nx ai-setup | 从单一配置自动为 Claude Code、Cursor、Copilot 等生成文件 |

### 4.2 版本化和迁移

| 变更类型 | VERSION 操作 | 说明 |
|---------|-------------|------|
| Schema 变更 | bump 主版本号 | 需要迁移脚本 |
| 脚本变更 | bump 次版本号 | 需要检查器重跑 |
| 文档变更 | bump 补丁版本号 | 无迁移 |

### 4.3 生产模式

| 模式 | 说明 |
|------|------|
| VERSION 是契约 | 不是营销。主版本号 bump = 状态迁移。安装器写入 `.workbench-version` |
| 单一来源跨工具 | 一个 `AGENTS.md` 通过符号链接分发到 Claude Code、Cursor、Copilot |
| uninstall.sh 拒绝非平凡状态 | 卸载包时不删除 `agent_state.json`、`task_board.json`、`outputs/` |
| 包作为可发布技能 | `skillkit install agent-workbench-pack` 跨 32 个智能体投放 |

---

## 5. 工程最佳实践

### 5.1 工作台包设计原则

| 原则 | 说明 |
|------|------|
| 版本化是契约 | 主版本号 = Schema/脚本变更，补丁 = 文档 |
| 幂等安装 | 安装两次应该不报错（除非使用 `--force`） |
| 框架无关 | 不依赖 LangGraph、OpenAI SDK、Claude SDK 的任何特定功能 |
| 渐进式披露 | 路由器 < 50 行，详细文档按需加载 |

### 5.2 中文场景特别建议

- **包文档（README.md、agent-rules.md）用中文写**——方便中文团队理解
- **Schema 和脚本的字段名/函数名保持英文**——跨工具兼容
- **安装器的输出用中英文混合**——状态信息用中文，路径和命令用英文

### 5.3 踩坑经验

- **包包含项目特定内容**——把团队特有的规则、验收命令放进包里。新仓库安装时规则不适用。**修复：** 项目特定内容放在目标仓库，包只包含通用面
- **不版本化**——包更新时不 bump VERSION，目标仓库不知道自己在用哪个版本。**修复：** 每次变更 bump VERSION，安装器写入 `.workbench-version`
- **卸载包删除了用户数据**——卸载脚本删除了 `agent_state.json`。**修复：** 卸载只删除包自己的文件（Schema、脚本、文档），不删除用户状态

---

## 6. 常见错误

### 错误 1：包包含项目特定内容

**现象：** 包中的 `agent-rules.md` 包含"不要修改 `lib/legacy/auth.py`"——这是上一个项目特有的规则。新项目安装时这条规则不适用，甚至可能阻止合理修改。

**原因：** 项目特定内容被打包进了通用工作台。

**修复：**
```
# ❌ 包包含项目特定规则
"不要修改 lib/legacy/auth.py"

# ✓ 包只包含通用面，项目特定内容放在目标仓库
pack/AGENTS.md          → 通用路由器
pack/docs/agent-rules.md → 通用规则模板
target-repo/docs/my-rules.md → 项目特定规则
```

### 错误 2：不版本化

**现象：** 包更新了 Schema，但目标仓库不知道。新版本的脚本按新 Schema 写入，但旧的读取器按旧 Schema 解析，静默损坏数据。

**原因：** 没有 VERSION 文件，没有 `.workbench-version` 记录

**修复：** 每次变更 bump VERSION。安装器每次运行时写入 `.workbench-version`。脚本启动时检查版本。

### 错误 3：卸载包删除用户数据

**现象：** 卸载脚本 `rm -rf` 删除了包目录，同时也删除了 `agent_state.json`、`task_board.json` 和 `outputs/`。用户丢失了所有工作台状态。

**原因：** 卸载脚本没有区分"包的文件"和"用户的文件"

**修复：** 卸载脚本只删除包自己创建的文件（Schema、脚本、文档、`AGENTS.md`）。用户的文件（状态、任务板、交接包）保留。如果用户文件有未提交变更，卸载脚本拒绝执行。

---

## 7. 面试考点

### Q1：工作台包的核心布局是什么？（难度：⭐）

**参考答案：**
七个面的打包：
- `AGENTS.md` + `docs/` — 指令面（路由器 + 规则 + 评分标准）
- `schemas/` — 状态面和范围面的 Schema
- `scripts/` — 运行时面（初始化、反馈、验证、交接）
- `bin/install.sh` — 幂等安装器

包是框架无关的——不依赖 LangGraph、OpenAI SDK、Claude SDK。

### Q2：为什么包需要版本化？三种 bump 的区别是什么？（难度：⭐⭐）

**参考答案：**
版本化让目标仓库知道自己的基线。三种 bump：

- **主版本号（Major）**— Schema 或脚本变更需要迁移时。破坏性变更
- **次版本号（Minor）**— 脚本变更不需要迁移。功能增强
- **补丁版本号（Patch）**— 仅文档变更。无影响

安装器每次运行时写入 `.workbench-version`。脚本启动时检查版本——如果版本不匹配，拒绝运行或提示迁移。

### Q3：包的渐进式披露如何工作？（难度：⭐⭐）

**参考答案：**
与阶段 14 · 33 的渐进式披露原则相同，但作用域是包级别：

- `AGENTS.md` — 路由器，< 50 行，每次会话都读
- `docs/agent-rules.md` — 完整规则集，启动时读
- `docs/reviewer-rubric.md` — 审查员评分标准，仅审查时读
- `docs/handoff-protocol.md` — 交接协议，仅交接时读

路由器只保留指针，不包含详细内容。智能体从路由器出发，最多两跳到达任何规则。

### Q4：如何处理包的跨工具分发？（难度：⭐⭐⭐）

**参考答案：**
包的核心是单一来源的 `AGENTS.md`。跨工具分发通过符号链接：

```bash
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md .github/copilot-instructions.md
ln -s AGENTS.md .cursorrules
```

Nx 的 `nx ai-setup` 自动完成这个操作。包不应为每个工具维护不同版本——这会导致漂移和维护开销。

安装器负责创建这些符号链接。目标仓库的 `AGENTS.md` 是唯一的真实来源，其他工具通过链接读取。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作台包 (Workbench Pack) | "入门套件" | 一个版本化目录，包含所有七个面的 Schema、脚本和文档 |
| 安装器 | "设置脚本" | `bin/install.sh`——幂等地投放包到目标仓库 |
| 包版本 (Pack Version) | "VERSION" | Schema/脚本变更 bump 主版本号，仅文档变更 bump 补丁 |
| 渐进式披露 | "分层文档" | 路由器 < 50 行，详细文档按需加载 |
| 幂等安装 | "可重复安装" | 安装两次结果相同（除非使用 --force） |

---

## 📚 小结

十一课的工作台面打包成一个可复用的目录——`AGENTS.md`（路由器）+ `docs/`（规则和评分标准）+ `schemas/`（状态和范围的契约）+ `scripts/`（初始化、反馈、验证、交接的运行时）+ `bin/install.sh`（幂等安装器）。版本化是契约，不是营销。包是框架无关的——不依赖任何特定智能体框架。

至此第 14 章（智能体工程）全部完成。从评估驱动开发（30）到工作台工程（31-32）到七个面的实现（33-40）到对比验证（41）和最终打包（42）——你构建了一个完整的、可复用的、跨框架的智能体工作台。

---

## ✏️ 练习

1. **【思考】** 一个额外的第五文档值得晋升到标准包中吗？论证你的取舍。

2. **【实现】** 将安装器重写为 Python，添加 `--dry-run` 标志。对比与 bash 安装器的体验。

3. **【实现】** 添加 `bin/uninstall.sh`，安全地移除包但拒绝在状态文件有非平凡变更时执行。什么算"非平凡"？

4. **【实现】** 添加 `lint_pack.py`——包与 VERSION 不一致时失败。接入包自身的 CI。

5. **【实现】** 编写从手工工作台迁移到此包的操作手册。什么顺序能最小化停机时间？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 工作台包 | `outputs/agent-workbench-pack/` | 七面打包——Schema、脚本、文档、安装器 |
| 安装器 | `bin/install.sh` | 幂等安装脚本 |
| 技能提示词 | `outputs/skill-workbench-pack.md` | 为项目生成特定的工作台包 |

---

## 📖 参考资料

1. [GitHub] SkillKit. https://github.com/rohitg00/skillkit — 跨 32 个智能体安装工作台包
2. [博客] Nx. "Teach Your AI Agent How to Work in a Monorepo". https://nx.dev/blog/nx-ai-agent-skills — 跨六个工具的单一来源生成
3. [官方文档] agents.md — 开放规范. https://agents.md/ — 你的包路由器必须实现的标准
4. [GitHub] HKUDS/OpenHarness. https://github.com/HKUDS/OpenHarness — 工作台包的参考实现
5. [博客] Augment Code. "A Good AGENTS.md Is a Model Upgrade". https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files — 包文档的质量标准

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
