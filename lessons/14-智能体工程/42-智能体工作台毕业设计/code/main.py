"""组装可复用的工作台包到 outputs/agent-workbench-pack/。

固定 Schema、脚本和模板，使其可投放到任何目标仓库。
重新运行是幂等的。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
PACK_DIR = HERE / "agent-workbench-pack"
BIN_DIR = PACK_DIR / "bin"
DOCS_DIR = PACK_DIR / "docs"
SCHEMAS_DIR = PACK_DIR / "schemas"
SCRIPTS_DIR = PACK_DIR / "scripts"


# ── AGENTS.md — 路由器（< 50 行） ─────────────────────────

AGENTS_MD = """\
# AGENTS.md

读取以下文件：

1. `agent_state.json` — 上一个会话停在哪
2. `task_board.json` — 什么在进行中、什么在待办
3. `docs/agent-rules.md` — 规则（按需加载）

完成定义：活动任务的验收命令退出码为 0。
验证命令：`python3 -m pytest -x`
"""

# ── Schema 定义 ────────────────────────────────────────────

STATE_SCHEMA = {
    "$id": "agent_state.schema.json",
    "type": "object",
    "required": ["schema_version", "active_task_id", "touched_files", "next_action"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "active_task_id": {"type": ["string", "null"], "pattern": r"^(T-\d{3,}|)$"},
        "touched_files": {"type": "array", "items": {"type": "string"}},
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

SCOPE_SCHEMA = {
    "$id": "scope_contract.schema.json",
    "type": "object",
    "required": ["task_id", "goal", "allowed_files", "forbidden_files", "acceptance_criteria"],
    "properties": {
        "task_id": {"type": "string"},
        "goal": {"type": "string"},
        "allowed_files": {"type": "array", "items": {"type": "string"}},
        "forbidden_files": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
    },
}

# ── 文档模板 ────────────────────────────────────────────────

AGENT_RULES = """\
# 智能体规则

## startup/state-file-fresh
- category: startup
- check: state_file_fresh
智能体在任何工具调用之前必须读取 agent_state.json。

## done/tests-pass
- category: definition_of_done
- check: tests_pass
任务完成的唯一标准是验收命令退出码为 0。
"""

REVIEWER_RUBRIC = """\
# 审查员评分标准

每个维度 0-2 分，满分 10 分。

| 维度 | 问题 |
|------|------|
| 问题匹配 | 这个改动解决的是任务描述的问题？ |
| 范围纪律 | 修改是否限于契约范围？ |
| 假设记录 | 假设写在可审查的地方？ |
| 验证质量 | 验收命令真正证明了目标？ |
| 交接就绪 | 下一个会话能干净地继续？ |

7 分以下软失败，5 分以下硬失败。
"""

# ── 安装脚本 ────────────────────────────────────────────────

INSTALL_SH = """\
#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-.}"
PACK_VERSION="1.0.0"

if [ -f "$TARGET/.workbench-version" ] && [ "${FORCE:-}" != "1" ]; then
    echo "ERROR: 工作台包已存在。使用 FORCE=1 覆盖。"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 复制包内容（排除 bin 和 VERSION）
for item in AGENTS.md docs schemas scripts; do
    cp -r "$SCRIPT_DIR/$item" "$TARGET/"
done

echo "$PACK_VERSION" > "$TARGET/.workbench-version"

if [ -d "$TARGET/.github/workflows" ]; then
    echo "CI 目录已存在。请手动添加 verify_agent.py 步骤。"
fi

echo "✅ 工作台包安装完成 (v$PACK_VERSION)"
echo "下一步：填充任务板，设置验收命令，运行 scripts/init_agent.py"
"""

# ── README ──────────────────────────────────────────────────

README = """\
# 智能体工作台包

可复用的智能体工作台——投放到任何仓库即可让智能体可靠工作。

## 安装

```bash
bash bin/install.sh /path/to/your/repo
```

## 内容

- `AGENTS.md` — 路由器（< 50 行）
- `docs/` — 规则、审查评分标准、交接协议
- `schemas/` — 状态、任务板、范围契约的 JSON Schema
- `scripts/` — 初始化、反馈、验证、交接生成器
- `bin/install.sh` — 幂等安装器

## 卸载

删除 `AGENTS.md`、`docs/`、`schemas/`、`scripts/`。用户的 `agent_state.json`、`task_board.json` 和 `outputs/` 保留。
"""


# ── 组装函数 ────────────────────────────────────────────────

def assemble_pack() -> None:
    """组装工作台包到 agent-workbench-pack/ 目录。"""
    # 清理旧内容
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)

    # 创建目录结构
    for d in [BIN_DIR, DOCS_DIR, SCHEMAS_DIR, SCRIPTS_DIR]:
        d.mkdir(parents=True)

    # 写入文件
    (PACK_DIR / "AGENTS.md").write_text(AGENTS_MD)
    (PACK_DIR / "VERSION").write_text("1.0.0\n")
    (PACK_DIR / "README.md").write_text(README)

    # 文档
    (DOCS_DIR / "agent-rules.md").write_text(AGENT_RULES)
    (DOCS_DIR / "reviewer-rubric.md").write_text(REVIEWER_RUBRIC)

    # Schema
    (SCHEMAS_DIR / "agent_state.schema.json").write_text(
        json.dumps(STATE_SCHEMA, indent=2) + "\n")
    (SCHEMAS_DIR / "task_board.schema.json").write_text(
        json.dumps(BOARD_SCHEMA, indent=2) + "\n")
    (SCHEMAS_DIR / "scope_contract.schema.json").write_text(
        json.dumps(SCOPE_SCHEMA, indent=2) + "\n")

    # 安装脚本
    install_sh = BIN_DIR / "install.sh"
    install_sh.write_text(INSTALL_SH)
    install_sh.chmod(0o755)


def print_tree() -> None:
    """打印包目录结构。"""
    import os
    for root, dirs, files in os.walk(PACK_DIR):
        level = len(Path(root).relative_to(PACK_DIR).parts)
        indent = "  " * level
        basename = Path(root).name if level > 0 else Path(root).name
        print(f"{indent}{basename}/")
        for f in sorted(files):
            print(f"{indent}  {f}")


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    assemble_pack()
    print("工作台包已组装到 agent-workbench-pack/\n")
    print_tree()

    # 幂等性测试：重新组装不应出错
    assemble_pack()
    print("\n幂等组装成功")


if __name__ == "__main__":
    main()
