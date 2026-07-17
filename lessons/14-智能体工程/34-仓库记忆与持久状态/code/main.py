"""Schema 优先的智能体状态管理，支持原子性写入。

写入 `agent_state.schema.json` 和 `task_board.schema.json`，
实现一个纯标准库的验证器（支持 required、type、enum、pattern、items），
以及一个带临时文件重命名的 StateManager，防止部分写入损坏。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
WORK = HERE / "workdir"


# ── Schema 定义 ──────────────────────────────────────────────

STATE_SCHEMA: dict[str, Any] = {
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


BOARD_SCHEMA: dict[str, Any] = {
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


# ── Schema 验证器 ────────────────────────────────────────────

class SchemaError(Exception):
    """Schema 验证失败。"""
    pass


def _check_type(value: Any, types: str | list[str]) -> bool:
    """检查值是否匹配给定类型（支持联合类型）。"""
    type_list = [types] if isinstance(types, str) else types
    for t in type_list:
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
    return False


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
        properties = schema.get("properties", {})
        unexpected = sorted(set(value.keys()) - set(properties.keys()))
        if unexpected:
            raise SchemaError(f"{path}: unexpected fields {unexpected}")
        for key, sub in properties.items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for idx, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{idx}]")


# ── 原子性写入 ──────────────────────────────────────────────

def atomic_write(path: Path, content: str) -> None:
    """原子性写入：临时文件 → fsync → 重命名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# ── 状态管理器 ──────────────────────────────────────────────

class StateManager:
    """加载、验证、修改和原子性持久化状态。"""

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


# ── 演示 ────────────────────────────────────────────────────

def main() -> None:
    WORK.mkdir(exist_ok=True)

    # 写入 Schema 文件
    schema_dir = WORK / "schemas"
    schema_dir.mkdir(exist_ok=True)
    (schema_dir / "agent_state.schema.json").write_text(json.dumps(STATE_SCHEMA, indent=2) + "\n")
    (schema_dir / "task_board.schema.json").write_text(json.dumps(BOARD_SCHEMA, indent=2) + "\n")

    state_path = WORK / "agent_state.json"
    board_path = WORK / "task_board.json"

    mgr = StateManager(state_path, STATE_SCHEMA)
    board_mgr = StateManager(board_path, BOARD_SCHEMA)

    # 初始化状态和任务板
    initial_state = {
        "schema_version": 1,
        "active_task_id": None,
        "touched_files": [],
        "assumptions": [],
        "blockers": [],
        "next_action": "pick next task",
    }
    initial_board = [
        {
            "id": "T-001",
            "goal": "验证 /signup 请求载荷",
            "owner": "builder",
            "acceptance": ["pytest -x test_app.py::test_signup_rejects_short_password"],
            "status": "todo",
        }
    ]
    mgr.commit(initial_state)
    board_mgr.commit(initial_board)

    # 修改状态——领取任务
    state = mgr.load()
    board = board_mgr.load()
    state["active_task_id"] = board[0]["id"]
    state["next_action"] = "read existing /signup handler"
    mgr.commit(state)

    print("state:", json.dumps(mgr.load(), indent=2))
    print("board:", json.dumps(board_mgr.load(), indent=2))

    # 验证错误写入被拒绝
    bad = dict(state)
    bad["active_task_id"] = "T-bogus"  # 不匹配 T-\d{3,} 模式
    try:
        mgr.commit(bad)
    except SchemaError as exc:
        print("rejected bad write:", exc)


if __name__ == "__main__":
    main()
