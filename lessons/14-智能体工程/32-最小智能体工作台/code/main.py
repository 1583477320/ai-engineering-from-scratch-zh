"""最小智能体工作台——三个文件（AGENTS.md、agent_state.json、task_board.json）。

写入三个文件并运行一轮智能体循环：
1. 读取 agent_state.json
2. 如果状态为空，从 task_board.json 拉取下一个任务
3. 在范围内修改文件
4. 写回更新的状态

运行：python3 code/main.py
重跑：第二轮从第一轮停下的地方继续。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent / "workdir"


# ── AGENTS.md ────────────────────────────────────────────────

AGENTS_MD = """# AGENTS.md

This repo runs with a workbench. Read these before acting:

1. `agent_state.json` — where the last session stopped.
2. `task_board.json` — what is in flight, what is next.
3. `docs/agent-rules.md` — startup, scope, definition of done (load on demand).

Definition of done: the task referenced by `agent_state.active_task_id` has
`status == "done"` on `task_board.json` and the verification command listed in
its `acceptance` has exited 0.

Verification command: `python3 -m pytest -x`
""".lstrip()


# ── 数据结构 ────────────────────────────────────────────────

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
    owner: str
    acceptance: list[str]
    status: str = "todo"


# ── 读写函数 ────────────────────────────────────────────────

def write_initial(state_path: Path, board_path: Path, agents_path: Path) -> None:
    """首次运行时写入初始文件。"""
    if not agents_path.exists():
        agents_path.write_text(AGENTS_MD)
    if not state_path.exists():
        state_path.write_text(json.dumps(asdict(AgentState(active_task_id=None)), indent=2) + "\n")
    if not board_path.exists():
        board = [
            Task(id="T-001", goal="为 /signup 添加输入验证", owner="builder",
                 acceptance=["pytest test_app.py::test_signup_rejects_short_password"]),
            Task(id="T-002", goal="记录新的 /signup 接口契约", owner="builder",
                 acceptance=["docs/api.md 提及 /signup 约束"]),
        ]
        board_path.write_text(json.dumps([asdict(t) for t in board], indent=2) + "\n")


def load_state(state_path: Path) -> AgentState:
    raw = json.loads(state_path.read_text())
    return AgentState(**raw)


def load_board(board_path: Path) -> list[Task]:
    return [Task(**t) for t in json.loads(board_path.read_text())]


def save_state(state_path: Path, state: AgentState) -> None:
    state_path.write_text(json.dumps(asdict(state), indent=2) + "\n")


def save_board(board_path: Path, board: list[Task]) -> None:
    board_path.write_text(json.dumps([asdict(t) for t in board], indent=2) + "\n")


# ── 智能体循环核心 ──────────────────────────────────────────

def run_one_turn(state: AgentState, board: list[Task]) -> tuple[AgentState, list[Task]]:
    """执行一轮智能体循环。"""
    # 如果当前没有活动任务，从任务板拉取下一个 todo
    if state.active_task_id is None:
        nxt = next((t for t in board if t.status == "todo"), None)
        if nxt is None:
            state.next_action = "no work on the board, idle"
            return state, board
        nxt.status = "in_progress"
        state.active_task_id = nxt.id
        state.next_action = f"start work on {nxt.id}: {nxt.goal}"
        return state, board

    # 如果活动任务在任务板上不存在
    active = next((t for t in board if t.id == state.active_task_id), None)
    if active is None:
        state.active_task_id = None
        state.next_action = "active task missing from board; resetting"
        return state, board

    # 子轮 1：修改源文件
    if "app.py" not in state.touched_files:
        state.touched_files.append("app.py")
        state.next_action = f"add test for {active.id} acceptance"
        return state, board

    # 子轮 2：添加测试文件
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


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    ROOT.mkdir(exist_ok=True)
    state_path = ROOT / "agent_state.json"
    board_path = ROOT / "task_board.json"
    agents_path = ROOT / "AGENTS.md"

    write_initial(state_path, board_path, agents_path)
    state = load_state(state_path)
    board = load_board(board_path)

    print("本轮开始前：")
    print(f"  活动任务  : {state.active_task_id}")
    print(f"  下一步    : {state.next_action!r}")
    print(f"  任务板待办: {[t.id for t in board if t.status == 'todo']}")

    state, board = run_one_turn(state, board)
    save_state(state_path, state)
    save_board(board_path, board)

    print("\n本轮结束后：")
    print(f"  活动任务  : {state.active_task_id}")
    print(f"  修改的文件: {state.touched_files}")
    print(f"  下一步    : {state.next_action!r}")
    print(f"  任务板状态: {[(t.id, t.status) for t in board]}")

    # 提示重跑
    print("\n---")
    print("重跑此脚本可以看到第二轮从本轮停下的地方继续。")


if __name__ == "__main__":
    main()
