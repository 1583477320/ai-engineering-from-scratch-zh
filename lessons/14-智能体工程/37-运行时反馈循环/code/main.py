"""封装 subprocess.run，支持结构化捕获、机密脱敏、日志轮转和命令谱系。

每条命令通过 run_with_feedback 运行。记录携带 argv、脱敏后的 stdout/stderr
尾部、退出码、持续时间、开始时间、智能体注释，以及 command_id/parent_command_id
对，使重试可追溯。JSONL 文件在 1 MB 时轮转，保持加载内存有上限。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
RECORD = HERE / "feedback_record.jsonl"

HEAD_LINES = 5
TAIL_LINES = 30
ROTATE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_ROTATIONS = 5

# 机密脱敏模式——每季度对照生产环境观察到的泄露格式审计
REDACTION_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?key|token)\s*[:=]\s*\S+"),
     r"\1=[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+"), "xox-[REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
     "[REDACTED PRIVATE KEY]"),
]


@dataclass
class FeedbackRecord:
    """一条反馈记录。"""
    command_id: str
    parent_command_id: str | None
    command: list[str]
    stdout_tail: str
    stderr_tail: str
    exit_code: int | None
    duration_ms: int
    started_at: float
    agent_note: str
    error: str | None = None
    truncations: dict[str, int] = field(default_factory=dict)
    redactions: dict[str, int] = field(default_factory=dict)


def redact(text: str) -> tuple[str, int]:
    """写入 JSONL 之前脱敏机密。读取时脱敏是安全漏洞。"""
    if not text:
        return text, 0
    hits = 0
    out = text
    for pattern, replacement in REDACTION_PATTERNS:
        out, n = pattern.subn(replacement, out)
        hits += n
    return out, hits


def deterministic_tail(text: str, head: int = HEAD_LINES, tail: int = TAIL_LINES) -> tuple[str, int]:
    """确定性截断：保留头部和尾部，用标记表示中间被截断的行数。"""
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text, 0
    cut = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"...truncated {cut} lines..."] + lines[-tail:]), cut


def _process_capture(text: str) -> tuple[str, int, int]:
    """先截断，再脱敏。返回 (text, cut_lines, redaction_hits)。"""
    tailed, cut = deterministic_tail(text)
    redacted, hits = redact(tailed)
    return redacted, cut, hits


def maybe_rotate() -> None:
    """当前文件超过 ROTATE_BYTES 时轮转。保留 MAX_ROTATIONS 个历史文件。"""
    if not RECORD.exists() or RECORD.stat().st_size < ROTATE_BYTES:
        return
    for idx in range(MAX_ROTATIONS, 0, -1):
        src = RECORD.with_suffix(RECORD.suffix + (f".{idx - 1}" if idx > 1 else ""))
        if src == RECORD:
            src = RECORD
        dst = RECORD.with_suffix(RECORD.suffix + f".{idx}")
        if src.exists():
            if idx == MAX_ROTATIONS and dst.exists():
                dst.unlink()
            try:
                src.rename(dst)
            except FileNotFoundError:
                pass


def run_with_feedback(
    command: list[str],
    agent_note: str = "",
    timeout_s: float = 30.0,
    parent_command_id: str | None = None,
) -> FeedbackRecord:
    """运行命令并捕获结构化反馈记录。"""
    started = time.time()
    command_id = uuid.uuid4().hex[:12]
    base_kwargs = dict(
        command_id=command_id,
        parent_command_id=parent_command_id,
        command=command,
        started_at=started,
        agent_note=agent_note,
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        out, cut_out, red_out = _process_capture(completed.stdout)
        err, cut_err, red_err = _process_capture(completed.stderr)
        record = FeedbackRecord(
            stdout_tail=out, stderr_tail=err,
            exit_code=completed.returncode,
            duration_ms=int((time.time() - started) * 1000),
            truncations={"stdout": cut_out, "stderr": cut_err},
            redactions={"stdout": red_out, "stderr": red_err},
            **base_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial_err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        out, cut_out, red_out = _process_capture(partial_out)
        err, cut_err, red_err = _process_capture(partial_err)
        record = FeedbackRecord(
            stdout_tail=out, stderr_tail=err,
            exit_code=None,
            duration_ms=int((time.time() - started) * 1000),
            error=f"timeout after {timeout_s}s",
            truncations={"stdout": cut_out, "stderr": cut_err},
            redactions={"stdout": red_out, "stderr": red_err},
            **base_kwargs,
        )
    except FileNotFoundError as exc:
        record = FeedbackRecord(
            stdout_tail="", stderr_tail="",
            exit_code=None,
            duration_ms=int((time.time() - started) * 1000),
            error=str(exc),
            **base_kwargs,
        )

    maybe_rotate()
    with RECORD.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return record


def loop_can_advance(record: FeedbackRecord) -> bool:
    """退出码为 None 时拒绝推进循环。"""
    return record.exit_code is not None


def load_all() -> list[FeedbackRecord]:
    """读取当前及已轮转的文件，使父命令谱系在轮转后仍然可追溯。"""
    def _rotation_key(p: Path) -> int:
        suffix = p.name[len(RECORD.name):]
        if not suffix:
            return 0  # 当前文件
        try:
            return int(suffix.lstrip("."))
        except ValueError:
            return 99
    paths = sorted(HERE.glob(RECORD.name + "*"), key=_rotation_key, reverse=True)
    by_id: dict[str, FeedbackRecord] = {}
    for path in paths:
        try:
            text = path.read_text()
        except FileNotFoundError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = FeedbackRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            by_id[record.command_id] = record  # 当前文件最后加载，优先级最高
    return list(by_id.values())


def retry_chain(command_id: str) -> list[FeedbackRecord]:
    """通过 parent_command_id 指针追溯重试链。"""
    records = {r.command_id: r for r in load_all()}
    chain: list[FeedbackRecord] = []
    cursor: str | None = command_id
    while cursor and cursor in records:
        chain.append(records[cursor])
        cursor = records[cursor].parent_command_id
    return list(reversed(chain))


def main() -> None:
    # 清除旧的测试文件
    for path in HERE.glob("feedback_record.jsonl*"):
        path.unlink()

    # 成功的命令
    ok = run_with_feedback(["python3", "-c", "print('hello')"], agent_note="预期输出 hello")

    # 包含机密的命令（测试脱敏）
    leak = run_with_feedback(
        ["python3", "-c",
         "print('Authorization: Bearer ya29.AbCdEf'); print('password=hunter2'); print('AKIAIOSFODNN7EXAMPLE')"],
        agent_note="预期机密被脱敏"
    )

    # 失败的命令
    fail = run_with_feedback(["python3", "-c", "import sys; sys.exit(2)"], agent_note="第一次尝试，会重试")

    # 重试（与父命令关联）
    retry = run_with_feedback(
        ["python3", "-c", "print('recovered'); import sys; sys.exit(0)"],
        agent_note="重试非零退出",
        parent_command_id=fail.command_id,
    )

    # 命令不存在
    missing = run_with_feedback([shlex.split("does-not-exist")[0]], agent_note="探测缺失的二进制文件")

    for label, rec in (("ok", ok), ("leak", leak), ("fail", fail), ("retry", retry), ("missing", missing)):
        print(f"{label}: cid={rec.command_id} parent={rec.parent_command_id or '-'} exit={rec.exit_code} "
              f"duration_ms={rec.duration_ms} redactions={rec.redactions or '-'}")
        if rec.error:
            print(f"  error: {rec.error}")
        if rec.stdout_tail and "REDACTED" in rec.stdout_tail:
            print(f"  stdout after redaction: {rec.stdout_tail!r}")

    chain = retry_chain(retry.command_id)
    print(f"\nretry chain for {retry.command_id}: {[r.command_id for r in chain]} (oldest -> newest)")
    print(f"{len(load_all())} records persisted in {RECORD.name}")


if __name__ == "__main__":
    main()
