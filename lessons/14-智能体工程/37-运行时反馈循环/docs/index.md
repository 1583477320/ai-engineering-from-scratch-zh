# 运行时反馈循环——让智能体相信事实而非自己的预测

> 没看过真实命令输出的智能体会猜测。反馈运行器捕获标准输出、标准错误、退出码和时序到结构化记录中，供下一轮读取。然后智能体对事实做出反应，而不是对自己预测的事实做出反应。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 32（最小工作台）、阶段 14 · 35（初始化脚本）
**预计时间：** ~50 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分运行时反馈和可观测性遥测——各自在智能体生命周期中的角色
- [ ] 构建一个反馈运行器，封装 shell 命令并持久化结构化记录
- [ ] 确定性地截断大输出，使循环保持在词元预算内
- [ ] 在缺少反馈时拒绝推进循环
- [ ] 实现机密脱敏（写入时脱敏而非读取时）和日志轮转策略

---

## 1. 问题

智能体说"正在运行测试"。下一条消息说"所有测试通过"。但现实是，没有一个测试真正跑过。智能体**想象**了输出，或者它运行了命令但从未读取结果，或者它读取了结果但静默截断了失败行。

这个差距是智能体工作台中最隐蔽的失效面之一。没有真实的命令输出，智能体无法对自己的操作进行反馈校正。它在猜测的基础上继续猜测，漂移越来越大。

反馈运行器消除了这个差距。每条命令都经过运行器。每条记录都包含命令、捕获的标准输出和标准错误、退出码、墙钟持续时间和一行智能体注释。智能体在下一轮读取记录。验证门控在任务结束时读取所有记录。

---

## 2. 概念

### 2.1 反馈记录包含什么

```
智能体循环 → run_with_feedback.py → subprocess → stdout/stderr/exit/duration → feedback_record.jsonl
                                                                                      ↓
                                                                                 智能体读取
                                                                                      ↓
                                                                                 验证门控读取
```

| 字段 | 为什么重要 |
|------|----------|
| `command` | 精确的 argv，没有 shell 展开的意外 |
| `stdout_tail` | 最后 N 行，确定性截断 |
| `stderr_tail` | 最后 N 行，与 stdout 分离 |
| `exit_code` | 明确无误的成功信号 |
| `duration_ms` | 发现慢探测和失控进程 |
| `started_at` | 时间戳，用于重放 |
| `agent_note` | 智能体在执行前写的预期 |

### 2.2 截断是确定性的

一个 50 MB 的日志会摧毁循环。运行器确定性地截断头部和尾部，带 `...truncated N lines...` 标记。相同输出总是产生相同记录。不采样；智能体需要看的部分（最终错误、最终摘要）在尾部。

### 2.3 反馈 vs 遥测

遥测（阶段 14 · 23，OTel GenAI 规范）供人类操作员跨时间审查运行。反馈供**本次运行的下一轮**使用。两者共享字段，但位于不同的文件中，保留策略也不同。

### 2.4 没有反馈就不推进

如果运行器在捕获退出码之前出错，记录会携带 `exit_code: null` 和 `error: <reason>`。智能体循环必须拒绝在 `null` 退出码上声称成功。没有退出码，就没有进展。

### 2.5 写入时脱敏，而非读取时

任何涉及 stdout 或 stderr 的记录都可能泄露机密。运行器在追加 JSONL 之前执行脱敏：剥离匹配 `Bearer `、`password=`、`api[_-]?key=`、`AKIA[0-9A-Z]{16}`（AWS）、`xox[baprs]-`（Slack）的行。

**写入时脱敏**：文件在磁盘上就是脱敏的，攻击者无法从中提取秘密。
**读取时脱敏**：文件在磁盘上仍有明文秘密——这是安全漏洞。

### 2.6 日志轮转

单个 `feedback_record.jsonl` 会无限增长。轮转策略：每文件上限 1 MB，溢出时轮转到 `.1`、`.2`，丢弃 `.5`。智能体循环只读取当前文件，所以运行时成本有上限。

---

## 3. 从零实现

### 第 1 步：定义反馈记录

```python
from dataclasses import dataclass, field

@dataclass
class FeedbackRecord:
    command_id: str                # 唯一标识
    parent_command_id: str | None # 重试链中指向父命令
    command: list[str]             # 精确的 argv
    stdout_tail: str               # 截断后的标准输出
    stderr_tail: str               # 截断后的标准错误
    exit_code: int | None         # 退出码（None 表示未完成）
    duration_ms: int              # 持续时间
    started_at: float             # 开始时间
    agent_note: str               # 智能体写入的预期
    error: str | None = None      # 运行错误
    truncations: dict[str, int] = field(default_factory=dict)  # 截断行数
    redactions: dict[str, int] = field(default_factory=dict)    # 脱敏次数
```

### 第 2 步：实现确定性截断

```python
HEAD_LINES = 5
TAIL_LINES = 30

def deterministic_tail(text: str, head=HEAD_LINES, tail=TAIL_LINES):
    """确定性地保留头部和尾部。"""
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text, 0
    cut = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"...truncated {cut} lines..."] + lines[-tail:]), cut
```

### 第 3 步：实现机密脱敏

```python
import re

REDACTION_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+"), "xox-[REDACTED]"),
]

def redact(text: str) -> str:
    """在写入 JSONL 之前脱敏。"""
    if not text:
        return text
    out = text
    for pattern, replacement in REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out
```

### 第 4 步：实现反馈运行器

```python
import subprocess
import time
import uuid
from pathlib import Path

ROTATE_BYTES = 1 * 1024 * 1024  # 1 MB

def run_with_feedback(command, agent_note="", timeout_s=30.0, parent_command_id=None):
    started = time.time()
    command_id = uuid.uuid4().hex[:12]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        out = redact(deterministic_tail(completed.stdout)[0])
        err = redact(deterministic_tail(completed.stderr)[0])
        record = FeedbackRecord(
            stdout_tail=out, stderr_tail=err,
            exit_code=completed.returncode,
            duration_ms=int((time.time() - started) * 1000),
            command_id=command_id, parent_command_id=parent_command_id,
            command=command, started_at=started, agent_note=agent_note,
        )
    except subprocess.TimeoutExpired as exc:
        record = FeedbackRecord(
            stdout_tail="", stderr_tail=str(exc),
            exit_code=None, duration_ms=int((time.time() - started) * 1000),
            error=f"timeout after {timeout_s}s",
            command_id=command_id, parent_command_id=parent_command_id,
            command=command, started_at=started, agent_note=agent_note,
        )
    except FileNotFoundError as exc:
        record = FeedbackRecord(
            stdout_tail="", stderr_tail="",
            exit_code=None, duration_ms=int((time.time() - started) * 1000),
            error=str(exc),
            command_id=command_id, parent_command_id=parent_command_id,
            command=command, started_at=started, agent_note=agent_note,
        )

    maybe_rotate()
    with Path("feedback_record.jsonl").open("a") as fh:
        import json
        fh.write(json.dumps(asdict(record)) + "\n")
    return record
```

### 第 5 步：实现轮转

```python
def maybe_rotate():
    """当前文件超过 1 MB 时轮转。"""
    path = Path("feedback_record.jsonl")
    if not path.exists() or path.stat().st_size < ROTATE_BYTES:
        return
    # 将 .4 移到 .5，.3 移到 .4，...，当前文件移到 .1
    for idx in range(4, 0, -1):
        src = Path(f"feedback_record.jsonl.{idx - 1}" if idx > 1 else "feedback_record.jsonl")
        dst = Path(f"feedback_record.jsonl.{idx}")
        if src.exists():
            if idx == 4 and dst.exists():
                dst.unlink()
            src.rename(dst)
```

### 第 6 步：拒绝无退出码的推进

```python
def loop_can_advance(record: FeedbackRecord) -> bool:
    """没有退出码意味着不能推进循环。"""
    return record.exit_code is not None
```

### 第 7 步：运行演示

```python
# 成功
ok = run_with_feedback(["python3", "-c", "print('hello')"], agent_note="预期输出 hello")

# 失败
fail = run_with_feedback(["python3", "-c", "import sys; sys.exit(2)"],
                          agent_note="第一次尝试，会重试")

# 重试（与父命令关联）
retry = run_with_feedback(["python3", "-c", "print('recovered'); sys.exit(0)"],
                           agent_note="重试非零退出",
                           parent_command_id=fail.command_id)

# 命令不存在
missing = run_with_feedback(["does-not-exist"], agent_note="探测缺失的二进制文件")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Claude Code 的 Bash 工具

Claude Code 的 Bash 工具已经捕获了 stdout、stderr、退出码和持续时间。本课的反馈运行器是框架无关的等价实现——适用于任何智能体产品。

### 4.2 LangGraph 节点

将任何 shell 节点包装在反馈运行器中，使记录在图状态之外持久化。这样即使图状态丢失，反馈记录仍然存在。

```python
from langgraph.graph import StateGraph

def my_node(state):
    record = run_with_feedback(["pytest", "-x"])
    return {"feedback": record}
```

### 4.3 CI 日志管道

将 JSONL 输入 CI 构件存储。审查员可以重放任何命令，无需重跑会话。

### 4.4 实践模式对照

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 写入时脱敏 | 所有场景 | 剥离 Bearer、password、API key、AWS key、Slack token |
| 日志轮转 | 长期运行 | 每文件 1 MB，保留 5 个历史文件 |
| 父命令 ID | 重试链 | `parent_command_id` 跟踪重试关系 |
| 拒绝空退出码 | 安全关键 | `exit_code: null` 时不推进循环 |

---

## 5. 工程最佳实践

### 5.1 反馈循环设计原则

| 原则 | 说明 |
|------|------|
| 写入时脱敏 | 文件在磁盘上就是脱敏的。读取时脱敏是安全漏洞 |
| 确定性截断 | 相同输出总是产生相同记录。不采样，保留头部和尾部 |
| 拒绝空退出码 | `exit_code: null` 意味着不推进循环 |
| 日志轮转 | 单文件上限 1 MB，保留 5 个历史文件 |

### 5.2 中文场景特别建议

- **脱敏模式要覆盖中文密钥格式**——除了英文的 `password=`，也要考虑中文 API 密钥中的 `密钥=`、`token=` 等格式
- **截断后的中文内容注意编码**——截断可能在中文字符中间断开，导致 JSON 编码问题。使用 UTF-8 安全截断
- **agent_note 用中文写**——方便中文团队审查，保持 command 和字段名用英文

### 5.3 踩坑经验

- **读取时脱敏是常见的生产事故**——JSONL 文件中泄露了 Bearer token，因为脱敏只在展示时做。**修复：** 在 `file.write()` 之前脱敏，写入后不保留明文
- **截断边界处理不当导致 JSON 损坏**——截断后的字符串在 JSON 中可能不完整。**修复：** 截断后再做 JSON 序列化
- **不轮转的日志文件无限增长**——一个长期运行的智能体会产生数百 MB 的反馈记录。**修复：** 1 MB 轮转，保留 5 个历史文件

---

## 6. 常见错误

### 错误 1：读取时脱敏而非写入时脱敏

**现象：** 磁盘上的 JSONL 文件中包含了 API 密钥的明文。一个权限配置错误导致文件被外部访问，密钥泄露。

**原因：** 机密脱敏只在展示时执行，而没有在写入时执行。写入时脱敏意味着文件在磁盘上就是安全的；读取时脱敏意味着文件在磁盘上仍然有明文，被攻击者获取后可以直接读取。

**修复：**
```python
# ❌ 读取时脱敏（不安全）
line = file.readline()
clean_line = redact(line)   # 磁盘上仍有明文

# ✓ 写入时脱敏（安全）
record.stdout_tail = redact(raw_stdout)
file.write(json.dumps(record))  # 磁盘上已脱敏
```

### 错误 2：不设退出码检查

**现象：** 命令因超时异常退出，但智能体说"测试通过"。循环继续推进，基于不存在的测试结果做决策。

**原因：** 没有检查 `exit_code` 是否为 `null`。超时、命令不存在、运行器内部错误都应该产生 `exit_code: null`，循环必须拒绝推进。

**修复：**
```python
def loop_can_advance(record):
    if record.exit_code is None:
        return False    # 拒绝推进
    return record.exit_code == 0
```

### 错误 3：不轮转日志

**现象：** 一个长期运行的任务产生了 500 MB 的 `feedback_record.jsonl`。每次加载都要读这个文件，循环越来越慢，最终 OOM。

**原因：** 没有设置文件大小上限和轮转策略。

**修复：** 每文件 1 MB 轮转，最多保留 5 个历史文件。智能体循环只读取当前文件——运行时成本有上限。CI 构件存储完整的轮转集用于审计。

---

## 7. 面试考点

### Q1：反馈运行器解决的核心问题是什么？（难度：⭐）

**参考答案：**
反馈运行器解决的是智能体"想象输出"的问题。智能体说"正在做测试"，下一条消息说"所有测试通过"，但现实是没有一个测试真正跑过。反馈运行器确保每条命令的输出、退出码、持续时间都被结构化捕获，智能体对事实做出反应，而不是对自己预测的事实做出反应。

### Q2：反馈和遥测有什么区别？为什么需要两者？（难度：⭐⭐）

**参考答案：**
反馈（Feedback）是供**本次运行的下一轮**使用的——智能体在循环中读取它来做决策。遥测（Telemetry）是供**人类操作员跨时间审查运行**使用的——用于调试、审计和趋势分析。

两者共享字段（命令、输出、退出码），但位于不同的文件中，保留策略也不同：反馈记录在任务结束后可以清理，遥测数据需要长期保存。阶段 14 · 23（OTel GenAI 规范）覆盖遥测端，本课覆盖反馈端。

### Q3：为什么要在写入时脱敏而不是读取时脱敏？（难度：⭐⭐）

**参考答案：**
写入时脱敏意味着文件在磁盘上就是安全的——攻击者即使拿到文件也无法提取机密。读取时脱敏意味着文件在磁盘上仍然有明文机密——攻击者拿到文件后可以直接读取。

任何涉及 stdout 或 stderr 的记录都可能泄露 Bearer token、API key、密码。脱敏模式应该覆盖常见的机密格式：`Bearer `、`password=`、`api_key=`、`AKIA*`（AWS）、`xox*-`（Slack）。脱敏模式应该每季度审计一次，对照生产环境观察到的机密格式更新。

### Q4：父命令 ID（parent_command_id）解决什么问题？（难度：⭐⭐⭐）

**参考答案：**
当智能体重试一个失败的命令时，如果没有 `parent_command_id`，两次命令在记录中看起来是独立成功的——审核看不到失败历史，验证门控无法评估"是几次失败后成功的"。

`parent_command_id` 指向前一次尝试。如果命令 B 是命令 A 的重试，B 的 `parent_command_id` 就是 A 的 `command_id`。审核员可以追溯重试链。阶段 14 · 40 的手续和阶段 14 · 38 的验证门控都依赖这个链。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 反馈记录 (Feedback Record) | "运行日志" | 带命令、输出、退出码、持续时间的结构化 JSONL 条目 |
| 尾部截断 (Tail Truncation) | "切日志" | 确定性头部+尾部捕获，使记录在词元预算内 |
| 拒绝空退出码 (Refuse-on-null) | "缺数据就不走" | `exit_code` 为 null 时循环不推进 |
| 智能体注释 (Agent Note) | "预期标签" | 智能体在执行前写的一行预测 |
| 反馈/遥测分离 (Telemetry Split) | "两份日志" | 反馈供下一轮使用，遥测供操作员使用 |

---

## 📚 小结

没有真实输出，智能体就活在幻觉里。反馈运行器消除了这个差距——每条命令被封装、捕获结构化输出、确定性地截断、写入时脱敏，并在 `exit_code: null` 时拒绝推进循环。你实现了一个完整的反馈运行器，理解了写入时脱敏为什么比读取时安全、日志轮转为什么防止无限增长，以及父命令 ID 如何让重试链变得透明。

下一课我们将把这些检查组合成一个完整的验证门控——在前几课的规则、状态、初始化、范围检查和反馈记录的基础上，构建一个可集成的验证层。

---

## ✏️ 练习

1. 【实现】为每条记录添加 `cwd` 字段，区分同一命令在不同目录下的运行结果。

2. 【实现】在 JSONL 追加之前添加脱敏步骤，剥离匹配 `Bearer ` 或 `password=` 的行。在夹具记录上测试。

3. 【实现】将 `feedback_record.jsonl` 的大小上限设为 1 MB，通过轮转到 `.1`、`.2` 文件来实现。论证轮转策略。

4. 【实现】添加 `parent_command_id`，使重试链可见：哪条命令产生了下一条命令消费的输入。

5. 【思考】如果将 JSONL 输入一个微型 TUI，突出显示最新的非零退出码。一个有用审查工具应该展示的八个关键特性是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 反馈运行器 | `code/main.py` | 结构化捕获、截断、脱敏、轮转、重试链 |
| 技能提示词 | `outputs/skill-feedback-runner.md` | 生成项目特定的反馈运行器和 JSONL 读取器 |

---

## 📖 参考资料

1. [官方文档] OpenTelemetry GenAI Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
2. [官方文档] Anthropic. "Effective Harnesses for Long-Running Agents". https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
3. [博客] Guardrails AI x MLflow — Deterministic Safety, PII, Quality Validators: https://guardrailsai.com/blog/guardrails-mlflow — 脱敏模式作为回归测试
4. [博客] Aport.io. "Best AI Agent Guardrails 2026: Pre-Action Authorization Compared". https://aport.io/blog/best-ai-agent-guardrails-2026-pre-action-authorization-compared/ — 前/后工具捕获
5. [博客] Andrii Furmanets. "AI Agents in 2026: Practical Architecture for Tools, Memory, Evals, Guardrails". https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails — 可观测性面

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
