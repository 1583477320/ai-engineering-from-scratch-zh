# 智能体初始化脚本——把"启动税"只交一次

> 每次冷启动都交一笔税。智能体读同样的文件、重试同样的探测、重新发现同样的路径。初始化脚本把这笔税只交一次，把答案写进状态。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 32（最小工作台）、阶段 14 · 34（仓库记忆）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别智能体每次会话都不应该重复做的工作——什么是"启动税"
- [ ] 构建一个确定性初始化脚本，探测运行时、依赖、仓库健康状态
- [ ] 将探测结果持久化，让智能体读取而不是重新运行检查
- [ ] 设计"大声失败、快速失败、在一个地方失败"的初始化失败策略

---

## 1. 问题

打开一个会话。智能体猜测 Python 版本。猜测测试命令。列出仓库根目录五次来找到入口点。尝试导入一个未安装的包。问用户配置文件在哪里。等到它做出真正的编辑时，一万个词元已经花在了本应由一个脚本完成的设置工作上。

这就是**启动税**——智能体每次会话都要重新发现显而易见的东西。Python 版本不会变。测试命令不会变。配置文件路径不会变。但智能体不知道这些，因为它没有记忆。

解决方案是一个初始化脚本，在智能体做任何事之前运行，把探测结果写进 `init_report.json`，智能体在启动时读取它。

---

## 2. 概念

### 2.1 初始化脚本探测什么

```
会话启动 → init_agent.py → 探测运行时/依赖/路径/环境/测试 → init_report.json
                                                                    │
                                                              健康？─┬─ 是 → 智能体循环
                                                                    └─ 否 → 大声失败，停止，报告给人类
```

| 探测项 | 为什么重要 |
|--------|----------|
| 运行时版本 | 错误的 Python 或 Node 版本会导致静默的版本兼容问题 |
| 依赖可用性 | 缺失的包后续修复成本是现在的十倍 |
| 测试命令 | 智能体必须知道如何验证；如果命令缺失，工作台就坏了 |
| 仓库路径 | 硬编码路径会漂移；解析一次并固定 |
| 环境变量 | 缺少 `OPENAI_API_KEY` 是失败面，不是运行时谜题 |
| 状态 + 任务板新鲜度 | 崩溃会话留下的过时状态是隐患 |
| 最后已知正常提交 | 会话结束时交接 diff 的锚点 |

### 2.2 大声失败，快速失败，在一个地方失败

探测失败意味着停止并报告给人类。不要"让智能体自己想办法"。初始化的全部意义就是在工作台坏了的时候拒绝启动。

### 2.3 幂等性

连续运行两次。第二次除了刷新时间戳外应该是空操作。幂等性使得脚本可以接入 CI、钩子或预任务斜杠命令。

### 2.4 初始化 vs 启动规则

规则（阶段 14 · 33）描述"行动前必须满足什么"。初始化是建立"那些规则可以被检查"的脚本。没有初始化的规则变成"请小心"。没有规则的初始化变成精致的失败。

---

## 3. 从零实现

### 第 1 步：定义探测函数

每个探测返回 `(name, status, detail)`。用装饰器自动计时。

```python
import time
from dataclasses import dataclass

@dataclass
class Probe:
    name: str
    status: str       # "pass", "warn", "fail"
    detail: str
    duration_ms: int = 0

PROBE_BUDGET_SECONDS = 3.0

def timed(probe_fn):
    """自动计时装饰器。超过预算的探测标记为 warn。"""
    def wrapper(*args, **kwargs) -> Probe:
        started = time.time()
        result = probe_fn(*args, **kwargs)
        result.duration_ms = int((time.time() - started) * 1000)
        if result.duration_ms > PROBE_BUDGET_SECONDS * 1000 and result.status == "pass":
            result.status = "warn"
            result.detail += f" (slow: {result.duration_ms}ms)"
        return result
    return wrapper
```

### 第 2 步：实现探测函数

```python
import importlib.util
import os
import shutil
import sys

REQUIRED_PYTHON = (3, 10)
REQUIRED_DEPS = ["json", "dataclasses"]
REQUIRED_TEST_COMMAND = "python3"

@timed
def probe_runtime() -> Probe:
    major, minor = sys.version_info[:2]
    if (major, minor) >= REQUIRED_PYTHON:
        return Probe("runtime", "pass", f"python {major}.{minor}")
    return Probe("runtime", "fail", f"need >= {REQUIRED_PYTHON}, have {major}.{minor}")

@timed
def probe_dependencies() -> Probe:
    missing = [dep for dep in REQUIRED_DEPS if importlib.util.find_spec(dep) is None]
    if missing:
        return Probe("dependencies", "fail", f"missing: {missing}")
    return Probe("dependencies", "pass", f"all of {REQUIRED_DEPS} importable")

@timed
def probe_test_command() -> Probe:
    if shutil.which(REQUIRED_TEST_COMMAND):
        return Probe("test_command", "pass", f"{REQUIRED_TEST_COMMAND} on PATH")
    return Probe("test_command", "fail", f"{REQUIRED_TEST_COMMAND} not on PATH")

@timed
def probe_env() -> Probe:
    required = ["OPENAI_API_KEY"]  # 按项目需求配置
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return Probe("env", "fail", f"missing: {missing}")
    return Probe("env", "pass", "all env vars present")
```

### 第 3 步：锁文件与 TTL 缓存

```python
import hashlib
import json
from pathlib import Path

LOCK_PATH = Path("prereqs.lock")
LOCK_TTL_SECONDS = 24 * 60 * 60  # 24 小时

def deps_fingerprint() -> str:
    """根据依赖配置生成指纹。"""
    h = hashlib.sha256()
    h.update(str(sorted(REQUIRED_DEPS)).encode())
    h.update(REQUIRED_TEST_COMMAND.encode())
    return h.hexdigest()[:16]

def lock_is_fresh() -> bool:
    """检查锁文件是否在 TTL 内且指纹匹配。"""
    if not LOCK_PATH.exists():
        return False
    try:
        lock = json.loads(LOCK_PATH.read_text())
    except json.JSONDecodeError:
        return False
    if lock.get("fingerprint") != deps_fingerprint():
        return False
    age = time.time() - lock.get("written_at", 0)
    return age < LOCK_TTL_SECONDS

def write_lock() -> None:
    """写入锁文件。"""
    LOCK_PATH.write_text(
        json.dumps({"fingerprint": deps_fingerprint(), "written_at": time.time()}, indent=2) + "\n"
    )
```

### 第 4 步：主函数

```python
import argparse

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="忽略锁文件，运行所有探测")
    args = parser.parse_args(argv)

    # 检查锁文件缓存
    if not args.no_cache and lock_is_fresh():
        print(f"锁文件有效（TTL {LOCK_TTL_SECONDS}s）；跳过探测")
        return 0

    # 运行所有探测
    probes = [probe_runtime(), probe_dependencies(), probe_test_command(), probe_env()]
    report = {
        "timestamp": time.time(),
        "probes": [{"name": p.name, "status": p.status, "detail": p.detail} for p in probes],
        "ok": all(p.status != "fail" for p in probes),
    }

    # 写入报告
    Path("init_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # 打印结果
    for p in probes:
        print(f"  {p.name:<15} {p.status:>4}  {p.detail}")

    if not report["ok"]:
        print("\n初始化失败；拒绝启动智能体", file=sys.stderr)
        return 1

    write_lock()
    print("\n初始化成功（锁已刷新）")
    return 0
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Claude Code 的钩子

Claude Code 的 `pre-task` 钩子可以调用初始化脚本，如果失败则拒绝启动智能体。

```json
{
  "hooks": {
    "pre-task": "python3 tools/init_agent.py"
  }
}
```

### 4.2 GitHub Actions

```yaml
jobs:
  setup-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python3 tools/init_agent.py
  agent:
    needs: setup-agent
    runs-on: ubuntu-latest
    steps:
      - run: # 智能体任务
```

初始化脚本作为独立的 job 运行；智能体 job 依赖它。

### 4.3 Docker 入口点

智能体容器在 exec 运行时之前运行初始化脚本；失败时日志会浮出。

```dockerfile
ENTRYPOINT ["python3", "tools/init_agent.py"]
CMD ["--", "agent-runtime"]
```

### 4.4 实践模式对照

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| 最后已知正常提交锚定 | 长期运行的智能体 | 对比当前提交与 LKG 文件，差异超过预算时拒绝启动 |
| 锁文件 + TTL | 频繁启动的会话 | 首次探测成功后写入锁文件，24 小时内跳过昂贵探测 |
| 热路径无网络/无 LLM | 所有初始化 | 探测是确定性管道，不调用外部服务 |

---

## 5. 工程最佳实践

### 5.1 初始化脚本设计原则

| 原则 | 说明 |
|------|------|
| 大声失败 | 探测失败时停止并报告，不要静默回退 |
| 幂等性 | 连续运行两次结果相同（除时间戳外） |
| 确定性 | 热路径不调用网络、LLM 或外部服务 |
| 有时间预算 | 每个探测超过 3 秒就是工作台气味 |

### 5.2 中文场景特别建议

- **探测报告使用英文字段名**——`init_report.json` 的字段名保持英文，便于跨工具兼容
- **探测详情可以用中文**——`detail` 字段用中文写，方便团队审查
- **环境变量探测要考虑中文路径**——某些工具在中文 Windows 路径下会有编码问题

### 5.3 踩坑经验

- **不要在初始化中调用 LLM**——调用 LLM 来分类失败的探测不是探测，是工作流。保持初始化确定性
- **锁文件的 TTL 不要太长**——24 小时是合理默认值。太长会导致环境变化后仍然跳过探测
- **LKG 差异预算是关键**——如果当前提交与最后已知正常提交的差异超过 50 个文件，拒绝启动。防止跨会话漂移累积

---

## 6. 常见错误

### 错误 1：初始化脚本中调用 LLM 或外部服务

**现象：** 初始化脚本在探测失败时调用 GPT-4 来"分析"问题。结果是初始化变成了一个不确定的工作流，有时成功有时失败，没有一致的行为。

**原因：** 探测应该是确定性的管道。调用 LLM 的不是探测，是工作流。

**修复：** 热路径中的所有探测必须是纯函数——给定相同输入，返回相同输出。如果探测需要超过 3 秒，要么移到初始化之外，要么缓存结果。

### 错误 2：初始化失败时静默回退

**现象：** 探测失败时打印一条警告然后继续。智能体在损坏的工作台上运行，产生难以调试的错误。

**原因：** 初始化的全部意义是在工作台坏了的时候拒绝启动。静默回退违背了这个目的。

**修复：** 任何 `block` 级别的探测失败时，脚本退出码非零，拒绝启动智能体。

### 错误 3：锁文件 TTL 设置过长

**现象：** 锁文件 TTL 设置为 7 天。第二天有人安装了新依赖，但锁文件仍然有效，探测被跳过。智能体在缺少依赖的环境中运行。

**原因：** TTL 过长导致环境变化后探测仍然被跳过。

**修复：** 默认 24 小时。锁文件包含依赖配置的指纹——如果配置变化，指纹不匹配，锁失效。

---

## 7. 面试考点

### Q1：什么是"启动税"？初始化脚本如何消除它？（难度：⭐）

**参考答案：**
启动税是智能体每次会话都要重新发现显而易见的东西所消耗的词元——Python 版本、测试命令、配置文件路径。这些信息在会话之间不会变化，但智能体没有跨会话记忆，每次都要重新探测。

初始化脚本在智能体做任何事之前运行，把探测结果写进 `init_report.json`。智能体在启动时读取这个文件，而不是重新运行检查。这把"每次交税"变成了"只交一次"。

### Q2：初始化脚本的失败策略是什么？为什么不能静默回退？（难度：⭐⭐）

**参考答案：**
失败策略是"大声失败、快速失败、在一个地方失败"。任何 `block` 级别的探测失败时，脚本退出码非零，拒绝启动智能体。

不能静默回退的原因：初始化的全部意义就是在工作台坏了的时候拒绝启动。如果静默回退，智能体在损坏的环境中运行，产生难以调试的错误。静默回退违背了初始化的目的。

### Q3：最后已知正常（LKG）提交锚定如何工作？（难度：⭐⭐）

**参考答案：**
在 `last_known_good.json` 中记录最后一次成功合并的提交 SHA。初始化时对比当前 HEAD 与 LKG 的差异。如果差异超过预算（默认 50 个文件），拒绝启动，要求人工确认新的基线。

这防止了跨会话漂移累积——每次会话都锚定在同一个 LKG 上，而不是在前一个会话的漂移基础上继续漂移。Cloudflare 的 AI 代码审查使用同样的模式来限定审查智能体的范围。

### Q4：锁文件 + TTL 缓存的原理是什么？和 Docker 层缓存有什么相似之处？（难度：⭐⭐⭐）

**参考答案：**
首次探测成功后写入 `prereqs.lock`，包含依赖配置的指纹和时间戳。后续运行检查锁文件：如果在 TTL 内（默认 24 小时）且指纹匹配，跳过所有探测。

这和 Docker 层缓存的原理完全相同：幂等探测 + 内容哈希 = 跳过。Docker 用文件内容哈希决定是否重建层；我们用依赖配置哈希决定是否跳过探测。两者都避免了重复执行昂贵但结果不变的操作。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 探测 (Probe) | "检查" | 返回 `(name, status, detail)` 的确定性函数 |
| 初始化报告 (Init Report) | "设置输出" | 写在状态旁边的 JSON，包含探测结果 |
| 幂等性 (Idempotency) | "可以重复运行" | 连续两次运行产生相同报告（除时间戳外） |
| 大声失败 (Fail Loud) | "不要吞掉错误" | 停止并报告给人类；不静默回退 |
| 启动税 (Setup Tax) | "引导成本" | 智能体每次会话重新发现显而易见的东西所消耗的词元 |

---

## 📚 小结

每次冷启动都交一笔税——智能体重新探测 Python 版本、测试命令、依赖状态。初始化脚本把这笔税只交一次，把答案写进 `init_report.json`。你实现了一个确定性探测系统，理解了幂等性、锁文件 TTL 缓存、最后已知正常提交锚定，以及"大声失败"的失败策略。

初始化脚本与上一课的规则集配合工作：规则描述"行动前必须满足什么"，初始化建立"那些规则可以被检查"的基础。没有初始化的规则是空话，没有规则的初始化是精致的失败。

---

## ✏️ 练习

1. 【实现】添加一个探测：对比当前提交与最后已知正常提交，如果差异超过 50 个文件则拒绝启动。

2. 【实现】让脚本写入 `prereqs.lock` 文件，如果锁文件超过 7 天则拒绝启动。

3. 【实现】添加 `--fix` 标志，自动安装缺失的开发依赖，但不修改运行时依赖。

4. 【实验】将探测从硬编码函数改为 YAML 注册表。论证这种取舍。

5. 【思考】为每个探测添加时间预算。超过 3 秒的探测是工作台气味——应该移到初始化之外还是缓存结果？写 200 字以内的分析。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 初始化脚本 | `code/main.py` | 确定性探测 + 锁文件缓存 + LKG 锚定 |
| 技能提示词 | `outputs/skill-init-script.md` | 面谈项目，将设置工作分类为探测，生成初始化脚本 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Effective Harnesses for Long-Running Agents". https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
2. [官方文档] GitHub Actions Composite Actions: https://docs.github.com/en/actions/sharing-automations/creating-actions/creating-a-composite-action
3. [博客] Augment Code. "How to Build Your AGENTS.md (2026)". https://www.augmentcode.com/guides/how-to-build-agents-md
4. [博客] Codex Blog. "Codex CLI Context Compaction". https://codex.danielvaughan.com/2026/03/31/codex-cli-context-compaction-architecture/ — 会话启动作为压缩感知的初始化
5. [博客] microservices.io. "GenAI Dev Platform: Guardrails". https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html — pre-commit + CI 检查作为初始化

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
