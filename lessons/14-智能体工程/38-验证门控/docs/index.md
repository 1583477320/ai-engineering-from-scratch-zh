# 验证门控——智能体不能给自己的作业打分

> 智能体不能标记自己的工作为完成。验证门控读取范围契约、反馈日志、规则报告和差异，回答一个问题：这个任务真的完成了吗？如果门控说不，任务就没完成，不管聊天记录里怎么说。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 33（规则）、阶段 14 · 36（范围）、阶段 14 · 37（反馈）
**预计时间：** ~55 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 39（审查员智能体）— 门控通过后，审查员进行定性评判

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 将验证门控定义为工作台工件上的确定性函数
- [ ] 将规则报告、范围报告、反馈记录和差异合并为单一裁决
- [ ] 生成 `verification_report.json`——审查员和 CI 都能读取
- [ ] 拒绝在任��� block 级失败下推进任务，没有例外
- [ ] 实现签名覆盖日志——用 HMAC 签名的覆盖条目，而非口头豁免

---

## 1. 问题

智能体太容易声明成功了。三种失败形态最常见：

- **"看起来不错。"** 模型读了自己的差异，决定它是正确的
- **"测试通过了。"** 说得很有信心，但没有测试实际运行的记录
- **"验收条件满足。"** 验收条件被松散地解释为"任何看起来像完成的东西"

工作台的修复方案是一个单一的验证门控——读取智能体已经产生的工件，做出判断。门控是确定性的。门控在版本控制中。门控接入 CI。智能体无法贿赂它。

---

## 2. 概念

### 2.1 门控检查什么

```
差异 → verify_agent.py → verification_report.json → 通过？→ 是 → 审查员智能体
范围报告 ↗                                     → 否 → 拒绝完成 + 报告给人类
规则报告 ↗
反馈记录 ↗
```

| 检查项 | 来源工件 | 严重性 |
|-------|---------|--------|
| 所有验收命令都运行了 | `feedback_record.jsonl` | block |
| 所有验收命令退出码为 0 | `feedback_record.jsonl` | block |
| 范围检查无禁止写入 | `scope_report.json` | block |
| 范围检查无越界写入 | `scope_report.json` | block 或 warn |
| 所有 block 级规则通过 | `rule_report.json` | block |
| 反馈中无 `null` 退出码 | `feedback_record.jsonl` | block |
| 修改的文件匹配范围 | 差异 + scope_report | warn |

`warn` 级发现标注裁决；`block` 级发现阻止 `passed: true`。

### 2.2 确定性，而非概率性

同一个工件集必须每次产生相同的裁决。**没有 LLM 评判器。** LLM 评判器属于审查员一侧（阶段 14 · 39），那里的目标是定性评估，而非状态判断。

### 2.3 一个报告，一个路径

每个任务关闭时门控生成一个 `verification_report.json`，写入 `outputs/verification/<task_id>.json`。CI 消费同一个路径。多个门控使用不同路径会分裂事实来源。

### 2.4 拒绝，没有例外

Block 级发现不能被智能体覆盖。它们只能被人类覆盖，且必须记录 `override_reason` 和 `overridden_by` 用户 ID。覆盖是签名的变更，不是智能体的决策。

### 2.5 防御纵深，而非单点门控

```
预提交钩子 → CI 状态检查 → 预工具授权钩子 → 预合并门控
```

每层都是确定性的，一层的失败会被下一层捕获。`microservices.io` 2026 年 3 月的操作手册明确指出：预提交钩子是不可绕过的——因为它不依赖智能体遵守指令。验证门控位于 CI / 预合并层。

---

## 3. 从零实现

### 第 1 步：定义数据结构和工件

```python
from dataclasses import dataclass, field

@dataclass
class Finding:
    code: str          # 如 "acceptance.missing"
    severity: str      # block | warn
    detail: str

@dataclass
class Artifacts:
    task_id: str
    acceptance_commands: list[str]          # 验收命令列表
    feedback: list[dict]                    # 反馈记录
    scope_report: dict                      # 范围检查报告
    rule_report: list[dict]                 # 规则检查报告
    coverage_report: dict | None = None     # 覆盖率报告
    head_commit: str = ""

@dataclass
class VerdictReport:
    task_id: str
    passed: bool
    strict: bool
    findings: list[Finding] = field(default_factory=list)
    coverage: dict | None = None
    head_commit: str = ""
```

### 第 2 步：实现验收命令检查

```python
def _acceptance_findings(art: Artifacts) -> list[Finding]:
    """检查验收命令是否运行且退出码为 0。"""
    findings = []
    commands_run = [str(rec.get("command")) for rec in art.feedback]

    # 检查命令是否运行
    for cmd in art.acceptance_commands:
        if cmd not in commands_run:
            findings.append(Finding("acceptance.missing", "block", f"never ran: {cmd}"))

    # 检查退出码
    for rec in art.feedback:
        cmd_str = str(rec.get("command"))
        if rec.get("exit_code") is None:
            findings.append(Finding("feedback.null_exit", "block", f"missing exit for {cmd_str}"))
        elif rec.get("exit_code") != 0 and cmd_str in art.acceptance_commands:
            findings.append(Finding("acceptance.failed", "block",
                                     f"exit {rec.get('exit_code')} on {cmd_str}"))
    return findings
```

### 第 3 步：实现范围检查和规则检查

```python
def _scope_findings(art: Artifacts) -> list[Finding]:
    findings = []
    if art.scope_report.get("forbidden_writes"):
        findings.append(Finding("scope.forbidden", "block",
                                 f"forbidden writes: {art.scope_report['forbidden_writes']}"))
    if art.scope_report.get("off_scope_writes"):
        findings.append(Finding("scope.off_scope", "warn",
                                 f"off-scope writes: {art.scope_report['off_scope_writes']}"))
    return findings

def _rule_findings(art: Artifacts) -> list[Finding]:
    return [Finding("rule.failed", "block", f"rule failed: {row.get('slug')}")
            for row in art.rule_report if not row.get("passed")]
```

### 第 4 步：实现覆盖率检查

```python
COVERAGE_FLOOR_DEFAULT = 0.80
COVERAGE_REGRESSION_DELTA = 0.01

def _coverage_findings(art: Artifacts, floor: float) -> list[Finding]:
    """Anthropic Hybrid Norm：可验证奖励（覆盖率）+ 评级评判。"""
    findings = []
    if not art.coverage_report:
        findings.append(Finding("coverage.missing", "warn",
                                 "no coverage_report.json; cannot enforce floor"))
        return findings
    current = float(art.coverage_report.get("current", 0.0))
    previous = float(art.coverage_report.get("previous", current))
    if current < floor:
        findings.append(Finding("coverage.below_floor", "block",
                                 f"coverage {current:.2%} below floor {floor:.0%}"))
    delta = previous - current
    if delta > COVERAGE_REGRESSION_DELTA:
        findings.append(Finding("coverage.regression", "block",
                                 f"dropped {delta:.2%} (prev {previous:.2%} -> {current:.2%})"))
    elif delta > 0:
        findings.append(Finding("coverage.minor_regression", "warn",
                                 f"dropped {delta:.2%}"))
    return findings
```

### 第 5 步：实现主验证函数

```python
def verify(art: Artifacts, strict=False, coverage_floor=COVERAGE_FLOOR_DEFAULT) -> VerdictReport:
    findings = (_acceptance_findings(art) + _scope_findings(art)
                + _rule_findings(art) + _coverage_findings(art, coverage_floor))

    if strict:
        # --strict：所有 warn 升级为 block。仅在发布分支开启
        findings = [Finding(f.code, "block" if f.severity == "warn" else f.severity, f.detail)
                    for f in findings]

    blocking = [f for f in findings if f.severity == "block"]
    return VerdictReport(task_id=art.task_id, passed=not blocking,
                         strict=strict, findings=findings,
                         coverage=art.coverage_report, head_commit=art.head_commit)
```

### 第 6 步：实现签名覆盖日志

```python
import hmac, hashlib, json, time

def _load_override_secret() -> str:
    """从环境变量加载覆盖密钥。生产环境从密钥管理器读取。"""
    secret = os.environ.get("VERIFY_OVERRIDE_SECRET")
    if secret:
        return secret
    raise RuntimeError("VERIFY_OVERRIDE_SECRET is unset")

def record_override(task_id, finding_code, reason, user_id, head_commit) -> dict:
    """追加一条签名覆盖条目。五个字段都必需。"""
    if not all([task_id, finding_code, reason, user_id, head_commit]):
        raise ValueError("override requires all five fields")
    payload = {"task_id": task_id, "finding_code": finding_code,
               "reason": reason, "user_id": user_id,
               "head_commit": head_commit, "ts": time.time()}
    # HMAC-SHA256 签名
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = hmac.new(
        _load_override_secret().encode(), canonical, hashlib.sha256
    ).hexdigest()[:32]
    with Path("overrides.jsonl").open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    return payload
```

### 第 7 步：运行演示

```python
def main():
    accept = ["pytest -x test_app.py::test_signup_rejects_short_password"]

    # T-001：干净通过
    clean = Artifacts(task_id="T-001", acceptance_commands=accept,
        feedback=[{"command": accept[0], "exit_code": 0}],
        scope_report={"forbidden_writes": [], "off_scope_writes": []},
        rule_report=[{"slug": "done/tests-pass", "passed": True}],
        coverage_report={"current": 0.84, "previous": 0.85},
        head_commit="a1b2c3d")

    # T-002：范围蔓延
    creep = Artifacts(task_id="T-002", acceptance_commands=accept,
        feedback=[{"command": accept[0], "exit_code": 0}],
        scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["README.md"]},
        rule_report=[{"slug": "forbidden/no-release-script-edits", "passed": False}],
        coverage_report={"current": 0.62, "previous": 0.80},
        head_commit="b2c3d4e")

    # T-003：缺验收
    no_accept = Artifacts(task_id="T-003", acceptance_commands=accept,
        feedback=[], scope_report={"forbidden_writes": [], "off_scope_writes": []},
        rule_report=[{"slug": "done/tests-pass", "passed": False}],
        head_commit="c3d4e5f")

    for art in [clean, creep, no_accept]:
        report = verify(art)
        print(f"task {report.task_id}: passed={report.passed} findings={len(report.findings)}")
        for f in report.findings:
            print(f"  [{f.severity}] {f.code}: {f.detail}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 CI 中的验证门控

```yaml
# .github/workflows/verify.yml
jobs:
  verify_agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python3 tools/verify_agent.py
      - name: Check verdict
        run: |
          jq -e '.passed == true' outputs/verification/*.json || exit 1
```

合并保护依赖 `passed: true`。门控是 CI 中的标准步骤。

### 4.2 预交接钩子

智能体运行时在生成交接文档前调用门控。没有绿色裁决就没有交接。

### 4.3 分层防御

| 层 | 位置 | 可绕过？ | 检查内容 |
|---|-------|---------|---------|
| 预提交钩子 | 开发者本地 | 否（git hook） | 基本 lint |
| CI 状态检查 | PR 中 | 否（CI gate） | 验证门控 |
| 预工具授权 | 运行时 | 否（框架级） | 工具调用范围 |
| 预合并门控 | 合入前 | 否（branch protection） | 完整验证报告 |

---

## 5. 工程最佳实践

### 5.1 门控设计原则

| 原则 | 说明 |
|------|------|
| 确定性 | 同一工件集始终产生相同裁决。没有 LLM |
| 拒绝无例外 | Block 级发现需要签名覆盖，智能体不能自覆盖 |
| 一条路径 | 每次任务关闭一个 `verification_report.json` |
| 防御纵深 | 多层门控——一层的失败被下一层捕获 |

### 5.2 中文场景特别建议

- **覆盖日志中的 `user_id` 使用团队统一标识**——中文名拼音或英文名，保持一致性
- **AI 生成的覆盖理由也要签名**——即使是 AI 辅助生成的 `reason`，也必须通过签名门控
- **`--strict` 模式在中文团队中要明确文档**——什么分支、什么场景下开启，避免误用

### 5.3 踩坑经验

- **在门控中用 LLM 做裁决**——门控变成非确定性的，同一份差异不同时间得出不同结论。**修复：** 门控只能是确定性函数，LLM 评判在审查员侧
- **覆盖没有签名**——Slack 上说"这个覆盖可以"，但没有记录、没有签名、无法审计。**修复：** 每次覆盖写入 `overrides.jsonl`，HMAC 签名
- **只有一层门控**——CI 门控挂了，改动直接合入。**修复：** 预提交 → CI → 预合并 三层

---

## 6. 常见错误

### 错误 1：门控中使用 LLM 评判

**现象：** 门控每次运行结果不同。上周通过的验证本周失败，因为没有明确的原因。团队开始忽略门控结果。

**原因：** 门控中使用了 LLM 来"判断代码质量"。LLM 是非确定性的——同一个输入可能产生不同输出。

**修复：**
```python
# ❌ 门控中调用 LLM
def judge_acceptance(code):
    response = llm(f"这段代码完成了任务吗？{code}")
    return response.contains("是")  # 非确定性

# ✓ 门控只做确定性检查
def verify(art):
    for cmd in art.acceptance_commands:
        if cmd not in commands_run:
            return Finding("acceptance.missing", "block")  # 确定性
```

### 错误 2：覆盖没有签名

**现象：** 开发者在 Slack 上说"这个覆盖没问题"，门控被绕过了。合入后出问题，没有人能追溯是谁批准的覆盖。

**原因：** 覆盖没有被记录和签名。

**修复：** 每次覆盖写入 `overrides.jsonl`，包含时间戳、用户 ID、原因、HEAD 提交和 HMAC 签名。没有签名的覆盖运行时会拒绝。

### 错误 3：没有防御纵深

**现象：** CI 门控是唯一的检查点。CI 挂了（配置错误、服务不可用、脚本崩溃），所有改动全部跳过检查直接合入。

**原因：** 单一门控没有冗余。

**修复：** 分层防御：预提交钩子（本地）→ CI 状态检查（PR）→ 预合并门控（GitHub branch protection）。每层确定性的，一层的失败被下一层捕获。

---

## 7. 面试考点

### Q1：验证门控回答什么问题？为什么必须是确定性的？（难度：⭐）

**参考答案：**
验证门控回答"这个任务真的完成了吗？"——它读取范围契约、反馈日志、规则报告和差异，做出通过/不通过的裁决。

必须是确定性的因为：同一个工件集必须始终产生相同的裁决。如果门控是非确定性的（用了 LLM），上周通过的验证这周失败，团队会失去对门控的信任。LLM 评判器属于审查员一侧（定性评估），而不属于门控（状态判断）。

### Q2：Block 级发现如何处理？覆盖的签名机制是什么？（难度：⭐⭐）

**参考答案：**
Block 级发现不能被智能体覆盖。只能被人类覆盖，且必须记录 `override_reason`、`overridden_by` 用户 ID 和 `head_commit`。覆盖条目通过 HMAC-SHA256 签名。

覆盖日志写入 `outputs/verification/overrides.jsonl`。每次覆盖包含时间戳、原因、用户、提交 SHA 和签名。运行时拒绝任何缺少签名的覆盖。审计跟踪是 git 追踪的。

### Q3：什么是防御纵深？验证门控在分层防御中的位置？（难度：⭐⭐）

**参考答案：**
防御纵深是分层防御：预提交钩子（本地）→ CI 状态检查（PR）→ 预工具授权钩子（运行时）→ 预合并门控（分支保护）。

验证门控位于 CI / 预合并层。`microservices.io` 的操作手册明确指出：预提交钩子是不可绕过的，因为它不依赖智能体遵守指令。验证门控之前的每一层都在门控之前捕获问题，门控是最后一道防线。

### Q4：`--strict` 模式在什么场景下使用？（难度：⭐⭐⭐）

**参考答案：**
`--strict` 模式将所有 `warn` 级发现升级为 `block`。用于发布分支、阻塞性 PR、事故后排查。

不是默认开启的，因为"事事严格"会腐蚀日常流程。按分支选择加入：发布分支开 `--strict`，日常开发分支保持正常模式。这样可以保持日常开发效率，同时在关键时刻保持安全。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 验证门控 (Verification Gate) | "拦住问题的检查" | 工作台工件上的确定性函数，产生通过/不通过裁决 |
| Block 严重性 | "硬失败" | 阻止 `passed: true`，需要签名覆盖 |
| 覆盖日志 (Override Log) | "为什么放行" | 带签名和用户 ID 的条目，由审查审计 |
| 验收命令 | "证明" | Shell 命令——退出码 0 就是"完成"的定义 |
| 防御纵深 | "多层防护" | 预提交 → CI → 预工具 → 预合并——每层捕获上层的遗漏 |
| 混合规范 (Hybrid Norm) | "刚柔并济" | 确定性检查（门控）+ 定性评判（审查员）各司其职 |

---

## 📚 小结

智能体不能给自己的作业打分。验证门控是工作台流程中的决定性边缘——它读取范围契约、反馈日志、规则报告和差异，确定性地回答"这个任务真的完成了吗？" Block 级发现需要签名覆盖，没有例外。防御纵深（预提交 → CI → 预工具 → 预合并）确保单点故障不会让未经验证的改动上线。

下一课将是审查员智能体——门控检查"做对了吗"，审查员检查"做了对的事吗"。

---

## ✏️ 练习

1. **【实现】** 添加 `coverage_floor` 检查：测试命令必须产生至少 80% 的覆盖率报告。决定哪个工件携带覆盖率数据。

2. **【实现】** 支持 `--strict` 模式，将所有 `warn` 升级为 `block`。记录哪些场景下 strict 模式是正确的默认值。

3. **【实现】** 让门控额外生成 Markdown 摘要。论证哪些字段属于摘要。

4. **【实现】** 添加 `time_since_last_human_touch` 检查：人工编辑后 60 秒内修改的文件免于越界标记。

5. **【实验】** 在你的产品中对真实的智能体差异运行门控。多少发现是真实的，多少是噪音？门控需要在哪儿改进？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 验证门控 | `code/main.py` | 确定性验证、签名覆盖、覆盖率检查 |
| 技能提示词 | `outputs/skill-verification-gate.md` | 将门控接入项目：哪些验收命令、哪些规则是 block、覆盖审计日志 |

---

## 📖 参考资料

1. [官方文档] OpenAI Agents SDK Guardrails: https://platform.openai.com/docs/guides/agents-sdk/guardrails
2. [博客] Anthropic. "Harness Design for Long-Running Application Development". https://www.anthropic.com/engineering/harness-design-long-running-apps
3. [博客] microservices.io. "GenAI Dev Platform: Guardrails". https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html — 预提交到 CI 的防御纵深
4. [博客] Cloudflare. "Orchestrating AI Code Review at Scale". https://blog.cloudflare.com/ai-code-review/
5. [GitHub] logi-cmd/agent-guardrails. https://github.com/logi-cmd/agent-guardrails — 合并门控规范：范围 + 变异测试门控

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
