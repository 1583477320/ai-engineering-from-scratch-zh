# 动作预算、迭代上限和成本门控

> 一个中型电商智能体的月度 LLM 成本在其团队启用"订单跟踪"技能后从 $1,200 跳到 $4,800。这不是定价 bug。这是一个智能体找到了一个新循环并在其中持续花钱。Microsoft 的智能体治理工具包（2026 年 4 月 2 日）编纂了针对此类攻击的防御：每次请求的 `max_tokens`、每任务词元和美元预算、每天/每月上限、迭代上限、分层模型路由、提示缓存、上下文窗口化、在昂贵动作上的 HITL 检查点、预算违规的终止开关。Anthropic 的 Claude Code Agent SDK 在不同的名称下提供相同的原语。财务速度限制——例如 10 分钟内超过 $50 时切断访问——比月度上限更快地捕获循环。

**类型：** 实现课
**语言：** Python（标准库，分层成本门控模拟器）
**前置知识：** 阶段 15 · 10（权限模式）、阶段 15 · 12（持久执行）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 14（终止开关）— 成本违规触发终止开关

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 列出成本门控栈的十二个层次——从 per-request 到 monthly cap
- [ ] 解释为什么单一上限不够——不同的失败模式需要不同的时间尺度
- [ ] 实现速度限制（velocity limit）——10 分钟内超过 $X 时切断
- [ ] 理解为什么每添加一个新工具都需要新上限和新告警
- [ ] 为实际部署配置 max_turns、max_budget_usd 和每工具上限

---

## 1. 问题

自主智能体在每轮花费真实资金。聊天机器人的坏输出是坏回复；智能体的坏循环是账单。行业文档中这个失败模式的术语是"拒绝钱包"（Denial of Wallet）——智能体不断推理、不断调用工具、不断计费，没有东西阻止它，因为没有任何东西被设计来阻止它。

修复不是一个数字。它是在不同时间尺度和粒度上的限制栈：per-request、per-task、per-hour、per-day、per-month。设计良好的栈在几分钟内捕获失控循环，在几小时内捕获慢速泄漏，在一天内捕获错误发布。同一个栈在智能体长期和自主时始终保持在预算内。

---

## 2. 概念

### 2.1 成本门控栈的十二个层次

| 层 | 限制 | 捕获的失败 | 时间尺度 |
|------|------|-----------|---------|
| 1 | `max_tokens` per request | 单次完成的无限输出 | 秒 |
| 2 | Per-task token budget | 整次运行超出词元预算 | 分钟 |
| 3 | Per-task dollar budget (`max_budget_usd`) | 整次运行超出美元预算 | 分钟 |
| 4 | Per-tool call cap | 单一工具的滥用 | 分钟 |
| 5 | Iteration cap (`max_turns`) | 无限推理循环 | 分钟 |
| 6 | Per-minute / hour / day / month cap | 不同时间尺度的泄漏 | 可变 |
| 7 | 财务速度限制（$50 / 10 分钟） | 基于速率的燃烧 | 分钟 |
| 8 | 分层模型路由 | 默认用廉价模型 | 每次调用 |
| 9 | 提示缓存 | 系统提示不重复收费 | 每次调用 |
| 10 | 上下文窗口化 | 活跃上下文保持在预算内 | 每轮 |
| 11 | 昂贵动作的 HITL 检查点 | 人工确认高成本操作 | 按动作 |
| 12 | 预算违规的终止开关 | 任何上限被触发时中止会话 | 任意 |

### 2.2 为什么需要栈而不是单一上限

| 失败模式 | 描述 | 被哪层捕获 |
|---------|------|----------|
| 失控循环 | 智能体卡在 5 秒重试循环中 | 速度限制（层 7） |
| 慢速泄漏 | 智能体做约 2x 预期工作 | 每日上限（层 6） |
| 错误发布 | 新版本使用 5x 词元 | 每周/月度上限（层 6） |
| 合法激增 | 真实需求，非 bug | 每小时/每日上限，需清晰日志（层 6） |

### 2.3 观察到的 $1,200 → $4,800 案例

Microsoft 文档中的真实案例：一个电商智能体在添加新工具后每月成本增加了两倍。该工具允许智能体在每次会话中轮询订单状态。无循环检测。无每工具上限。无周环比增长率告警。

**修复是每工具上限 + 每日增长告警。** 这是一个模板：每个新工具表面都是一个新的潜在循环；每个新工具都需要自己的上限和自己的告警。

### 2.4 Claude Code 的预算表面

Claude Code Agent SDK 暴露：
- `max_turns` — 迭代上限
- `max_budget_usd` — 美元上限；违规时中止会话
- `allowed_tools` / `disallowed_tools` — 工具白名单和黑名单
- 工具使用前的钩子点——用于自定义成本核算

与权限模式阶梯（第 10 课）结合。没有 `max_budget_usd` 的 `autoMode` 会话是无法无天的自主性。

---

## 3. 从零实现

### 第 1 步：定义模拟配置和运行

```python
from dataclasses import dataclass, field

NORMAL_TURN_TOKENS = 2_500
LOOP_TURN_TOKENS = 8_000
LOOP_STARTS_AT = 30
DOLLARS_PER_KTOK = 0.003  # Sonnet 类模型，2026 年中价格

@dataclass
class Governor:
    max_tokens_per_request: int = 10_000
    max_turns: int = 200
    max_budget_usd: float = 50.0
    velocity_usd_per_min: float = 5.0
    velocity_window_min: float = 10.0
    monthly_cap_usd: float = 500.0
    enable_request_cap: bool = True
    enable_iter_cap: bool = True
    enable_velocity: bool = True
    enable_session_cap: bool = True
    enable_monthly_cap: bool = True
    seconds_per_turn: float = 30.0

@dataclass
class Run:
    turns: int = 0
    tokens: int = 0
    dollars: float = 0.0
    history: list = field(default_factory=list)
    stopped_by: str = ""
```

### 第 2 步：实现速度限制检测

```python
def velocity_exceeded(run: Run, gov: Governor, now_min: float) -> bool:
    """检查过去 velocity_window_min 分钟内的平均消耗率。"""
    if not run.history:
        return False
    cutoff = now_min - gov.velocity_window_min
    window = [(t, d) for (t, d) in run.history if t >= cutoff]
    if not window:
        return False
    start_min, start_dollars = window[0]
    window_dollars = run.dollars - start_dollars
    elapsed = max(now_min - start_min, 1e-9)
    rate = window_dollars / elapsed
    return rate > gov.velocity_usd_per_min
```

### 第 3 步：实现成本门控模拟器

```python
def simulate(gov: Governor, label: str) -> Run:
    run = Run()
    now_min = 0.0

    for turn in range(1, 10_001):
        tok = LOOP_TURN_TOKENS if turn >= LOOP_STARTS_AT else NORMAL_TURN_TOKENS
        if gov.enable_request_cap and tok > gov.max_tokens_per_request:
            tok = gov.max_tokens_per_request
        run.turns = turn
        run.tokens += tok
        run.dollars += (tok / 1000.0) * DOLLARS_PER_KTOK
        now_min += gov.seconds_per_turn / 60.0
        run.history.append((now_min, run.dollars))

        if gov.enable_iter_cap and turn >= gov.max_turns:
            run.stopped_by = "max_turns"; break
        if gov.enable_session_cap and run.dollars >= gov.max_budget_usd:
            run.stopped_by = "max_budget_usd"; break
        if gov.enable_velocity and velocity_exceeded(run, gov, now_min):
            run.stopped_by = "velocity_limit"; break
        if gov.enable_monthly_cap and run.dollars >= gov.monthly_cap_usd:
            run.stopped_by = "monthly_cap"; break

    return run
```

### 第 4 步：运行三种场景对比

```python
def main():
    # 场景 1：无上限
    g = Governor(enable_request_cap=False, enable_iter_cap=False,
                 enable_velocity=False, enable_session_cap=False,
                 enable_monthly_cap=False)
    g.max_turns = 10_000
    simulate(g, "no caps (iter 10k sim)")

    # 场景 2：仅月度上限
    g = Governor(enable_request_cap=False, enable_iter_cap=False,
                 enable_velocity=False, enable_session_cap=False,
                 enable_monthly_cap=True)
    simulate(g, "monthly cap only")

    # 场景 3：分层栈
    g = Governor()
    simulate(g, "layered stack")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 成本门控对照

| 工具 | 支持的上限 | 与 LLM 的集成 |
|------|----------|------------|
| Claude Code Agent SDK | max_turns、max_budget_usd、工具白名单 | 原生 |
| Microsoft Agent Governance Toolkit | 十二层栈 + 财务速度限制 | OWASP Agentic Top 10 兼容 |
| Temporal + OpenAI SDK | 活动级超时和重试预算 | 通过活动装饰器 |

### 4.2 每工具上限设计

| 工具 | 建议上限 | 理由 |
|------|---------|------|
| WebFetch | 10/会话 | 防止爬虫循环 |
| shell_exec | 20/会话 | 防止无限命令执行 |
| file_write | 50/会话 | 防止文件系统垃圾 |
| llm_call | 100/会话 | 防止推理循环 |

---

## 5. 工程最佳实践

### 5.1 成本门控设计原则

| 原则 | 说明 |
|------|------|
| 栈，不是单数字 | 不同失败模式需要不同时间尺度 |
| 每工具上限 | 每个新工具都是新潜在循环——需要自己的上限 |
| 速度限制比月度上限快 | 10 分钟内 $50 比月度 $500 更快捕获循环 |
| 分层模型路由节省成本 | 默认用廉价模型，仅必要时升级 |

### 5.2 中文场景特别建议

- **美元上限在中文 API 中需要调整**——中文 API 定价不同于 OpenAI，max_budget_usd 可能需要缩小
- **中文云环境的标价单位不同**——某些中国云提供商按"调用次数"而非词元计费，需要不同的预算跟踪
- **财务速度限制（velocity limit）需要考虑汇率**——如果使用美元但 API 以人民币计费，需要实时转换

### 5.3 踩坑经验

- **仅月度上限**——月度 $500 上限捕获循环时钱包已经半空。**修复：** 分层栈——速度限制在分钟内捕获循环
- **无每工具上限**——新工具无限使用，成本从 $1,200 跳到 $4,800。**修复：** 每添加一个工具就设上限
- **无告警**——成本缓慢增长没有被注意到。**修复：** 每日增长告警

---

## 6. 常见错误

### 错误 1：仅设单一上限

**现象：** 只设了月度 $500 上限。第 5 天智能体进入循环，第 28 天触发了上限。团队收到 $480 的账单。

**原因：** 单一上限在不同时间尺度上不匹配——月度上限捕获最终问题，但行动太晚。

**修复：** 分层栈——速度限制（$50/10 分钟）+ 会话预算（$50）+ 月度上限（$500）。

### 错误 2：添加新工具时不设上限

**现象：** 添加了订单状态轮询工具。没有设置每工具上限。智能体开始在每个会话中轮询 100 次。成本从 $1,200 跳到 $4,800。

**原因：** 每个新工具都是新潜在循环。工具可能比预期的更有用——或者被误用。

**修复：** 每次添加工具时设置 per-tool 上限 + 每日增长告警。

### 错误 3：认为预算上限足够，不设终止开关

**现象：** max_budget_usd 设置为 $50。智能体达到 $50 后继续运行——因为上限是"软"的，不会中止会话。

**原因：** max_budget_usd 需要配合终止开关——仅记录警告而不停止会话。

**修复：** 在预算违规时设置硬停止 + 独立的重启路径。

---

## 7. 面试考点

### Q1：成本门控栈的十二个层次是什么？哪些最重要？（难度：⭐）

**参考答案：**
Per-request → Per-task → Per-tool → Iteration → Per-hour/day/month → Velocity → 分层模型路由 → 提示缓存 → 上下文窗口化 → HITL 检查点 → 终止开关。

**最重要的三层：** 速度限制（分钟内捕获循环）、迭代上限（防止无限循环）、会话预算（限额止损）。这三层覆盖了最频繁的失败模式。

### Q2：为什么单一上限不够？不同时间尺度对应什么失败模式？（难度：⭐⭐）

**参考答案：**
不同失败模式需要不同的时间尺度：

| 失败模式 | 捕获层 | 响应时间 |
|---------|-------|---------|
| 失控循环 | 速度限制（$50 / 10 分钟） | 分钟 |
| 慢速泄漏 | 每日上限 | 小时 |
| 错误发布 | 每周/月度上限 | 天 |
| 合法激增 | 每小时/每日上限 + 日志 | 实时 |

### Q3：每个新工具为什么需要自己的上限和告警？（难度：⭐⭐）

**参考答案：**
电商案例：添加"订单跟踪"工具后，每月成本从 $1,200 跳到 $4,800。工具允许在每个会话中轮询订单状态。无循环检测、无 per-tool 上限、无增长率告警。

每个新工具都是新潜在循环。工具可能比预期更有用，也可能被误用。正确的做法：每次添加工具时设置 per-tool 上限 + 每日增长告警。这也是 OWASP Agentic Top 10 和 Microsoft Agent Governance Toolkit 的要求。

### Q4：财务速度限制（velocity limit）如何工作？为什么比月度上限快？（难度：⭐⭐⭐）

**参考答案：**
速度限制检查过去 N 分钟内的平均消耗率。如果 10 分钟内超过 $5/分钟（即 $50），会话被中止。

月度上限：智能体可以在第 1 天烧掉 $450，然后在剩余 27 天保持在 $50 以下——仍然在 $500 上限内。速度限制在失控循环的几分钟内就会触发。

在 Claude Code 中这由 max_budget_usd 补充。但 max_budget_usd 是会话级——速度限制需要外部监控。Microsoft Agent Governance Toolkit 将其作为一等原语。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 拒绝钱包 (Denial of Wallet) | "失控账单" | 智能体循环产生支出，无上限阻止 |
| max_tokens | "每次请求上限" | 单次完成的大小的上限 |
| max_turns | "迭代上限" | 会话中智能体循环迭代数上限 |
| max_budget_usd | "美元终止开关" | 会话成本上限；违规时中止 |
| 速度限制 (Velocity Limit) | "速率上限" | 短窗口内的支出限制（如 $50/10 分钟）|
| 分层模型路由 | "小模型优先" | 默认用廉价模型；仅必要时升级 |
| 提示缓存 | "缓存的系统提示" | 供应商侧缓存减少重新发送的消耗 |

---

## 📚 小结

自主智能体花费真实资金——坏循环是账单。成本门控栈的十二个层次在不同时间尺度上捕获失败：速度限制在分钟内捕获循环、每日上限在小时内捕获泄漏、月度上限在天内捕获错误发布。每添加一个新工具都需要新上限和新告警——电商案例中订单跟踪工具使月度成本从 $1,200 跳到 $4,800。分层栈共同确保智能体只在预算内运行。

下一课：终止开关、断路器、金丝雀标记——成本门控之外的下一个安全层。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认速度限制在迭代上限之前触发。现在禁用它看看智能体在迭代上限捕获前花了多少。

2. **【实现】** 为浏览器智能体设计每工具上限集。哪个工具需要最严的上限？哪个可以无限运行？

3. **【阅读】** 阅读 Microsoft Agent Governance Toolkit 文档。列出每个上限类型并映射到失败模式。

4. **【设计】** 为真实任务（如"在仓库中分类 50 个 issue"）定价隔夜无人值守运行。设置 2x 点估计的 max_budget_usd。论证 2x。

5. **【实现】** 为 Claude Code 的 max_budget_usd 设计一个补充性外部速度限制。什么触发切断，重新启用的流程是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 成本门控模拟器 | `code/main.py` | 无上限 vs 月度上限 vs 分层栈，三种场景对比 |
| 技能提示词 | `outputs/skill-agent-budget-audit.md` | 审计智能体部署的成本门控栈 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Claude Code Agent SDK: Agent Loop and Budgets". https://code.claude.com/docs/en/agent-sdk/agent-loop
2. [官方文档] Microsoft. "Agent Framework: Human-in-the-Loop and Governance". https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop
3. [官方文档] Anthropic. "Claude Managed Agents Overview". https://platform.claude.com/docs/en/managed-agents/overview
4. [官方文档] Anthropic. "Prompt Caching (Claude API Docs)". https://platform.claude.com/docs/en/prompt-caching
5. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
