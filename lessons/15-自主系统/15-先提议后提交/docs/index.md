# 人在环中：先提议后提交

> 2026 年 HITL 的共识是具体的。它不是"智能体问，用户点击批准。"它是先提议后提交：提议的动作被持久化到一个带幂等键的持久存储；呈现给审查者意图、数据血缘、涉及权限、爆炸半径、回滚计划；仅在正面确认后提交；执行后验证确认副作用确实发生。LangGraph 的 `interrupt()` + PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现相同的形状。规范的失败模式是橡皮戳批准："批准？"在没有审查的情况下被点击。文档化的缓解是带有明确清单的挑战-回应模式。

**类型：** 实现课
**语言：** Python（标准库，带幂等性的先提议后提交状态机）
**前置知识：** 阶段 15 · 12（持久执行）、阶段 14 · 39（审查员智能体）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 14（终止开关）— 金丝雀标记检测持久化攻击；阶段 14 · 40（多会话交接）— 交接包使用相同的结构化元数据

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述先提议后提交的四阶段状态机：提议→呈现→提交→验证
- [ ] 解释幂等键为什么防止重试时的双重执行
- [ ] 实现一个带幂等键的先提议后提交系统——提议在持久存储中保存，重试时自动去重
- [ ] 对比橡皮戳批准与挑战-回应清单——后者是 EU AI Act Article 14 合规的形状
- [ ] 区分后果动作（始终 HITL）、可逆动作（有时 HITL）和读取/检查（从不 HITL）

---

## 1. 问题

智能体采取动作。用户决定：批准还是不批准。如果决定是瞬间的，那可能不是审查。如果决定是有结构的，它是慢的但可信的。

2023 年的 HITL 模式是同步提示："智能体想发送邮件到 X，内容为 Y——批准？"用户点击批准。每个人都觉得系统是安全的。实际上这个表面被大量橡皮戳——用户快速批准，批准预测很少，当智能体出错时审计跟踪显示用户无法回忆的批准历史。

2026 年的模式——先提议后提交——将 HITL 移到持久基板上，附加结构化元数据，并要求正面确认。

---

## 2. 概念

### 2.1 四阶段状态机

```
提议 → 持久化（含意图、血缘、权限、爆炸半径、回滚计划、幂等键）
  ↓
呈现 → 审查者看到所有元数据（人类，不是智能体自己审查自己）
  ↓
提交 → 正面确认 → 动作执行
  ↓
验证 → 重新读取目标资源确认副作用确实发生
```

### 2.2 幂等键

没有幂等键时，瞬态故障后的重试可以双重执行已批准的动作。

```
用户批准：从 A 转 $100 到 B
网络闪断
工作流重试
用户批准了一次，但转账执行了两次
```

幂等键将批准绑定到单一的、唯一的副作用；第二次执行是空操作。这是 Stripe 和 AWS API 使用的相同模式。

### 2.3 橡皮戳 vs 挑战-回应

| 模式 | 特征 | 问题 |
|------|------|------|
| 橡皮戳 | "批准？" → 快速点击 | 无真正审查；审计显示无法回忆的批准历史 |
| 挑战-回应 | 清单强制回答三个问题 | 不能回答 = 拒绝或升级 |

挑战-回应清单：
- "你理解这个动作触及了什么资源吗？"
- "你验证了爆炸半径是可接受的吗？"
- "如果失败你有回滚计划吗？"

不是为了官僚主义——是强制函数。无法勾选框的审查者要么要求澄清（升级），要么拒绝（安全默认）。

### 2.4 什么算后果动作

| 类别 | 动作 | 是否 HITL |
|------|------|----------|
| 后果动作 | 不可逆写入、金融交易、出站通信、生产数据库变更、破坏性文件系统操作 | 始终 |
| 可逆动作 | 本地文件编辑、预发环境变更、有清晰回滚的可逆写入 | 有时 |
| 读取/检查 | 读文件、列出资源、调用只读 API | 从不 |

### 2.5 提交后验证

"提交运行了"不等于"副作用发生了"。网络分区和竞态条件可以产生一个认为自己成功而后端没有持久化的流程。验证步骤在提交后重新读取目标资源确认。

---

## 3. 从零实现

### 第 1 步：定义提议和持久存储

```python
import hashlib, json, os
from dataclasses import dataclass, field

@dataclass
class Proposal:
    thread_id: str
    action: str
    payload: dict
    intent: str
    lineage: str
    blast_radius: str
    rollback: str

    def key(self) -> str:
        sig = json.dumps({"t": self.thread_id, "a": self.action,
                          "p": self.payload}, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

class Store:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({}, f)

    def save(self, key, record):
        data = self.all()
        data[key] = record
        with open(self.path, "w") as f:
            json.dump(data, f)

    def all(self):
        with open(self.path) as f:
            return json.load(f)
```

### 第 2 步：实现提议→提交→验证流程

```python
def propose(store, proposal):
    k = proposal.key()
    existing = store.all().get(k)
    if existing:
        print(f"  [propose] 幂等：记录 {k} 已存在 (status={existing['status']})")
        return k
    record = {"status": "waiting", **vars(proposal)}
    store.save(k, record)
    print(f"  [propose] 记录 {k} 已存储，等待审查")
    return k

def commit(store, k):
    data = store.all()
    rec = data[k]
    if rec["status"] == "committed":
        print(f"  [commit] 幂等：{k} 已提交，不重新执行")
        return True
    if rec["status"] != "approved":
        print(f"  [commit] 拒绝：{k} status={rec['status']}")
        return False
    execute(proposal)  # 执行副作用
    rec["status"] = "committed"
    store.save(k, rec)
    return True
```

### 第 3 步：实现挑战-回应清单

```python
def checklist_approve(store, k, understood, verified, rollback_ready):
    if not (understood and verified and rollback_ready):
        print("  [approve:checklist] 拒绝（清单不完整）")
        return False
    rec = store.all()[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "challenge_response"
    store.save(k, rec)
    print("  [approve:checklist] 批准（三个检查全部通过）")
    return True
```

### 第 4 步：运行三种场景

```python
def main():
    # 场景 1：干净的挑战-回应批准流程
    p1 = Proposal(thread_id="t-001", action="email.send", ...)
    k = propose(store, p1)
    checklist_approve(store, k, understood=True, verified=True, rollback_ready=True)
    commit(store, k)

    # 场景 2：重试幂等——已提交的动作不重新执行
    commit(store, k)  # 重试 → 幂等
    print(f"  副作用数：{len(SIDE_EFFECTS)} (不变)")

    # 场景 3：橡皮戳 vs 挑战-回应
    p3 = Proposal(thread_id="t-003", action="db.drop_table", ...)
    ok = checklist_approve(store, p3, understood=True, verified=True, rollback_ready=False)
    if not ok:
        commit(store, p3)  # 拒绝 → commit 拒绝
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 各框架的 HITL 实现

| 框架 | API | 持久化 |
|------|-----|--------|
| LangGraph | `interrupt()` | PostgreSQL 检查点 |
| Microsoft Agent Framework | `RequestInfoEvent` | 持久化 HITL 请求 |
| Cloudflare | `waitForApproval()` | Durable Objects |

API 名称不同；形状相同。

### 4.2 EU AI Act Article 14 合规

Article 14 要求高风险 AI 系统有有效的人类监督。"有效"不是装饰性的。监管语言明确排除了橡皮戳模式。带有挑战-回应的先提议后提交是通过 Article 14 审查的形状。

---

## 5. 工程最佳实践

### 5.1 先提议后提交设计原则

| 原则 | 说明 |
|------|------|
| 提议必须持久化 | 确保重试时幂等 |
| 元数据必须完整 | 意图、血缘、权限、爆炸半径、回滚计划 |
| 幂等键必须唯一 | 基于 (thread_id, action, payload) 的哈希 |
| 挑战-回应防橡皮戳 | 三个问题必须正面回答 |
| 提交后验证 | 确认副作用确实发生 |

### 5.2 中文场景特别建议

- **HITL 清单的验证项用中文写**——方便非英文母语的审查员
- **邮件/通知动作要考虑中国互联网法规**——例如短信发送需要合规审批
- **挑战-回应清单在中文下更有效**——中文用户更容易被"是否理解"这样的检查点阻止

---

## 6. 常见错误

### 错误 1：使用橡皮戳批准

**现象：** "批准？"被点击而不审查。所有审计跟踪显示批准，但没有任何真正的审查。

**原因：** 默认 UI 产生快速批准而不进行真正审查。

**修复：** 挑战-回应清单：三个问题必须正面回答才能启用批准按钮。

### 错误 2：重试时双重执行

**现象：** 智能体批准后重试，"发送邮件"动作执行了两次。用户收到了两封相同的邮件。

**原因：** 没有幂等键。重试时动作被重新执行。

**修复：** 基于 (thread_id, action, payload) 的幂等键。重试时检查已批准记录并跳过执行。

### 错误 3：提交后不验证

**现象：** "提交运行了"——但后端实际上没有持久化。竞态条件或网络分区导致"成功"但实际未完成。

**原因：** 没有验证步骤。

**修复：** 提交后重新读取目标资源确认副作用确实发生。与数据库 `RETURNING` 子句或 AWS `GetObject` 相同的模式。

---

## 7. 面试考点

### Q1：先提议后提交的四阶段是什么？与传统 HITL 的区别是什么？（难度：⭐）

**参考答案：**
四阶段：提议→呈现→提交→验证。

传统 HITL 是同步提示："智能体想 X——批准？"用户快速点击。没有结构化元数据，没有幂等键，没有提交后验证。先提议后提交将 HITL 移到持久基板上，附加意图/血缘/权限/爆炸半径/回滚计划，并要求正面确认。

### Q2：幂等键为什么防止双重执行？（难度：⭐⭐）

**参考答案：**
基于 (thread_id, action, payload) 的哈希生成唯一幂等键。第一次批准后状态变为 "committed"。重试时提交函数检查：如果键已存在且状态为 "committed"，不重新执行。

这是 Stripe 和 AWS API 使用的相同模式。将批准绑定到单一的副作用；第二次执行是空操作。

### Q3：挑战-回应清单如何防止橡皮戳？（难度：⭐⭐）

**参考答案：**
默认 UI 的 "批准/拒绝" 按钮产生快速批准而不进行真正审查。

挑战-回应清单要求正面回答三个问题：(1) 你理解触及了什么资源吗？(2) 你验证了爆炸半径吗？(3) 你有回滚计划吗？不能回答 = 拒绝或升级。不是官僚主义——是强制函数。EU AI Act Article 14 明确要求这种形状。

### Q4：后果动作、可逆动作、读取/检查的 HITL 策略是什么？（难度：⭐⭐⭐）

**参考答案：**
- **后果动作**（始终 HITL）：不可逆写入、金融交易、出站通信、生产数据库变更、破坏性文件系统操作
- **可逆动作**（有时 HITL）：本地文件编辑、预发环境变更、有清晰回滚的可逆写入
- **读取/检查**（从不 HITL）：读文件、列出资源、调用只读 API

不是每个动作都需要先提议后提交。分类基于副作用的不可逆性和范围。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 先提议后提交 | "两阶段批准" | 持久化提议 + 正面确认 + 验证 |
| 幂等键 | "重试安全令牌" | 每提议唯一；第二次执行为空操作 |
| 数据血缘 | "来源" | 导致提议的特定源内容 |
| 爆炸半径 | "最坏情况" | 如果动作出错的影响范围 |
| 橡皮戳 | "快速批准" | 未真正审查就点击"批准" |
| 挑战-回应 | "强制清单" | 审查者必须正面确认特定问题 |

---

## 📚 小结

先提议后提交是 2026 年 HITL 的规范形状：提议持久化、呈现元数据、正面确认、提交后验证。幂等键防止重试时的双重执行。挑战-回应清单防止橡皮戳——EU AI Act Article 14 要求这种形状。后果动作始终 HITL，可逆动作有时 HITL，读取/检查从不 HITL。与 LangGraph `interrupt()`、Microsoft `RequestInfoEvent`、Cloudflare `waitForApproval()` 集成。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认已批准提议的重试使用持久记录，不重新执行。现在修改幂等键以包含时间戳，显示重试双重执行。

2. **【实现】** 扩展提议记录以包含 `rollback` 字段。模拟验证步骤失败的执行。显示回滚自动触发。

3. **【设计】** 为特定动作设计挑战-回应清单（如"发布到公共 Twitter 账户"）。审查者必须回答哪三个问题？为什么这三个？

4. **【思考】** 选择一个同步"批准？"提示足够的场景（不需要持久存储）。解释为什么，以及你接受的风险类别。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 先提议后提交状态机 | `code/main.py` | 幂等键 + 挑战-回应 + 重试防双重执行 |
| 技能提示词 | `outputs/skill-hitl-design.md` | 审议提议的 HITL 工作流是否具备正确形状 |

---

## 📖 参考资料

1. [官方文档] Microsoft. "Agent Framework: Human in the Loop". https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop — `RequestInfoEvent`、持久化批准
2. [官方文档] Cloudflare. "Agents: Human in the Loop". https://developers.cloudflare.com/agents/concepts/human-in-the-loop/ — `waitForApproval()` 和 Durable Objects
3. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy — HITL 作为长期风险的缓解
4. [官方文档] EU AI Act — Article 14: Human Oversight. https://artificialintelligenceact.eu/article/14/ — 高风险系统的监管基线
5. [官方文档] Anthropic. "Claude's Constitution (January 2026)". https://www.anthropic.com/news/claudes-constitution — 监督周围的宪法框架

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
