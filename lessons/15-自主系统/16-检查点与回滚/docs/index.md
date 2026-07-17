# 检查点与回滚

> 每个图状态转换都持久化。当工作者崩溃时，其租约过期，另一个工作者在最新检查点处接手。Cloudflare Durable Objects 跨小时或周持有状态。先提议后提交（第 15 课）为每个动作定义了回滚计划。提交后验证关闭循环。EU AI Act Article 14 使有效的人类监督成为高风险系统的强制要求——实践中这意味着检查点必须可查询，回滚必须经过演练，审计跟踪必须在部署后存活。尖锐的失败模式：没有幂等键和前置条件检查，瞬态故障后的重试可能双重执行已批准的动作。提交后验证正是捕获它的东西。

**类型：** 实现课
**语言：** Python（标准库，检查点和回滚状态机）
**前置知识：** 阶段 15 · 12（持久执行）、阶段 15 · 15（先提议后提交）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 40（多会话交接）— 交接包使用检查点恢复状态；阶段 14 · 42（工作台毕业设计）— 检查点是工作台包的核心原语

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解检查点、租约、前置条件、提交后验证和回滚如何组成完整的动作安全链
- [ ] 实现一个带幂等键和前置条件的检查点系统——同一动作重试不双重执行，状态漂移时自动终止
- [ ] 区分带内回滚（直接反转）和补偿事务（SAGA 模式）
- [ ] 解释"先标记为已完成再执行"vs"先执行再标记"的持久性权衡
- [ ] 识别 Article 14 对检查点和回滚的合规要求

---

## 1. 问题

持久执行（第 12 课）使崩溃的智能体可恢复。先提议后提交（第 15 课）使已批准的动作可审计。本课将它们连接：当已批准的动作部分执行、崩溃并恢复时会发生什么？回滚何时运行，针对什么状态？

在所有实际系统中，真正有效组合是：**幂等键 + 前置条件检查 + 提交后验证 + 验证失败时回滚**。

---

## 2. 概念

### 2.1 每个转换都持久化

图状态转换是工作流从一个命名状态移动到另一个的每一步。朴素实现只在特定提交点持久化；生产实现持久化每个转换。成本（几次额外写入）相对于可靠性收益（重放落在任何地方、租约恢复精确）很小。

### 2.2 租约恢复

当工作者崩溃时，工作流没有丢失；租约（工作者正在执行此运行的短期声明）过期。另一个工作者拾起最新检查点并恢复。租约机制使生产系统能够承受滚动部署而不丢失进行中的工作。

### 2.3 幂等键 + 前置条件

仅幂等键不够。考虑：工作流被批准"在余额 > $1000 时从 A 转 $100 到 B"。工作流被提交，中途崩溃，恢复。如果只检查幂等键并恢复执行，转账运行一次（正确）。但如果崩溃和恢复之间，A 的余额通过另一个工作流降到 $500。幂等检查仍然通过；前置条件不通过。没有前置条件检查，我们发送了透支。

每个后果动作需要两者：
- **幂等键**：防止双重执行
- **前置条件检查**：确认状态仍然与批准时一致

### 2.4 提交后验证

"工具返回 200"不是验证。真正的验证重新读取目标状态并确认副作用确实发生。

| 模式 | 验证方法 |
|------|---------|
| 数据库更新 | `UPDATE ... RETURNING *` 然后断言行匹配预期状态 |
| 邮件发送 | 提交后检查已发送文件夹 |
| 文件写入 | 读回文件并哈希 |
| API 调用 | 后续 `GET` 目标资源 |

### 2.5 回滚计划

| 类型 | 方法 | 示例 |
|------|------|------|
| 带内回滚 | 直接反转副作用 | `DELETE` after `INSERT`；发送更正邮件 |
| 补偿事务 | 新动作中和原始动作 | SAGA 模式 |
| 带外回滚 | 告警人类，暂停工作流 | 留下坏状态供调查 |

没有回滚的操作（"我们无法撤销这个"）必须在提议中标记，并在提交时需要更强的 HITL。

### 2.6 最尖锐的失败模式：双重执行

1. 动作被批准，幂等键 k
2. 提交开始，执行，返回 200
3. 工作流在持久化"已提交"状态之前崩溃
4. 工作流恢复；看到"已批准但未提交"；重新执行
5. 副作用触发两次

缓解：执行前持久化"进行中"意图，带幂等键执行，仅在提交后验证成功后标记"已提交"。

---

## 3. 从零实现

### 第 1 步：定义检查点存储

```python
import hashlib, json, os
from dataclasses import dataclass

@dataclass
class Checkpoint:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, k: str, v: dict) -> None:
        data = self.load()
        data[k] = v
        # 原子性写入：写临时文件 → fsync → 重命名
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
```

### 第 2 步：实现带前置条件的转账工作流

```python
DB = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}

def run_transfer(cp, txid, from_acct, to_acct, amount, min_balance,
                 inject_crash_after_execute=False, inject_verify_fail=False):
    k = hashlib.sha256(txid.encode()).hexdigest()[:12]
    record = cp.load().get(k, {"status": "new"})

    # 幂等性：任何终态都短路
    terminal = {"committed": "idempotent-skip", "verified": "ok",
                "rolled-back": "verify-fail-rolled-back", "aborted-precondition": "aborted"}
    if record["status"] in terminal:
        return terminal[record["status"]]

    # 前置条件检查
    if DB[f"balance_{from_acct}"] - amount < min_balance:
        cp.save(k, {"status": "aborted-precondition"})
        return "aborted-precondition"

    # 记录意图（崩溃时保留）
    prior_last = DB["last_transfer_id"]
    cp.save(k, {"status": "committed", "txid": txid,
                "prior_last_transfer_id": prior_last})

    # 执行
    persist_transfer(txid, from_acct, to_acct, amount)
    if inject_crash_after_execute:
        raise RuntimeError("simulated crash after execute")

    # 提交后验证
    if inject_verify_fail or DB["last_transfer_id"] != txid:
        rollback_transfer(txid, from_acct, to_acct, amount, prior_last)
        cp.save(k, {"status": "rolled-back"})
        return "verify-fail-rolled-back"

    cp.save(k, {"status": "verified"})
    return "ok"
```

### 第 3 步：运行四种场景

```python
def main():
    # 场景 1：干净运行
    cp = Checkpoint(...)
    out = run_transfer(cp, "tx-001", "A", "B", 100, min_balance=200)
    print(f"  result={out}")  # ok

    # 场景 2：提交中途崩溃，重试 → 幂等捕获
    cp = Checkpoint(...)
    try:
        run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200,
                     inject_crash_after_execute=True)
    except RuntimeError:
        pass
    out = run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200)
    print(f"  retry result={out}")  # idempotent-skip

    # 场景 3：前置条件失败（余额低于最低值）
    out = run_transfer(cp, "tx-003", "A", "B", 10_000, min_balance=200)
    print(f"  result={out}")  # aborted-precondition

    # 场景 4：验证失败 → 回滚
    out = run_transfer(cp, "tx-004", "A", "B", 100, min_balance=200,
                       inject_verify_fail=True)
    print(f"  result={out}")  # verify-fail-rolled-back
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 各框架的检查点机制

| 框架 | 检查点后端 | 租约恢复 | 提交后验证 |
|------|----------|---------|----------|
| LangGraph | PostgreSQL | 工作者租约过期后另一个接手 | 手动 |
| Cloudflare Durable Objects | 跨小时/周状态 | 对象生命周期管理 | 内置 |
| Microsoft Agent Framework | 工作流 API | 重放 + 幂等性 | 手动 |

### 4.2 Article 14 合规要求

| 要求 | 实现 |
|------|------|
| 检查点可查询 | PostgreSQL / 查询 API |
| 回滚经过演练 | 至少端到端运行一次 |
| 审计跟踪在部署后存活 | 检查点后端非易失 |
| 失败验证被告警 | 不静默记录 |

---

## 5. 工程最佳实践

### 5.1 检查点和回滚设计原则

| 原则 | 说明 |
|------|------|
| 每个转换都持久化 | 不仅提交点——重放可以落在任何地方 |
| 幂等键 + 前置条件 | 两者都必需——幂等防双重执行，前置条件防状态漂移 |
| 提交后验证 | "返回 200"≠"副作用发生"——重新读取目标资源确认 |
| 回滚计划必须经过演练 | 至少端到端运行一次 |

### 5.2 中文场景特别建议

- **数据库事务在中文云数据库中的行为一致**——事务隔离级别在 MySQL 和 PostgreSQL 中行为相同
- **回滚计划中涉及的第三方服务需要考虑中国互联网法规**——例如邮件回滚可能需要遵守中国反垃圾邮件规定
- **检查点后端在中文云环境中同样可用**——阿里云 RDS PostgreSQL、腾讯云 TDSQL 都支持

---

## 6. 常见错误

### 错误 1：仅用幂等键不检查前置条件

**现象：** 工作流批准在余额 > $1000 时转账 $100。崩溃后恢复，幂等检查通过（没执行过）。但恢复期间余额降到了 $500。转账执行——发送了透支。

**原因：** 幂等键防止双重执行，但不检查状态是否仍然与批准时一致。

**修复：** 每个后果动作需要幂等键 + 前置条件检查。

### 错误 2：先标记"已完成"再执行

**现象：** 工作流在执行副作用之前将状态标记为"已完成"。执行失败。下次运行看到"已完成"，跳过——实际从未执行。

**原因：** "先标记完成再执行"模式：在执行前记录意图。但如果崩溃发生在保存意图之后、执行之前，重试看到"已完成"并跳过。

**修复：** 执行前记录"进行中"意图，执行后验证通过才标记"已完成"。验证失败 → 回滚。

### 错误 3：提交后不验证

**现象：** 工具返回 200，工作流认为成功。实际上后端因为竞态条件没有持久化。工作流认为完成但副作用从未发生。

**原因：** "返回 200"≠"副作用发生"。

**修复：** 提交后重新读取目标资源。数据库用 `RETURNING *`，邮件检查已发送文件夹，API 用后续 `GET`。

---

## 7. 面试考点

### Q1：检查点、租约、前置条件、验证、回滚如何组成完整的动作安全链？（难度：⭐⭐）

**参考答案：**
- **检查点**：每次状态转换持久化，崩溃后可以从任何地方恢复
- **租约**：工作者的短期声明，崩溃时过期，其他工作者在最新检查点接手
- **前置条件**：确认状态仍然与批准时一致（如余额 > $1000）
- **提交后验证**：重新读取目标资源确认副作用确实发生
- **回滚**：验证失败时反转副作用（带内回滚、补偿事务或带外告警）

每个捕获不同的失败类别。没有一个单独足够。

### Q2：为什么幂等键不够，还需要前置条件？（难度：⭐⭐）

**参考答案：**
幂等键防止双重执行。但如果崩溃和恢复之间状态漂移了（另一个工作流改变了余额），幂等检查仍然通过——因为动作从未执行过。但前置条件检查会失败——因为状态不再与批准时一致。

没有前置条件，你发送了透支。幂等键 + 前置条件 = 完整的检查。

### Q3：Article 14 对检查点和回滚的合规要求是什么？（难度：⭐⭐）

**参考答案：**
Article 14 要求"有效的人类监督"。实践中实现为：
- 检查点必须可查询——审计员可以检查任何状态转换
- 回滚必须经过演练——至少端到端运行一次
- 审计跟踪必须在部署后存活——检查点后端非易失
- 失败验证被告警——不静默记录

没有验证 + 回滚路径的流程无法通过 Article 14 测试。

### Q4："先标记完成再执行"有什么风险？如何缓解？（难度：⭐⭐⭐）

**参考答案：**
风险：工作流在执行副作用之前将状态标记为"已完成"。执行失败。下次运行看到"已完成"，跳过——实际从未执行。

缓解：执行前记录"进行中"意图，执行后验证通过才标记"已完成"。验证失败 → 回滚。

```
"先标记完成再执行"：
  1. 标记 "committed"
  2. 执行副作用  ← 崩溃在这里？
  3. 验证

"先执行再标记"：
  1. 执行副作用
  2. 验证
  3. 标记 "verified"  ← 崩溃在这里？重试看到 "committed" 并跳过
```

最佳实践：执行前记录"进行中"意图，执行后验证通过才标记"已验证"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 检查点 (Checkpoint) | "保存点" | 每个图状态转换都持久化到持久存储 |
| 租约 (Least) | "工作者声明" | 工作者正在执行此运行的短期声明；崩溃时过期 |
| 前置条件 (Precondition) | "状态门" | 确认状态仍然与批准的动作一致的断言 |
| 提交后验证 | "重新读取检查" | 在目标系统中确认副作用确实发生 |
| 带内回滚 | "直接反转" | 用逆操作反转副作用（DELETE after INSERT） |
| 补偿事务 | "SAGA 反转" | 中和原始动作的新动作 |
| Article 14 | "EU AI Act 人类监督" | 可查询的检查点、经过演练的回滚、可审计的跟踪 |

---

## 📚 小结

检查点 + 租约恢复使崩溃的智能体可恢复。幂等键 + 前置条件防止双重执行和状态漂移。提交后验证确认副作用确实发生。回滚在验证失败时执行——带内、补偿或带外。Article 14 要求检查点可查询、回滚经过演练、审计跟踪在部署后存活。四个原语各捕获一个失败类别。

下一课：宪法 AI 和规则覆盖——将对齐原则编码到模型层。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。验证四种场景。对于崩溃-执行-重试场景，确认动作在整个重试过程中恰好执行一次。

2. **【实验】** 修改"先标记完成再执行"模式，使状态写入在动作之后。重新运行崩溃场景。测量有多少重复动作。

3. **【设计】** 为特定生产动作设计回滚计划（如"发布到 Slack 频道"）。分类为带内、补偿或带外。论证你的选择。

4. **【实现】** 为一个你了解的工作流识别每个状态转换。标记每个的持久化需求（持久化/不持久化）。计算你当前未持久化的数量。

5. **【实验】** 演练回滚测试：设计一个端到端测试，运行真实工作流，崩溃它，并确认回滚路径触发。测试断言什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 检查点工作流 | `code/main.py` | 幂等键 + 前置条件 + 验证 + 回滚的四种场景 |
| 技能提示词 | `outputs/skill-rollback-rehearsal.md` | 设计回滚演练测试 |

---

## 📖 参考资料

1. [官方文档] Microsoft. "Agent Framework: Checkpointing and HITL". https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop — 检查点原语和租约恢复
2. [官方文档] Cloudflare. "Agents: Human in the Loop". https://developers.cloudflare.com/agents/concepts/human-in-the-loop/ — Durable Objects 作为状态基板
3. [官方文档] EU AI Act — Article 14. https://artificialintelligenceact.eu/article/14/ — 监管基线
4. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy — 长期工作流的可靠性框架
5. [官方文档] Anthropic. "Claude Code Agent SDK: Agent Loop". https://code.claude.com/docs/en/agent-sdk/agent-loop — Claude Code Routines 的工作流形状

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
