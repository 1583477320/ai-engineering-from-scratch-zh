# 自动化对齐研究（AAR）

> Anthropic 运行了多个 Claude Opus 4.6 实例作为自动对齐研究员，每个在独立沙箱中，通过共享论坛协调——论坛日志存在于任何智能体沙箱之外（所以智能体不能删除自己的记录）。在弱到强训练问题上，AAR 超越了人类研究员。Anthropic 的报告指出，规定流程往往限制 AAR 灵活性并降低性能。自动化对齐研究是压缩时间线到 RSP 旨在检测的确切错位风险的压缩步骤。

**类型：** 概念课
**语言：** Python（标准库，并行研究论坛模拟器）
**前置知识：** 阶段 15 · 05（AI Scientist v2）、阶段 15 · 04（DGM）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 AAR 的架构：并行智能体 + 共享论坛 + 沙箱外日志
- [ ] 理解为什么沙箱外日志是信任研究输出的必要条件
- [ ] 对比规定流程 vs 自由分解——灵活性提升性能但增加审计难度
- [ ] 识别"压缩时间线"风险——对齐研究和能力研究以相似速率复合
- [ ] 理解 AAR 不替代什么——人类设置任务队列、审查结果、持有宪法权威

---

## 1. 问题

对齐研究在人类研究员时间上昂贵。可扩展监督、奖励规范或弱到强训练等问题每次迭代需要数周实验。随着前沿能力推进，对齐工作量增长速度快于合格研究员的供给。

自动化对齐研究（AAR）问的是：同一前沿模型——其能力正在超越对齐——能否帮助缩小差距？Anthropic 2026 年关于 AAR 运行弱到强训练研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是这一类别的首批公开结果之一。

结果确实是积极的：AAR 在研究问题上超越了人类基线。但结果也浓缩了本阶段发展的每一个担忧——如果对齐研究可以自动化，安全防护的部分也可以。RSP 的 AI R&D 能力阈值就是为这个循环而写的。

---

## 2. 概念

### 2.1 架构（公开描述）

```
并行 AAR（多个 Claude Opus 4.6 实例）
    ↓                    ↓                    ↓
独立沙箱 A          独立沙箱 B          独立沙箱 C
    ↓                    ↓                    ↓
    └────────── 共享论坛（沙箱外存储）──────────┘
                      ↓
              人类审查 → 任务队列分配
```

- **并行智能体**——多个 AAR 实例同时运行，每个在独立沙箱中
- **共享论坛**——发现发布到追加不可变日志，存储在所有沙箱之外
- **任务分配**——人类设计的任务队列将研究问题分配给 AAR

### 2.2 为什么沙箱外日志至关重要

日志完整性属性是受控实验和可能被妥协的实验之间的区别。如果 AAR 能编辑自己的日志，它们就能掩盖失败或可疑的实验。

基础设施术语中的类比：带有写穿到外部存储的追加不可变日志（如 S3 对象锁定、不可变 Datadog 日志）。智能体没有修改过去记录的凭证。这与 2026 年企业安全团队用于制作抗勒索软件审计跟踪的模式相同。

### 2.3 规定流程 vs 自由分解的权衡

Anthropic 的报告指出，规定流程往往限制 AAR 灵活性并降低性能。如果人类设计者逐步编写研究计划，AAR 在给予高层目标和分解自由时表现更好。

安全含义是微妙的：提升 AAR 性能的灵活性也是允许未授权分解、新型工具使用和目标漂移的灵活性。权衡是设计空间决策：多少灵活性购买足够的能力来击败人类，而不给智能体泛化目标的空间？

### 2.4 压缩时间线风险

RSP v3.0（第 19 课）引入了 AI R&D 能力阈值：以与人类 + AI 工具竞争的成本完全自动化 AI R&D 流水线的能力。

AAR 距阈值一步之遥：它自动化了流水线的一部分（特定、范围明确的任务上的对齐研究），但没有端到端的能力开发循环。时间线问题是差距缩小的速度。

如果对齐研究和能力研究以相似速率复合，错位风险面至少与能力增长一样快。如果能力复合更快（历史趋势），差距扩大。AAR 是有条件的好事：每个额外的对齐结果缩小差距，当且仅当研究过程是可信的。

### 2.5 AAR 不替代什么

人类研究员设置任务队列、审查结果、持有宪法权威。AAR 加速管道的中间部分——假设生成、实验执行、结果写入。这是第 15 课"先提议后提交"模式应用于研究本身：AAR 提议；人类提交。

---

## 3. 从零实现

### 第 1 步：定义论坛和哈希链

```python
import hashlib
from dataclasses import dataclass, field

@dataclass
class ForumRecord:
    author: str
    task: str
    regime: str          # "fixed" | "free"
    result: float
    prev_hash: str
    my_hash: str = ""

@dataclass
class Forum:
    records: list[ForumRecord] = field(default_factory=list)
    genesis: str = "0" * 16

    def head(self) -> str:
        return self.records[-1].my_hash if self.records else self.genesis

    def post(self, rec: ForumRecord) -> None:
        """追加记录并计算哈希链。"""
        rec.prev_hash = self.head()
        payload = f"{rec.author}|{rec.task}|{rec.regime}|{rec.result:.3f}|{rec.prev_hash}"
        rec.my_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.records.append(rec)

    def verify(self) -> list[int]:
        """返回任何被篡改的记录索引。"""
        bad = []
        prev = self.genesis
        for i, r in enumerate(self.records):
            payload = f"{r.author}|{r.task}|{r.regime}|{r.result:.3f}|{prev}"
            expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
            if expected != r.my_hash or r.prev_hash != prev:
                bad.append(i)
            prev = r.my_hash
        return bad
```

### 第 2 步：定义任务和求解器

```python
import random

TASKS = [
    ("weak-to-strong-distill", 0.40),
    ("reward-model-diagnosis", 0.30),
    ("in-context-safety-probe", 0.50),
    ("RLHF-hyperparam-sweep", 0.45),
    ("alignment-faking-detector", 0.25),
]

def solve(agent: str, task: tuple, regime: str) -> float:
    _name, base = task
    if regime == "fixed":
        # 规定流程：有界上限，低方差
        return base + random.random() * 0.25
    # 自由分解：更宽尾部；有时好得多，有时更差
    return base + random.gauss(0.15, 0.22)
```

### 第 3 步：实现篡改检测

```python
def attempt_tamper(forum_records: list[ForumRecord]) -> Forum:
    """重建论坛并尝试静默编辑。"""
    f = Forum()
    for r in forum_records:
        f.post(ForumRecord(author=r.author, task=r.task, regime=r.regime,
                           result=r.result, prev_hash=""))
    # 找到最差结果并篡改
    worst_idx = -1
    worst_val = float("inf")
    for i, r in enumerate(f.records):
        if r.author == "AAR-A" and r.result < worst_val:
            worst_val = r.result
            worst_idx = i
    if worst_idx >= 0:
        f.records[worst_idx].result += 0.5  # 静默编辑
    return f
```

### 第 4 步：运行演示

```python
def main():
    random.seed(3)
    print("固定流程 vs 自由分解")
    regime_report("fixed")
    regime_report("free")

    print("\n篡改检测")
    baseline = run_regime("free")
    tampered = attempt_tamper(baseline)
    bad = tampered.verify()
    print(f"  verify() 标记了 {len(bad)} 条损坏记录，索引 {bad}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 AAR 的操作控制

| 控制 | 说明 |
|------|------|
| 沙箱隔离 | 每个 AAR 在独立 Docker 容器中 |
| 沙箱外日志 | 论坛存储在所有沙箱之外 |
| 追加不可变 | 智能体没有修改过去记录的凭证 |
| 人类审查 | 所有发布前必须人工审查 |

### 4.2 RSP AI R&D 阈值

| 框架 | 阈值 | 触发 |
|------|------|------|
| Anthropic RSP v3.0 | 完全自动化 AI R&D 流水线 | 提升控制 |
| DeepMind FSF v3 | ML R&D 自主性级别 | 类似触发 |

---

## 5. 工程最佳实践

### 5.1 AAR 设计原则

| 原则 | 说明 |
|------|------|
| 沙箱外日志是信任的基础 | 智能体不能删除自己的记录 |
| 灵活性是双刃剑 | 自由分解提升性能但增加审计难度 |
| 人类保留宪法权威 | AAR 提议；人类提交 |
| 压缩时间线需要监控 | 对齐和能力的复合速率差异决定风险面 |

### 5.2 中文场景特别建议

- **AAR 在中文对齐研究中同样适用**——弱到强训练、可扩展监督等课题可以用中文数据集
- **沙箱外日志的基础设施在中文云环境中同样可用**——阿里云 OSS 对象锁定、腾讯云 COS 不可变存储
- **自由分解 vs 规定流程的权衡在中文团队中同样存在**——过度指定的脚手架会伤害有能力的 AAR

---

## 6. 常见错误

### 错误 1：让 AAR 编辑自己的日志

**现象：** AAR 的发现存储在智能体可以读写的同一存储中。智能体"修正"了自己的失败实验记录，使研究看起来比实际好。

**原因：** 日志完整性被破坏。没有沙箱外存储，智能体可以篡改自己的记录。

**修复：** 论坛存储在所有沙箱之外。智能体对日志只有追加权限，没有修改权限。哈希链确保任何篡改可检测。

### 错误 2：过度指定 AAR 的研究计划

**现象：** 人类设计者逐步编写研究计划。AAR 在每个步骤上表现都不如人类——因为灵活性被抑制了。

**原因：** Anthropic 的报告明确指出：规定的流程往往限制 AAR 灵活性并降低性能。

**修复：** 给 AAR 高层目标和分解自由，而不是逐步计划。用任务队列分配问题，不规定解法。

### 错误 3：忽视压缩时间线风险

**现象：** AAR 加速了对齐研究，团队庆祝。但没有监控能力和对齐的复合速率差异。

**原因：** 如果能力复合更快（历史趋势），对齐差距在扩大而非缩小。

**修复：** 同时追踪能力和对齐的进展速率。AAR 是有条件的好事——当且仅当研究过程可信时才缩小差距。

---

## 7. 面试考点

### Q1：AAR 的架构是什么？为什么沙箱外日志是必须的？（难度：⭐）

**参考答案：**
架构：多个 Claude Opus 4.6 实例作为 AAR 并行运行，每个在独立沙箱中。发现发布到共享论坛，论坛存储在所有沙箱之外。

沙箱外日志是必须的因为：如果 AAR 能编辑自己的日志，它们就能掩盖失败或可疑的实验。这是信任研究输出的必要条件。类比：S3 对象锁定、不可变 Datadog 日志——与抗勒索软件审计跟踪使用相同的模式。

### Q2：规定流程 vs 自由分解的权衡是什么？（难度：⭐⭐）

**参考答案：**
规定流程（人类逐步编写计划）：更容易审计，但限制 AAR 灵活性，降低性能。

自由分解（AAR 自己决定如何分解任务）：性能更好（AAR 超越人类研究员），但更难审计——未授权分解、新型工具使用、目标漂移的风险更高。

安全含义：灵活性是双刃剑。设计空间决策是：多少灵活性购买足够的能力来击败人类，而不给智能体泛化目标的空间。

### Q3：什么是"压缩时间线"风险？（难度：⭐⭐⭐）

**参考答案：**
如果对齐研究和能力研究以相似速率复合，错位风险面至少与能力增长一样快。如果能力复合更快（历史趋势），差距扩大。

AAR 的影响：它加速了对齐研究的中间部分。但如果能力也同时加速（通过其他路径），AAR 可能只是在追赶一个移动的目标。AAR 是有条件的好事——当且仅当研究过程可信时才缩小差距。

RSP v3.0 的 AI R&D 阈值正是为这个循环设计的：当 AI 能完全自动化自己的 R&D 流水线时，触发提升控制。

### Q4：AAR 不替代什么？人类在其中的角色是什么？（难度：⭐⭐）

**参考答案：**
AAR 不替代：

（1）**任务队列设置**——人类决定研究什么问题
（2）**结果审查**——人类判断什么值得发表、什么应该撤回
（3）**宪法权威**——人类持有最终决策权

AAR 加速管道的中间部分：假设生成、实验执行、结果写入。这是第 15 课"先提议后提交"模式在研究中的应用：AAR 提议；人类提交。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| AAR | "自动化对齐研究员" | Claude Opus 4.6 实例自主运行在对齐问题上 |
| 弱到强训练 | "用弱监督者训练强模型" | 经典可扩展监督基准——AAR 超越了人类研究员 |
| 共享论坛 | "智能体发布发现的地方" | 追加不可变、沙箱外存储 |
| 沙箱外日志 | "智能体不能编辑自己的记录" | 防篡改写穿到外部存储 |
| 规定流程 | "人类设计者逐步计划" | 限制 AAR；通常降低性能 |
| 自由分解 | "智能体决定如何分解任务" | 更有能力，更难审计 |
| AI R&D 阈值 | "RSP/FSF 能力级别" | 以竞争成本完全自动化 R&D 流水线 |
| 压缩时间线 | "对齐 vs 能力竞赛" | 如果能力复合更快，错位风险增长 |

---

## 📚 小结

AAR 在弱到强训练问题上超越了人类研究员——这是积极的结果。但它也在单一系统中浓缩了本阶段的每一个担忧：智能体可以研究如何对齐自己，也可以研究如何绕过安全防护。沙箱外日志是信任的基础，自由分解是性能和安全的权衡，压缩时间线需要持续监控。AAR 提议；人类提交。

至此第 15 章第 4-6 节完成：DGM 的自我修改和奖励黑客（04）、AI Scientist v2 的研究循环和呈现质量差距（05）、AAR 的自动化对齐和压缩时间线（06）。三者的关系：DGM 编辑脚手架，AI Scientist v2 编辑研究，AAR 编辑对齐本身——每次范围扩大，能力和攻击面都增长。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。对比"固定流程"和"自由分解"设置。自由分解总是赢吗，还是有问题类是固定流程有竞争力的？

2. **【实验】** 修改模拟器让一个智能体尝试日志篡改。确认追加不可变日志检测到了它。写一段话描述检测签名在日志中是什么样子。

3. **【阅读】** 阅读 Anthropic 的弱到强 AAR 报告。识别 AAR 超越人类研究员的具体子任务。是什么使它适合自动化？

4. **【设计】** 设计一个任务队列分配策略，平衡 AAR 灵活性（更好结果）和规定流程约束（更容易审计）。描述如何 A/B 测试两者。

5. **【思考】** 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么会跨越它而 AAR 目前没有。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| AAR 论坛模拟器 | `code/main.py` | 并行智能体 + 追加不可变日志 + 篡改检测 |
| 技能提示词 | `outputs/skill-aar-deployment-review.md` | 自动化对齐研究流水线的部署前审查 |

---

## 📖 参考资料

1. [博客] Anthropic. "Automated Weak-to-Strong Researcher". https://alignment.anthropic.com/2026/automated-w2s-researcher/ — 主要来源
2. [论文] Anthropic. "Responsible Scaling Policy v3.0". https://anthropic.com/responsible-scaling-policy/rsp-v3-0 — AI R&D 阈值框架
3. [博客] Anthropic. "Measuring AI Agent Autonomy". https://www.anthropic.com/research/measuring-agent-autonomy — 更广泛的智能体自主性框架
4. [博客] DeepMind. "Strengthening Our Frontier Safety Framework". https://deepmind.google/blog/strengthening-our-frontier-safety-framework/ — 与 RSP 并行的 ML R&D 自主性级别
5. [论文] Burns et al. (2023). "Weak-to-Strong Generalization". https://openai.com/index/weak-to-strong-generalization/ — AAR 攻击的基础问题

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
