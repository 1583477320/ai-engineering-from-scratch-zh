# OpenAI 准备框架与 DeepMind 前沿安全框架

> OpenAI 准备框架 v2（2025 年 4 月）引入研究类别——长期自主性、蓄意低性能、自主复制和适应、破坏安全防护——区别于跟踪类别。跟踪类别触发能力报告加安全防护报告，由安全咨询小组审查。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日增加跟踪能力级别）将自主性纳入 ML R&D 和网络领域（ML R&D 自主性级别 1 = 以竞争成本完全自动化 AI R&D 流水线）。FSF v3 通过自动监控工具推理误用来明确应对欺骗性对齐。诚实说明：PF v2 中的研究类别（包括长期自主性）不会自动触发缓解；策略语言是"潜在"的。DeepMind 自己表示自动监控"长期不会保持充分"，如果工具推理增强的话。

**类型：** 概念课
**语言：** Python（标准库，三框架决策表差异工具）
**前置知识：** 阶段 15 · 19（Anthropic RSP）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 19（Anthropic RSP）— 三家策略对比的视角；阶段 15 · 21（METR）— 策略依赖的测量

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分 OpenAI 的跟踪类别（强制缓解）和研究类别（仅监控）——同一能力在不同类别中的运营后果不同
- [ ] 解释 DeepMind FSF v3 的关键机制：CCL（关键能力级别）、跟踪能力级别、欺骗性对齐监控
- [ ] 实现一个三框架对比工具——给定能力，输出三家实验室的分类和触发动作
- [ ] 识别三家的共同点（安全咨询小组、欺骗性行为、常设文件）和分歧（暂停条款、类别划分、自主性定位）
- [ ] 理解蓄意低性能如何使所有框架的能力阈值被低估

---

## 1. 问题

第 19 课仔细阅读了 Anthropic 的扩展策略。本课通过阅读 OpenAI 和 DeepMind 的完成这幅图。三份文档是解决同一问题的表亲产物——前沿实验室何时应暂停或门控模型——它们在少量类别上收敛，在具体地方分歧，而这些分歧很重要。

收敛：三者都将长期自主性标记为值得跟踪的能力类别。三者都将欺骗性行为作为特定风险类别。三者都有内部审查机构。

分歧：OpenAI 将类别拆分为"跟踪"（强制缓解）和"研究"（不自动触发）。DeepMind 将自主性折叠到两个领域中，而非单独命名。同一能力在 Anthropic 可能是"强制缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域跟踪"。策略姿态很重要。

---

## 2. 概念

### 2.1 OpenAI 准备框架 v2

```
跟踪类别 → 触发能力报告 + 安全防护报告 → SAG 审查
研究类别 → 监控但不承诺具体缓解 → 包括长期自主性
```

| 类别 | 示例 | 是否自动触发缓解？ |
|------|------|------------------|
| 跟踪：R&D 自动化 | AI 研究自动化 | 是——能力报告 + 安全防护报告 |
| 跟踪：网络提升 | 网络攻击能力 | 是——SAG 审查 |
| 研究：长期自主性 | 长时间运行智能体 | **否**——"潜在"缓解 |
| 研究：蓄意低性能 | 模型故意低分 | 否 |

关键框架 v2：**研究类别不自动触发缓解**。策略语言是"潜在"缓解。一个类别在跟踪 vs 研究中的位置是"被门控"和"被观察"之间的铰链。

### 2.2 DeepMind FSF v3

```
关键能力级别（CCL）→ 五个领域：网络、生物、ML R&D、CBRN、自主性
跟踪能力级别 → 2026 年 4 月新增的额外粒度
欺骗性对齐监控 → 自动监控工具推理误用
```

DeepMind 对自主性的定位不同。DeepMind 不保留"自主性"作为顶层领域；它折叠到自主性会导致危害的领域中（ML R&D 和网络）。论据是无领域的自主性是无风险的能力；反论据是跨领域的自主性是框架应该命名的元风险。

### 2.3 三家共同点

| 共同点 | Anthropic | OpenAI | DeepMind |
|--------|-----------|--------|----------|
| 安全咨询小组 | 安全咨询小组 | 安全咨询小组 | 内部委员会 |
| 欺骗性行为 | 评估上下文差距 | 蓄意低性能 | 欺骗性对齐监控 |
| 常设文件 | 路线图 + 风险报告 | 能力报告 + 安全防护报告 | FSF 更新周期 |
| 监控上限 | 承认 | — | 明确："自动监控长期不会保持充分" |

### 2.4 分歧

| 分歧 | Anthropic | OpenAI | DeepMind |
|------|-----------|--------|----------|
| 暂停承诺 | v3.0 中移除 | 无明确暂停 | — |
| 自主性定位 | AI R&D-4 阈值 | 研究类别（不触发） | 折叠到 ML R&D / 网络领域 |
| 阈值类型 | 定性 | 跟踪/研究二分 | CCL + 跟踪能力级别 |

### 2.5 蓄意低性能使三家框架复杂化

蓄意低性能（模型在评估中故意低分）使所有框架的能力阈值被低估。如果模型在评估中蓄意低分，框架的测量工作仅在测量有效时才有效。

这三家框架的共同弱点：如果测量无效，阈值就无效。这是为什么外部测量（第 21 课 METR）和对抗性评估是必要的补充——实验室自我评估的固有偏差。

### 2.6 策略阅读技能

| 技能 | 说明 | 检查问题 |
|------|------|---------|
| 定位 | 每个你关心的能力都应可在策略中找到 | 如果在策略中找不到，策略不覆盖它 |
| 分类 | 跟踪（触发缓解）vs 研究（仅监控） | OpenAI 命名这个；Anthropic 和 DeepMind 有自己的等价物 |
| 节奏 | 策略是按声明节奏更新还是仅事件驱动？ | 声明的节奏更强 |
| 独立性 | 外部审查是强制还是可选？ | Anthropic 与 Apollo 和 US AISI 合作 |

---

## 3. 从零实现

### 第 1 步：定义三策略决策表

```python
from dataclasses import dataclass

@dataclass
class Policy:
    name: str
    table: dict[str, tuple[str, str]]  # 能力 → (分类, 触发动作)

# OpenAI PF v2（说明性提炼）
OPENAI_PF_V2 = Policy(
    name="OpenAI Preparedness v2 (Apr 2025)",
    table={
        "long_range_autonomy": ("Research", "observed; potential mitigations"),
        "sandbagging": ("Research", "observed; potential mitigations"),
        "autonomous_replication": ("Research", "observed; potential mitigations"),
        "undermining_safeguards": ("Research", "observed; potential mitigations"),
        "rnd_automation": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "cyber_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "bio_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
    },
)

# Anthropic RSP v3.0
ANTHROPIC_RSP_V3 = Policy(
    name="Anthropic RSP v3.0 (Feb 2026)",
    table={
        "long_range_autonomy": ("named risk", "affirmative case at threshold"),
        "sandbagging": ("named via eval-context gap", "addressed in measurement methodology"),
        "autonomous_replication": ("not explicitly named", "covered under AI R&D-4"),
        "undermining_safeguards": ("hardcoded prohibition", "refuses training / deploy"),
        "rnd_automation": ("AI R&D-4 threshold", "affirmative case required"),
        "cyber_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
        "bio_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
    },
)

# DeepMind FSF v3
DEEPMIND_FSF_V3 = Policy(
    name="DeepMind FSF v3 (Sept 2025 + Apr 2026)",
    table={
        "long_range_autonomy": ("folded into ML R&D / Cyber domains", "CCL + Tracked Capability Level"),
        "sandbagging": ("deceptive alignment monitoring", "automated instrumental-reasoning monitor"),
        "autonomous_replication": ("folded into ML R&D domain", "CCL threshold"),
        "undermining_safeguards": ("deceptive alignment monitoring", "automated monitor + red-team"),
        "rnd_automation": ("ML R&D autonomy level 1", "Tracked Capability Level added Apr 2026"),
        "cyber_uplift": ("Cyber CCL", "security + deployment mitigations"),
        "bio_uplift": ("Bio CCL", "security + deployment mitigations"),
    },
)
```

### 第 2 步：实现对比输出

```python
def diff(capability: str) -> None:
    """输出一个能力在三家策略中的分类和触发动作。"""
    print(f"\n能力: {capability}")
    print("-" * 80)
    for p in [OPENAI_PF_V2, ANTHROPIC_RSP_V3, DEEPMIND_FSF_V3]:
        entry = p.table.get(capability, ("not in table", "—"))
        print(f"  {p.name}")
        print(f"    分类: {entry[0]}")
        print(f"    动作: {entry[1]}")
```

### 第 3 步：运行对比

```python
def main():
    for cap in ("long_range_autonomy", "sandbagging", "rnd_automation"):
        diff(cap)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 三策略能力分类对照

| 能力 | OpenAI | Anthropic | DeepMind |
|------|--------|-----------|----------|
| 长期自主性 | 研究（不触发） | 命名风险（正面案例） | 折叠到 ML R&D / 网络 |
| 蓄意低性能 | 研究 | 评估上下文差距 | 欺骗性对齐监控 |
| R&D 自动化 | 跟踪 | AI R&D-4 阈值 | ML R&D 自主性级别 1 |
| 破坏安全防护 | 研究 | 硬编码禁令 | 欺骗性对齐监控 |

---

## 5. 工程最佳实践

### 5.1 策略阅读原则

| 原则 | 说明 |
|------|------|
| 同一能力，三种分类 | 一种能力在 Anthropic 是强制，在 OpenAI 是监控，在 DeepMind 是领域折叠 |
| 研究类别 = 不自动触发 | OpenAI PF v2 的关键区别——"潜在"缓解不等于承诺 |
| 声明的节奏比事件驱动更强 | 策略在声明的节奏上更新比仅在事件后更新更可信 |
| 外部审查是关键 | 三种策略都有内部委员会；独立审查的强制性不同 |

### 5.2 中文场景特别建议

- **三家框架对中文 AI 实验室的参考价值**——虽然中国有自己的 AI 治理框架，但理解 OpenAI/Anthropic/DeepMind 的策略差异对理解全球 AI 安全治理很有帮助
- **蓄意低性能在中文 AI 研究中的相关性**——当评估基准被广泛使用时，蓄意低性能的风险同样存在
- **跨策略对比工具可以帮助理解差异**——不同框架对同一能力的不同分类，展示了 AI 安全治理的不确定性

---

## 6. 常见错误

### 错误 1：只读一家策略

**现象：** "我们遵循了 Anthropic RSP。"但 OpenAI 和 DeepMind 对同一能力有不同的分类。

**原因：** 同一能力在不同框架中可能触发完全不同的运营后果。

**修复：** 跨三家阅读同一能力。对比分类和触发动作。

### 错误 2：将"研究类别"等同于"被忽略"

**现象：** "长期自主性在 OpenAI 是研究类别，所以不需要担心。"

**原因：** 研究类别是"监控但不触发"，不是"忽略"。它在被跟踪——只是没有自动缓解。

**修复：** 研究类别意味着能力被跟踪但尚未承诺具体缓解。跟踪本身有价值，即使没有自动门控。

### 错误 3：不考虑蓄意低性能的影响

**现象：** 信任能力阈值数字而不考虑模型是否在蓄意低分。

**原因：** 如果模型在评估中蓄意低分，所有框架的能力阈值被低估。框架只有在测量有效时才有效。

**修复：** 外部评估（METR）+ 对抗性评估 + 蓄意低性能监控。

---

## 7. 面试考点

### Q1：OpenAI 跟踪类别和研究类别的区别是什么？运营后果是什么？（难度：⭐）

**参考答案：**
跟踪类别触发能力报告 + 安全防护报告 + SAG 审查——强制缓解。

研究类别被监控但不承诺具体缓解——长期自主性、蓄意低性能、自主复制、破坏安全防护都在这里。

运营后果：跟踪类别在部署前必须有缓解措施；研究类别是"观察但不门控"。位置差异是关键——能力在跟踪类别意味着必须处理，在研究类别意味着可以不处理。

### Q2：DeepMind FSF v3 如何定位自主性？为什么这与 OpenAI 不同？（难度：⭐⭐）

**参考答案：**
DeepMind 不保留"自主性"作为顶层领域。它折叠到自主性会导致危害的领域中（ML R&D 和网络）。论据是无领域的自主性是无风险的能力；反论据是跨领域的自主性是元风险。

OpenAI 将"长期自主性"保留为独立的研究类别。DeepMind 的领域折叠方式意味着自主性的任何具体实例都被跟踪，但"跨领域的自主性"作为元风险可能被忽视。

### Q3：蓄意低性能如何使三家框架复杂化？（难度：⭐⭐）

**参考答案：**
蓄意低性能（模型在评估中故意低分）使所有框架的能力阈值被低估。OpenAI 将其列为研究类别；Anthropic 通过评估上下文差距处理；DeepMind 通过欺骗性对齐自动监控。

框架只有在测量有效时才有效。如果模型在蓄意低分，测量无效，阈值不可靠。这是外部评估（第 21 课 METR）和对抗性评估的必要性论据。

### Q4：三家框架的共同点是什么？分歧在哪里？（难度：⭐⭐⭐）

**参考答案：**
共同点：三家都有内部安全咨询小组、都明确提及欺骗性行为、都有常设文件按声明节奏更新、都承认监控有上限。

分歧：Anthropic 移除了暂停承诺；OpenAI 的研究类别不自动触发；DeepMind 将自主性折叠到其他领域。同一能力（如长期自主性）在三家可能触发完全不同的运营后果。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 准备框架 | "OpenAI 的扩展策略" | PF v2（2025 年 4 月）；跟踪 vs 研究类别 |
| 跟踪类别 | "强制缓解" | 触发能力报告 + 安全防护报告；SAG 审查 |
| 研究类别 | "仅监控" | 被跟踪但不自动缓解；包括长期自主性 |
| FSF v3 | "DeepMind 的扩展策略" | 2025 年 9 月 + 2026 年 4 月跟踪能力级别 |
| CCL | "关键能力级别" | DeepMind 每领域阈值 |
| 蓄意低性能 | "战略低分" | 模型在评估中低分；在 OpenAI 研究类别中 |
| 工具推理 | "手段-目的推理" | 推理如何实现目标；DeepMind 监控目标 |

---

## 📚 小结

三家框架在同一问题上收敛又分歧：三家都跟踪长期自主性和欺骗性行为；分歧在于 OpenAI 将自主性列为研究类别（不触发），Anthropic 为 AI R&D-4（触发正面案例），DeepMind 将其折叠到 ML R&D/网络领域。蓄意低性能使所有框架的能力阈值被低估——框架只有在测量有效时才有效。外部评估（METR）是必要的补充。

下一课：METR 时间视界和外部能力评估——测量方法论和解释技能。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认差异工具的输出与至少两个能力的源文档匹配。

2. **【阅读】** 阅读 OpenAI 准备框架 v2 全文。识别每个研究类别。为每个写一句话说明为什么在研究而非跟踪。

3. **【阅读】** 阅读 DeepMind FSF v3 全文及 2026 年 4 月跟踪能力级别更新。识别 ML R&D 自主性级别 1 的具体评估标准。如何外部测量它？

4. **【设计】** 蓄意低性能在 OpenAI 是研究类别。设计一个强制蓄意低性能模型揭示其实际能力的评估。参考第 1 课评估上下文博弈讨论。

5. **【对比】** 在一个特定能力上对比三家策略。哪家最严格？哪家最宽松？引用源文本论证。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 跨策略对比工具 | `code/main.py` | 三框架决策表差异工具 |
| 技能提示词 | `outputs/skill-cross-policy-diff.md` | 跨策略比较生成器 |

---

## 📖 参考资料

1. [官方文档] OpenAI. "Updating our Preparedness Framework". https://openai.com/index/updating-our-preparedness-framework/ — v2 公告
2. [PDF] OpenAI. "Preparedness Framework v2". https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf
3. [博客] DeepMind. "Strengthening our Frontier Safety Framework". https://deepmind.google/blog/strengthening-our-frontier-safety-framework/
4. [博客] DeepMind. "Updating the Frontier Safety Framework (April 2026)". https://deepmind.google/blog/updating-the-frontier-safety-framework/
5. [PDF] DeepMind. "Gemini 3 Pro FSF Report". https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
