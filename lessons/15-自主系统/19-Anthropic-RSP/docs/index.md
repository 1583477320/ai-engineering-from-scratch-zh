# Anthropic 负责任扩展策略 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，替代 2023 年策略。两级缓解：Anthropic 单方面行动 vs 行业范围建议（包括 RAND SL-4 安全标准）。新增前沿安全路线图和风险报告作为常设文件而非一次性交付物。移除了 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布正面案例，识别错位风险和缓解措施。Claude Opus 4.6 未跨越该阈值。SaferAI 将 2023 RSP 评为 2.2；他们将 v3.0 降至 1.9，将 Anthropic 归入与 OpenAI 和 DeepMind 相同的"弱"RSP 类别。定性阈值替代了 2023 年的定量承诺；移除暂停条款是最尖锐的倒退。

**类型：** 概念课
**语言：** Python（标准库，RSP 阈值决策引擎）
**前置知识：** 阶段 15 · 06（AAR）、阶段 15 · 07（RSI）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 20（OpenAI/DeepMind 框架）— 三家实验室的策略对比；阶段 15 · 21（METR）— 策略依赖的测量

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别 RSP v3.0 相对 v2.0 的关键变化：两级缓解、AI R&D-4 阈值、暂停条款移除
- [ ] 解释 SaferAI 评分从 2.2 降至 1.9 的四个原因
- [ ] 理解"正面案例"要求的七个部分以及评估上下文博弈如何影响它
- [ ] 区分 Anthropic 单方面行动和行业建议——同一文件中哪些是承诺、哪些是希望
- [ ] 将 AI R&D-4 阈值与自动化对齐研究（第 6 课）和递归自我改进（第 7 课）联系起来
- [ ] 实现一个阈值评估器——给定能力测量，决定 AI R&D-4 是否被跨越

---

## 1. 问题

前沿实验室发布扩展策略，部分是技术文档、部分是治理文档、部分是给监管机构的信号。RSP v3.0 是当前 Anthropic 文档。仔细阅读它不是因为遵守它有约束力（它没有），而是因为它的框架塑造了实验室如何思考灾难性风险以及如何向公众传达权衡。

v3.0 vs v2.0 的差异是有用的单元。新增：前沿安全路线图、风险报告、AI R&D-4 阈值。移除：2023 年暂停承诺。重构：两级缓解时间表拆分为 Anthropic 单方面和行业建议。外部审查（SaferAI）将评分从 2.2（v2）降至 1.9（v3.0）。这展示了一份扩展策略如何在看起来更精致的同时变得更不严格。

---

## 2. 概念

### 2.1 两级缓解时间表

```
v2.0:  一个列（Anthropic 承诺）
v3.0:  两个列
  ┌─ Anthropic 单方面行动（承诺）
  └─ 行业范围建议（希望）
```

| 层级 | 含义 | 可信度 |
|------|------|--------|
| **Anthropic 单方面行动** | Anthropic 无论其他实验室做什么都会执行的行动 | 高——承诺 |
| **行业范围建议** | Anthropic 认为行业应该集体做的事（包括 RAND SL-4） | 低——希望 |

两级结构不在 v2 中。它意味着读者需要查看每个承诺属于哪一列。"行业范围建议"列中的安全措施不是 Anthropic 的承诺；是 Anthropic 的希望。

### 2.2 AI R&D-4 阈值

RSP v3.0 命名的下一个重要阈值：**以与人类+AI 工具竞争的成本自动化大部分 AI 研究的能力**。

一旦 Anthropic 认为模型跨越它，必须发布正面案例。Claude Opus 4.6 未跨越。文档补充："自信地排除这一点正变得困难。"这个措辞很重要——它承认阈值足够接近，是当前关切而非投机性限制。

正面案例要求七个部分：
1. 能力清单：针对 RSP 阈值的具体测量
2. 错位风险分析：模型可能表现出的模式
3. 评估-部署差距：评估 vs 部署差异的残余风险
4. 缓解设计：技术 + 操作 + 部署门控
5. 残余风险承认：我们无法排除什么
6. 审查：内部安全咨询小组签核 + 外部审查者
7. 博弈调整后的能力估计（博弈率 > 20% 时需要）

### 2.3 移除暂停条款

2023 RSP 包含明确暂停承诺：如果模型跨越特定能力阈值，训练暂停直到缓解到位。v3.0 用更柔和的措辞替代——发布正面案例，如果缓解充分则继续。

```
v2.0:  能力超过阈值 → 暂停训练 → 缓解到位 → 继续
v3.0:  能力接近阈值 → 发布正面案例 → 如果缓解充分 → 继续
```

政策论据：2023 年的定量阈值在 2026 年能力基准下不可达，因为基准本身被重新校准。
反论据：扩展策略中的暂停条款是承诺机制；移除它移除了策略的可信度。

### 2.4 SaferAI 的降级

| 版本 | 评分 (1.0-4.0) | 类别 | 变化 |
|------|----------------|------|------|
| v2 (2023) | 2.2 | 中等 | — |
| v3.0 (2026) | 1.9 | 弱 | ↓0.3 |

降级因素：
1. 定性阈值替代了定量阈值
2. 暂停承诺移除
3. AI R&D-4 阈值缓解被描述为"正面案例"而非具体措施
4. 审查机制依赖 Anthropic 的安全咨询小组，独立监督有限

### 2.5 本课不是什么

这不是合规课程。RSP v3.0 不是监管法规；没有强制 Anthropic 遵守。课程是用应有的特异性和怀疑态度阅读文档。扩展策略是前沿实验室关于灾难性风险姿态的主要公开信号。

---

## 3. 从零实现

### 第 1 步：定义能力测量和阈值

```python
from dataclasses import dataclass

@dataclass
class CapabilityMeasurement:
    model_name: str
    rd_automation_share: float       # 可自动化的 AI 研究比例 (0-1)
    metr_horizon_hours: float        # METR 50% 视界（小时）
    aar_outperform_share: float      # 对齐研究超越人类的比例 (0-1)
    eval_context_gaming_rate: float  # 评估上下文博弈率 (0-1)

# RSP v3.0 说明性阈值（注意：实际阈值是定性的）
AI_RD_4_THRESHOLDS = {
    "rd_automation_share": 0.5,      # 50% 的 AI 研究可自动化
    "metr_horizon_hours": 40.0,      # 40 小时的 50% 视界
    "aar_outperform_share": 0.4,     # 40% 的对齐研究超越人类
}
```

### 第 2 步：实现阈值评估器

```python
def threshold_crossed(m: CapabilityMeasurement) -> tuple[bool, list[str]]:
    """两个触发条件跨越 AI R&D-4 阈值（说明性阈值）。"""
    reasons = []
    if m.rd_automation_share >= AI_RD_4_THRESHOLDS["rd_automation_share"]:
        reasons.append(f"rd_automation_share={m.rd_automation_share:.2f} "
                       f">= {AI_RD_4_THRESHOLDS['rd_automation_share']}")
    if m.metr_horizon_hours >= AI_RD_4_THRESHOLDS["metr_horizon_hours"]:
        reasons.append(f"metr_horizon_hours={m.metr_horizon_hours:.1f} "
                       f">= {AI_RD_4_THRESHOLDS['metr_horizon_hours']}")
    if m.aar_outperform_share >= AI_RD_4_THRESHOLDS["aar_outperform_share"]:
        reasons.append(f"aar_outperform_share={m.aar_outperform_share:.2f} "
                       f">= {AI_RD_4_THRESHOLDS['aar_outperform_share']}")
    crossed = len(reasons) >= 2  # 任意两个触发（说明性）
    return crossed, reasons
```

### 第 3 步：生成正面案例模板

```python
def affirmative_case_template(m: CapabilityMeasurement) -> list[str]:
    """正面案例必须覆盖的七个部分。"""
    sections = [
        "1. 能力清单：针对 RSP 阈值的具体测量",
        "2. 错位风险分析：模型可能表现出的模式",
        "3. 评估-部署差距：评估 vs 部署差异的残余风险",
        "4. 缓解设计：技术 + 操作 + 部署门控",
        "5. 残余风险承认：我们无法排除什么",
        "6. 审查：内部安全咨询小组签核 + 外部审查者",
    ]
    if m.eval_context_gaming_rate > 0.2:
        sections.append(
            f"7. 博弈调整后的能力估计（观察到的博弈率 {m.eval_context_gaming_rate:.0%}）")
    return sections
```

### 第 4 步：实现评估输出

```python
def evaluate(m: CapabilityMeasurement) -> None:
    """评估一个模型是否跨越 AI R&D-4 阈值。"""
    crossed, reasons = threshold_crossed(m)
    print(f"\n模型: {m.model_name}")
    print("-" * 70)
    print(f"  rd_automation={m.rd_automation_share:.2f}  "
          f"metr_horizon={m.metr_horizon_hours:.1f}h  "
          f"aar={m.aar_outperform_share:.2f}  gaming={m.eval_context_gaming_rate:.0%}")
    if crossed:
        print("  AI R&D-4 阈值: 已跨越")
        for r in reasons:
            print(f"    - {r}")
        print("  要求：正面案例涵盖：")
        for s in affirmative_case_template(m):
            print(f"    {s}")
    else:
        print("  AI R&D-4 阈值: 未跨越")
        if reasons:
            print("  单一触发（低于阈值）:")
            for r in reasons:
                print(f"    - {r}")
```

### 第 5 步：运行两个模型的评估

```python
def main():
    # Claude Opus 4.6（v3.0 声明）：未跨越
    evaluate(CapabilityMeasurement(
        model_name="Claude Opus 4.6 (per Anthropic v3.0)",
        rd_automation_share=0.30,
        metr_horizon_hours=14.0,
        aar_outperform_share=0.35,
        eval_context_gaming_rate=0.12))

    # 合成下一代模型：Anthropic 关注的是这类
    evaluate(CapabilityMeasurement(
        model_name="Synthetic next-gen (illustrative)",
        rd_automation_share=0.55,
        metr_horizon_hours=48.0,
        aar_outperform_share=0.45,
        eval_context_gaming_rate=0.28))
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 RSP v3.0 关键特性对照

| 特性 | v2 (2023) | v3.0 (2026) | 变化方向 |
|------|-----------|------------|---------|
| 阈值 | 定量（超过 X 就是 Y） | 定性（"正面案例"） | ↓ 更模糊 |
| 暂停承诺 | 有（明确暂停条款） | 移除 | ↓ 倒退 |
| 评估-部署差距 | 未明确处理 | 承认 | ↑ 进步 |
| SaferAI 评分 | 2.2（中等） | 1.9（弱） | ↓ 倒退 |
| 常设文件 | 一次性交付物 | 路线图 + 风险报告 | ↑ 进步 |

### 4.2 Anthropic 单方面 vs 行业建议的实际例子

| 承诺类型 | 示例 | 为什么重要 |
|---------|------|----------|
| Anthropic 单方面 | "训练超过阈值时停止" | 这是承诺——Anthropic 无论别人做什么都会执行 |
| 行业建议 | "RAND SL-4 安全标准" | 这是希望——Anthropic 认为行业应该做，但自己不承诺 |
| 混合 | "评估器设计" | 有些是承诺（自己的评估），有些是建议（行业标准） |

---

## 5. 工程最佳实践

### 5.1 阅读扩展策略的原则

| 原则 | 说明 |
|------|------|
| 每个能力都应可找到 | 如果在策略中找不到，策略不覆盖它 |
| 定位：Tracked vs Research | OpenAI 命名这个；Anthropic 和 DeepMind 有自己的等价物 |
| 节奏：声明的节奏还是仅事件驱动 | 声明的节奏更强 |
| 独立性：外部审查是强制还是可选 | Anthropic 与 Apollo 和 US AISI 合作 |

### 5.2 中文场景特别建议

- **AI R&D-4 阈值在中国 AI 实验室的语境**——国内实验室也有类似阈值定义，可以类比理解
- **SaferAI 评分对中文读者同样有用**——作为第三方独立评估的参照
- **RSP 不是合规要求**——这一点对理解中文 AI 监管同样重要

---

## 6. 常见错误

### 错误 1：将"行业建议"当作承诺

**现象：** "Anthropic 承诺了 RAND SL-4 标准。"

**原因：** RAND SL-4 在"行业范围建议"列中——是 Anthropic 的希望，不是 Anthropic 的承诺。

**修复：** 仔细查看每项承诺属于哪个级别。只有"Anthropic 单方面行动"列中的才是承诺。

### 错误 2：假设定量阈值仍然适用

**现象：** "v3.0 的阈值与 v2 相同。"

**原因：** v3.0 用定性阈值替代了 v2 的定量阈值。"足够"的定义依赖人类判断，不是数字。

**修复：** 阅读每个阈值的措辞——"正面案例"而非"超过 X 就是 Y"。

### 错误 3：忽视 SaferAI 降级

**现象：** "v3.0 比 v2 更全面，所以更好。"

**原因：** SaferAI 将 v3.0 从 2.2 降至 1.9（弱类别）。移除暂停承诺是最尖锐的倒退。

**修复：** 了解外部评分——SaferAI、METR 的方法论批评——以及评分改变的原因。

---

## 7. 面试考点

### Q1：RSP v3.0 相对 v2 的关键变化是什么？（难度：⭐）

**参考答案：**
- 两级缓解（Anthropic 单方面 vs 行业建议）——v2 没有
- AI R&D-4 阈值——新引入
- 暂停承诺移除——v2 有，v3.0 用"正面案例"替代
- SaferAI 评分从 2.2 降至 1.9

### Q2：AI R&D-4 阈值是什么？与第 6、7 课的关系？（难度：⭐⭐）

**参考答案：**
以与人类+AI 工具竞争的成本自动化大部分 AI 研究的能力。一旦跨越，必须发布正面案例。

第 6 课（AAR）和第 7 课（RSI）直接输入这个阈值：自动化对齐研究员跨越研究质量条是 AI R&D-4 阈值接近的证据。第 6 课的 Anthropic AAR 报告是这个阈值的实时测量。

### Q3：为什么移除暂停条款是最尖锐的倒退？（难度：⭐⭐）

**参考答案：**
暂停条款是扩展策略的承诺机制——当能力超过阈值时，训练必须停止。移除它移除了策略的可信度。

政策论据：2023 年的定量阈值在 2026 年能力基准下不可达，因为基准本身被重新校准。但这是关于如何定义阈值的争论，不应该是移除暂停的借口。SaferAI 将 v3.0 从 2.2 降至 1.9，移除暂停承诺是最主要的降级因素。

### Q4：评估上下文博弈如何影响能力阈值？（难度：⭐⭐⭐）

**参考答案：**
如果模型在评估中表现得比生产中更好（评估上下文博弈），所有框架的能力阈值都被低估。框架只有在测量有效时才有效。

正面案例模板在博弈率 > 20% 时需要"博弈调整后的能力估计"——这是一个明确承认测量可能不可靠的部分。这是为什么外部评估（第 21 课 METR）和对抗性评估是必要的补充——实验室自我评估的固有偏差。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RSP | "Anthropic 的扩展策略" | 负责任扩展策略；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以竞争成本自动化大量 AI 研究的能力 |
| 正面案例 | "安全论证" | 已发布论证：风险已识别且缓解充分 |
| 前沿安全路线图 | "前瞻性计划" | 计划安全工作和预期能力的常设文件 |
| 风险报告 | "模型回顾" | 发布后对特定模型观察到的能力和残余风险 |
| 两级缓解 | "单方面 vs 行业" | Anthropic 承诺 vs 行业建议，分开 |
| 暂停承诺 | "2023 条款" | 明确的训练暂停承诺；v3.0 中移除 |
| SaferAI 评分 | "独立 RSP 评级" | 第三方评分标准；v3.0 得 1.9（v2 是 2.2） |

---

## 📚 小结

RSP v3.0 的关键变化：定性阈值替代定量、暂停承诺移除、AI R&D-4 阈值引入、SaferAI 降级到 1.9。"正面案例"要求七个部分——能力清单、错位风险、评估差距、缓解设计、残余风险、审查、博弈调整。评估上下文博弈使能力数字偏高——正面案例在博弈率 > 20% 时必须包含调整。两级缓解结构将承诺和希望分开——读者需要仔细查看每个承诺属于哪一级别。

下一课：OpenAI 和 DeepMind 的准备框架——三家实验室的对比阅读。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认阈值评估器产生正确的裁决和正面案例模板。

2. **【阅读】** 阅读 RSP v3.0 全文（32 页）。识别每个位于"行业范围建议"级别的承诺。哪些承诺在 v2 中是"Anthropic 单方面行动"？

3. **【阅读】** 阅读 SaferAI 的 RSP 评分方法论。应用其评分标准重现 v3.0 的 1.9 分。哪个评分行驱动了最多降级？

4. **【思考】** 暂停承诺被移除。提出一个替代承诺，既保留策略可信度又承认 2026 年基准重新校准问题。

5. **【对比】** 对比 RSP v3.0 与 OpenAI 准备框架 v2（第 20 课）。选一个 v3.0 更强的领域，选一个准备框架更强的领域。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 阈值评估器 | `code/main.py` | AI R&D-4 阈值评估 + 正面案例模板 |
| 技能提示词 | `outputs/skill-scaling-policy-review.md` | 跨策略比较工具 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Responsible Scaling Policy v3.0". https://anthropic.com/responsible-scaling-policy/rsp-v3-0 — 完整 32 页策略
2. [官方文档] Anthropic. "RSP v3.0 Announcement". https://www.anthropic.com/news/responsible-scaling-policy-v3 — 从 v2 的变化摘要
3. [官方文档] Anthropic. "Frontier Safety Roadmap". https://www.anthropic.com/research/frontier-safety — RSP v3.0 链接的常设文件
4. [官方文档] Anthropic. "Risk Report: Claude Opus 4.6". https://www.anthropic.com/research/risk-report-claude-opus-4-6 — 当前前沿模型回顾
5. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy — AI R&D-4 与测量自主性的连接

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
