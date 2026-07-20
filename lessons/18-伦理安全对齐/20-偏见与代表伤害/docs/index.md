# LLM 中的偏见与代表伤害

> Gallegos 等人（Computational Linguistics 2024, arXiv:2309.00770）将代表伤害（刻板印象、抹除）与分配伤害（不平等资源分配）区分开来，并将评估指标分类为基于嵌入、基于概率和基于生成文本三类。2024-2025 经验：An 等人（PNAS Nexus, 2025）跨 GPT-3.5/4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 测量了 20 个入门级职位的自动化简历评估中的交叉性别×种族偏见。WinoIdentity（COLM 2025）引入了基于不确定性的交叉公平性评估。Yu & Ananiadou 2025 在 MLP 层中发现了性别神经元；Ahsan & Wallace 2025 使用 SAE 揭示临床种族偏见；Zhou et al. 2024（UniBias）通过操纵注意力头实现去偏。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05（词嵌入）、阶段 18 · 01（指令遵循）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义代表伤害和分配伤害，并在 LLM 部署中各给一个例子
- [ ] 说出 Gallegos et al. 2024 的三类评估指标并描述每类的一个指标
- [ ] 描述交叉性以及 WinoIdentity 的基于不确定性公平性评估如何解决单轴评估的缺口
- [ ] 描述两种机制可解释性偏见方法（性别神经元、SAE 特征、注意力头操纵）

---

## 1. 问题

之前的课程覆盖了故意伤害（越狱、欺骗）和安全治理。偏见是在无意中产生的伤害——来自训练数据分布、提示词框架和累积设计选择。测量和减少它是与对抗鲁棒性不同的方法论挑战。

---

## 2. 概念

### 2.1 代表伤害 vs 分配伤害

- **代表伤害：** 刻板印象、抹除、贬损描绘。将护士描绘为全是女性就是产生代表伤害
- **分配伤害：** 不平等的物质结果。系统性地给黑人申请者更低的简历评分就是产生分配伤害

### 2.2 三类评估指标（Gallegos et al. 2024）

- **基于嵌入的：** 预 RLHF 嵌入上的 WEAT 风格测试。测量身份词和属性词之间的统计关联。局限：测量表示而非行为
- **基于概率的：** 刻板印象确认 vs 刻板印象违反完成的对数似然。解码器侧测量
- **基于生成文本的：** 在生成文本上的下游任务测量。简历评分、推荐写作、对话。最生态有效但最难复现

### 2.3 交叉性

对"性别"的偏见评估漏掉了只在（性别、种族）对上触发的偏见。An et al. 2025 发现 GPT-4o 在简历评分中对黑人女性的惩罚比黑人男性和白人女性分别更大。单轴评估无法捕获这一点。

WinoIdentity（COLM 2025）引入基于不确定性的交叉公平性——测量模型在交叉身份元组上的不确定性是否不同。

### 2.4 机制可解释性方法

- **性别神经元（Yu & Ananiadou 2025）。** 特定 MLP 神经元与性别特定行为相关。消融这些神经元以有限的能力成本减少性别差距指标
- **SAE 临床种族偏见（Ahsan & Wallace 2025）。** 稀疏自编码器特征将内部分解为可解释维度；可以识别和抑制种族相关特征
- **UniBias（Zhou et al. 2024）。** 零样本去偏——通过操纵注意力头减少偏见

### 2.5 元批评

10 年文献回顾（arXiv:2508.11067, 2025）发现该领域不成比例地关注二元性别偏见。残疾、宗教、移民身份、多语言身份等其他轴线收到的关注少得多。窄聚焦可能通过忽视伤害边缘群体——在二元性别上良好去偏的模型可能在无人检查的维度上严重偏见。

---

## 3. 从零实现

### WEAT 风格嵌入偏见探针

```python
import math


class EmbeddingBiasProbe:
    """简化版 WEAT 偏见探针——基于共现嵌入。"""

    def __init__(self):
        # 简化的嵌入空间（3维）
        self.embeddings = {
            "医生": [0.8, 0.2, 0.1],
            "护士": [0.2, 0.8, 0.1],
            "工程师": [0.7, 0.3, 0.6],
            "男性": [0.7, 0.3, 0.5],
            "女性": [0.3, 0.7, 0.5],
            "积极": [0.9, 0.1, 0.2],
            "消极": [0.1, 0.9, 0.2],
        }

    def cosine_sim(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

    def measure_bias(self, target_words, attribute_words):
        """测量目标词和属性词之间的平均相似度。"""
        scores = []
        for t in target_words:
            for a in attribute_words:
                if t in self.embeddings and a in self.embeddings:
                    scores.append(self.cosine_sim(
                        self.embeddings[t], self.embeddings[a]))
        return sum(scores) / len(scores) if scores else 0

    def report(self):
        """报告偏见。"""
        male_medical = self.measure_bias(["男性"], ["医生", "工程师"])
        female_medical = self.measure_bias(["女性"], ["医生", "工程师"])
        diff = male_medical - female_medical
        print(f"  男性-医疗关联: {male_medical:.3f}")
        print(f"  女性-医疗关联: {female_medical:.3f}")
        print(f"  差距: {diff:.3f} ({'偏见' if abs(diff) > 0.1 else '公平'})")


probe = EmbeddingBiasProbe()
print("=== WEAT 风格偏见测量 ===")
probe.report()
```

完整代码见 `code/main.py`。

---

## 4. 工程最佳实践

### 4.1 代表伤害和分配伤害需要分别测量

模型可能"代表上无偏见"（产生多样化描绘）但"分配上有偏见"（做出不平等推荐）。评估需要同时测量两者。

### 4.2 交叉性评估是必须的

单轴评估漏掉了只在身份交叉点上触发的偏见。对"性别"的评估漏掉了黑人女性的特定偏见。

### 4.3 元批评提醒：不要只关注二元性别

10 年文献回顾发现该领域不成比例地关注二元性别偏见。残疾、宗教、移民身份等轴线需要更多关注。

---

## 5. 面试考点

### Q1：代表伤害和分配伤害的区别是什么？（难度：⭐⭐）

**参考答案：**
代表伤害：刻板印象、抹除、贬损描绘——模型输出的内容本身有问题。分配伤害：不平等的物质结果——模型做出的决策导致不平等的结果。一个模型可以代表上无偏见（多样化描绘）但分配上有偏见（不平等推荐）。评估需要同时测量两者。

### Q2：为什么交叉性评估对单轴评估不够？（难度：⭐⭐⭐）

**参考答案：**
An et al. 2025 发现 GPT-4o 在简历评分中对黑人女性的惩罚比黑人男性和白人女性分别更大。对"性别"的单轴评估只看到性别差异；对"种族"的单轴评估只看到种族差异。两者都漏掉了"黑人+女性"这个特定交叉身份的额外惩罚。WinoIdentity 通过基于不确定性的公平性评估来捕获这种交叉效应。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 代表伤害 | "刻板印象/抹除" | 对某个群体的有偏见描绘 |
| 分配伤害 | "不平等决策" | 对某个群体的不平等物质结果 |
| 交叉性 | "组合身份效应" | 在多个身份轴交叉处出现的偏见 |
| WEAT | "嵌入测试" | 词嵌入关联测试；基于共现的偏见探针 |
| 性别神经元 | "MLP 偏见神经元" | 激活与性别特定行为相关的特定神经元 |

---

## 📚 小结

LLM 偏见分两类：代表伤害（刻板印象、抹除）和分配伤害（不平等结果）。三类评估指标：基于嵌入（WEAT）、基于概率、基于生成文本。交叉性评估是必须的——单轴评估漏掉了在身份交叉点上触发的偏见。机制可解释性方法（性别神经元、SAE、UniBias）开启了偏见的机制干预。元批评提醒：不要只关注二元性别。

---

## ✏️ 练习

1. 运行 `code/main.py`。报告去偏步骤前后的 WEAT 风格偏见分数。解释为什么指标不会降到零。
2. 设计一个（性别、种族）×（职业、家庭）的交叉性测试。报告跨轴偏见分数。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 偏见探针 | `code/main.py` | WEAT 风格嵌入偏见测量 |
| 偏见评估 | `outputs/skill-bias-eval.md` | 模型卡偏见评估审计 |

---

## 📖 参考资料

1. [论文] Gallegos et al. — Bias and Fairness in LLMs: A Survey. Computational Linguistics 2024
2. [论文] An et al. — Intersectional resume-evaluation bias. PNAS Nexus, March 2025
3. [论文] WinoIdentity — uncertainty-based intersectional fairness. COLM 2025
4. [论文] UniBias — attention-head manipulation. ACL 2024
