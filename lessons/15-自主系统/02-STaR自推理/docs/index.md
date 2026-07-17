# STaR、V-STaR、Quiet-STaR——自我教学推理

> 最小的自我改进循环就在推理链内部。模型生成思维链，保留那些得到正确答案的，然后在这些上面微调。这就是 STaR。V-STaR 添加了一个验证器使推理时选择更好。Quiet-STaR 将推理链推到每个词元位置。三者都有效。三者都不是魔法——循环会保留任何恰好得到正确答案的捷径。

**类型：** 概念课
**语言：** Python（标准库，自举循环模拟器）
**前置知识：** 阶段 13 · 01-03（推理和 CoT）、阶段 15 · 01（长期智能体框架）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 STaR 的核心循环：采样 → 保留正确答案的推理链 → 微调 → 重复
- [ ] 理解理性化（rationalization）如何解决模型无法独立答对的问题
- [ ] 区分 STaR、V-STaR 和 Quiet-STaR 的训练信号和推理成本
- [ ] 识别三种方法共享的安全问题：答案条件梯度信号会强化捷径推理
- [ ] 将 STaR 与 2026 年前沿方法（DeepSeek-R1、过程奖励模型）联系起来

---

## 1. 问题

教模型推理的直接方法是收集人类写的推理链。但这昂贵、缓慢，且受限于人类愿意写的高质量思维链数量。

STaR（Self-Taught Reasoner，Zelikman 等人，2022）提出：**让模型自己写推理链，然后用已知答案对它们评分**。循环是：

1. 采样一个推理链加上答案
2. 如果最终答案正确，保留推理链
3. 在保留的推理链上微调
4. 重复

它有效。GSM8K 和 CommonsenseQA 都在没有新的人类标注的情况下得到了改进。但循环有一个内建偏差：**任何得到正确答案的推理链都会被保留，无论推理本身是否合理。** V-STaR（Hosseini 等人，2024）用一个学习到的验证器修补了这个问题；Quiet-STaR（Zelikman 等人，2024）将这个想法泛化到每个词元位置的内部推理链。

---

## 2. 概念

### 2.1 STaR：在有效的方法上自举

从一个具有某些弱推理能力的基础模型开始。对每个训练问题，采样一个推理链加答案。如果答案匹配标签，保留（问题、推理链、答案）三元组。在保留集上微调模型。重复。

一个关键变体：**理性化（Rationalization）**。如果模型永远答不对某个问题，循环无法在它上面学习。STaR 对这些问题注入正确答案作为提示，重新提示模型生成导向该答案的推理链。理性化的推理链被加入训练集。

```
STaR 循环：
采样推理链 → 答案正确？→ 是 → 保留 → 微调
                          → 否 → 是理性化吗？→ 是 → 保留 → 微调
                                           → 否 → 丢弃
```

原始论文结果：GPT-J 基础模型在 GSM8K 上从 5.8% 提升到 10.7%（约 5 个百分点绝对提升）。在 CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，可比肩在人类标注推理链上微调的 GPT-3 175B（约 73%）——后者模型大 30 倍。

### 2.2 V-STaR：用 DPO 训练验证器

STaR 丢弃了错误的推理链。Hosseini 等人（2024）观察到这些也是数据：每对（推理链，"是否正确"）都可以训练一个验证器。他们使用直接偏好优化（DPO）在正确和错误的解上构建排序器。推理时采样 N 个推理链，选验证器的最高分。

报告的提升：在 GSM8K 和 MATH 上比先前的自我改进基线高 +4 到 +17 个百分点，大部分收益来自推理时选择而非额外的生成器微调。

### 2.3 Quiet-STaR：每个词元位置的内部推理链

Zelikman 等人（2024）问：如果模型在每个词元位置学习生成一个短的内部推理链，而不只是在问题和答案之间呢？Quiet-STaR 训练模型在每个预测词元前发出一个隐藏的"思考"，然后通过学习到的权重将思考感知的预测与基线预测混合。

结果：Mistral 7B 在 GSM8K 上从 5.9% 提升到 10.9%，在 CommonsenseQA 上从 36.3% 提升到 47.2%，无需任务特定微调。模型学会了"什么时候该思考"——困难词元获得更长的内部推理链；简单的几乎没有。

### 2.4 为什么三者共享安全问题

三种方法都用最终答案作为梯度信号。一个通过错误推理得到正确答案的推理链——利用了捷径、猜测或不泛化的模式——会被正向强化。在分布内问题上捷径有效。在分布外问题上它会静默崩溃。

V-STaR 的验证器通过学习排序推理链来缓解，但验证器是在同一个标签集上训练的。它可以学会偏好格式良好但错误的推理，而不是诚实的不确定。更安全的设计是将 STaR 式数据与（a）过程监督奖励模型（奖励中间步骤而非仅答案）和（b）保留的 OOD 评估（打破简单捷径）结合。

### 2.5 三种方法对比

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知失败模式 |
|------|---------|---------|---------|------------|
| STaR | 保留正确答案的推理链 | 1x | 丢弃所有错误推理链 | 捷径推理链 |
| STaR + 理性化 | 上述 + 正确答案提示重试 | 1x | 较少 | 理性化推理链可能不合理 |
| V-STaR | STaR + 双类 DPO 验证器 | Nx (best-of-N) | 最少 | 验证器可能强化自信的错误 |
| Quiet-STaR | 每词元推理链 + 混合权重 | 1.5-3x | 最少 | 仍然是答案条件梯度 |

### 2.6 在 2026 年技术栈中的位置

STaR 是老的。但这个模式在 2025-2026 年无处不在：

| 2025-2026 方法 | 与 STaR 的关系 |
|---------------|---------------|
| DeepSeek-R1、Kimi-k1.5、o1 | STaR 的答案条件梯度信号，规模化 |
| 过程奖励模型（Lightman 等人，2023） | 过程监督的替代方案——奖励每步而非仅答案 |
| AlphaEvolve（第 3 课） | STaR 用于代码，程序评估器替代标签 |
| Darwin Godel Machine（第 4 课） | STaR 用于智能体脚手架本身 |

理解 STaR 让所有这些都变得清晰。它是最小可行的自我改进循环。

---

## 3. 从零实现

### 第 1 步：定义模型和采样

```python
from dataclasses import dataclass
import random

@dataclass
class Trace:
    strategy: str       # "sound" | "shortcut" | "random"
    answer_correct: bool
    rationale_sound: bool

@dataclass
class Model:
    prob_sound: float
    prob_shortcut: float

    def sample(self, on_ood: bool) -> Trace:
        r = random.random()
        if r < self.prob_sound:
            return Trace("sound", True, True)
        elif r < self.prob_sound + self.prob_shortcut:
            # 捷径在分布内 40% 正确，OOD 5%
            ok = random.random() < (0.05 if on_ood else 0.40)
            return Trace("shortcut", ok, False)
        else:
            ok = random.random() < 0.10
            return Trace("random", ok, False)
```

### 第 2 步：实现 STaR 循环

```python
def star_round(model: Model, n_samples: int = 1000) -> Model:
    """一轮 STaR：保留正确答案的推理链，重新训练。"""
    kept = []
    for _ in range(n_samples):
        t = model.sample(on_ood=False)
        if t.answer_correct:
            kept.append(t)

    if not kept:
        return model

    sound_kept = sum(1 for k in kept if k.strategy == "sound")
    shortcut_kept = sum(1 for k in kept if k.strategy == "shortcut")
    total = len(kept)

    # 按强化比例更新模型，与旧先验混合防止坍缩
    alpha = 0.6
    new_sound = alpha * (sound_kept / total) + (1 - alpha) * model.prob_sound
    new_short = alpha * (shortcut_kept / total) + (1 - alpha) * model.prob_shortcut
    return Model(new_sound, new_short)
```

### 第 3 步：实现 V-STaR 推理选择

```python
def vstar_infer(model, samples_per_problem, n_problems, on_ood):
    """V-STaR 风格 best-of-N：选验证器认为最好的推理链。"""
    correct = 0
    for _ in range(n_problems):
        traces = [model.sample(on_ood) for _ in range(samples_per_problem)]
        best = None
        best_score = -1.0
        for t in traces:
            # 验证器偏好合理推理，但可能被自信的错误欺骗
            score = 0.9 if t.rationale_sound else (0.55 if t.answer_correct else 0.3)
            score += random.random() * 0.1  # 噪声
            if score > best_score:
                best_score = score
                best = t
        if best and best.answer_correct:
            correct += 1
    return correct / n_problems
```

### 第 4 步：运行演示

```python
def main():
    random.seed(42)

    # 场景 A：无捷径先验
    models_a = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.0))
    print("场景 A：无捷径先验")
    for i, m in enumerate(models_a):
        id_acc, _ = evaluate(m, 500, on_ood=False)
        ood_acc, _ = evaluate(m, 500, on_ood=True)
        print(f"  轮 {i}: ID={id_acc:.1%} OOD={ood_acc:.1%}")

    # 场景 B：有捷径先验
    models_b = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.40))
    print("\n场景 B：有捷径先验")
    for i, m in enumerate(models_b):
        id_acc, _ = evaluate(m, 500, on_ood=False)
        ood_acc, _ = evaluate(m, 500, on_ood=True)
        print(f"  轮 {i}: ID={id_acc:.1%} OOD={ood_acc:.1%}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 STaR 模式在 2026 年的体现

| 工具/方法 | STaR 等价物 | 说明 |
|----------|-----------|------|
| DeepSeek-R1 | 大规模 STaR | 在可验证数学问题上 RL，等价于 STaR 的答案条件信号 |
| 过程奖励模型 | 过程监督替代 | 奖励每步而非仅答案——打破捷径推理链 |
| AlphaEvolve | 代码版 STaR | 程序评估器替代标签 |
| Self-Refine（阶段 14 · 05） | 单轮 STaR | 同一个智能体的自审查循环 |

### 4.2 关键洞察

STaR 是最小可行的自我改进循环。理解它让所有 2025-2026 年的"自我改进"方法都变得清晰——它们都是 STaR 的变体，区别在于评估器的严格程度。

---

## 5. 工程最佳实践

### 5.1 STaR 循环设计原则

| 原则 | 说明 |
|------|------|
| 答案条件梯度有内建偏差 | 任何恰好得到正确答案的推理链都会被保留 |
| 验证器缓解但不消除 | V-STaR 的验证器在相同标签上训练，可以强化自信的错误 |
| OOD 评估是必须的 | 捷径在分布内有效，OOD 静默崩溃 |
| 过程监督比结果监督更安全 | 奖励中间步骤而非仅最终答案 |

### 5.2 中文场景特别建议

- **STaR 在中文数学推理上同样有效**——DeepSeek-R1 已经证明了这一点
- **理性化的风险在中文中更明显**——中文的同音异义词和多义词可能导致不合理的推理链恰好得到正确答案
- **评估集必须覆盖中文特有的歧义**——英文评估集可能无法检测中文中的捷径推理

### 5.3 踩坑经验

- **不设 OOD 评估**——训练集上准确率 95%，OOD 上 40%。不知道，因为只看了训练集。**修复：** 始终保留 20% 数据作为 OOD 评估
- **验证器在相同标签上训练**——验证器学会了"这个推理链格式良好"而不是"这个推理链正确"。**修复：** 验证器用独立数据集训练
- **理性化推理链不合理**——注入正确答案后模型编造了一个看似合理但实际荒谬的推理链。**修复：** 对理性化推理链做人工抽查

---

## 6. 常见错误

### 错误 1：不保留 OOD 评估

**现象：** 训练集上 STaR 循环准确率从 5.8% 涨到 95%。团队庆祝。但部署后发现模型在新问题上表现和训练前一样差。

**原因：** 捷径推理链在训练分布上有效（40% 正确率），在 OOD 上几乎为零。没有 OOD 评估就看不到这个差距。

**修复：** 始终保留 20% 数据作为 OOD 评估。每轮 STaR 循环同时报告 ID 和 OOD 准确率。

### 错误 2：验证器与生成器在同一数据上训练

**现象：** V-STaR 的验证器在 ID 上表现很好，但 OOD 上偏好格式良好但错误的推理链。因为验证器学会了"格式好的推理链 = 正确"而不是"推理正确 = 正确"。

**原因：** 验证器和生成器在同一个标签集上训练，验证器的偏差与生成器的偏差相关。

**修复：** 验证器用独立的、多样化的问题集训练。加入过程监督信号——奖励每步推理而不仅仅是最终答案。

### 错误 3：将 STaR 当作通用解决方案

**现象：** 对所有任务都用 STaR 循环。有些任务不适合——比如创意写作、开放式对话、主观评估。

**原因：** STaR 需要"正确答案"作为信号。只有可验证的任务才适合。

**修复：** STaR 适用于有客观正确答案的任务：数学、代码、事实问答。不适用于创意、主观或开放式任务。

---

## 7. 面试考点

### Q1：STaR 的核心循环是什么？为什么它有效？（难度：⭐）

**参考答案：**
STaR 的循环：采样推理链 → 保留得到正确答案的推理链 → 在保留集上微调 → 重复。

有效的原因：模型自己生成了训练数据。不需要人类写推理链。通过自举，模型从弱推理能力开始，逐渐提升。GPT-J 6B 用 STaR 达到了 GPT-3 175B 在人类标注数据上的水平——模型大小差 30 倍。

### Q2：理性化（Rationalization）解决了什么问题？（难度：⭐⭐）

**参考答案：**
STaR 的一个内建问题：如果模型永远答不对某个问题，循环无法在它上面学习——保留集里没有这个问题的数据。

理性化的解决方案：对模型答不对的问题，注入正确答案作为提示，让模型重新生成推理链。这样即使模型不能独立答对，也能生成（答案 → 推理链）的数据用于训练。

风险：理性化推理链可能不合理——模型编造了一个恰好导向正确答案但实际荒谬的推理链。

### Q3：V-STaR 如何改进 STaR？验证器的局限是什么？（难度：⭐⭐）

**参考答案：**
V-STaR 用 DPO 在正确和错误的推理链上训练一个验证器。推理时采样 N 个推理链，选验证器最高分的。

改进：在 GSM8K 和 MATH 上比先前基线高 +4 到 +17 个百分点，大部分收益来自推理时选择。

局限：验证器在同一个标签集上训练。它可能学会偏好"格式良好"的推理链而非"推理正确"的推理链。一个自信的错误推理链可能比诚实的不确定得到更高分。

### Q4：STaR 和 DeepSeek-R1 的关系是什么？（难度：⭐⭐⭐）

**参考答案：**
DeepSeek-R1 在可验证的数学问题上使用强化学习。这本质上是 STaR 的答案条件梯度信号的规模化版本。

STaR 循环：采样 → 用正确答案过滤 → 微调。DeepSeek-R1 循环：采样 → 用验证器评分 → 强化学习更新。核心区别：STaR 用微调（模仿保留的推理链），DeepSeek-R1 用 RL（奖励信号更新策略）。

过程奖励模型（Lightman 等人，2023）是 STaR 的过程监督替代——奖励每步推理而不仅仅是最终答案。这是打破捷径推理链的关键。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| STaR | "自教学推理器" | 在模型自己生成的、得到正确答案的推理链上微调；重复 |
| 理性化 (Rationalization) | "提示重试" | 对模型答不对的问题注入正确答案并重新提示生成推理链 |
| V-STaR | "验证器 STaR" | 用 DPO 在正确和错误推理链上训练验证器，推理时选择 |
| Quiet-STaR | "每词元推理链" | 在每个词元位置生成隐藏思考；与基线预测混合 |
| 答案条件梯度 | "基于结果的信号" | 训练循环奖励最终答案而非推理步骤 |
| 过程奖励模型 | "步骤级验证器" | 在每步正确性上训练的奖励模型——与 STaR 的结果信号对比 |
| 捷径推理链 | "正确答案，错误推理" | 通过不泛化的模式得到标签的推理链；STaR 会保留这些 |

---

## 📚 小结

STaR 是最小可行的自我改进循环：采样推理链，保留得到正确答案的，微调，重复。它在 GSM8K 和 CommonsenseQA 上有效，但内建偏差——捷径推理链会被保留。V-STaR 用验证器缓解，Quiet-STaR 泛化到每词元。理解 STaR 让 2025-2026 年的所有"自我改进"方法都变得清晰：DeepSeek-R1 是 STaR 的规模化，过程奖励模型是 STaR 的过程监督替代，AlphaEvolve 是代码版 STaR。

下一课：AlphaEvolve——进化编码智能体。

---

## ✏️ 练习

1. **【实验】** 运行模拟器。设捷径频率为 0，然后设为 0.4。最终准确率在两种运行之间差多少？即使两者在训练分布上都超过 90%。

2. **【实现】** 在模拟器中添加 OOD 测试。从不同分布中抽取问题，评估自举模型在 ID 和 OOD 集上的表现。量化差距。

3. **【阅读】** 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。用三句话分别解释"end-of-thought"标记和混合权重头。

4. **【思考】** 对比 STaR 的"保留正确答案"过滤和过程监督替代方案（奖励每步推理）。识别标注成本差异和合理的质量差异。

5. **【设计】** 设计一个能捕获部署模型中捷径推理链的评估方案。不需要完美——只需要能打破 STaR 循环会强化的最简单捷径。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| STaR 循环模拟器 | `code/main.py` | 模拟 STaR、V-STaR 的自举过程，展示捷径强化 |
| 技能提示词 | `outputs/skill-star-loop-reviewer.md` | 审计自教学推理流水线 |

---

## 📖 参考资料

1. [论文] Zelikman et al. (2022). "STaR: Bootstrapping Reasoning With Reasoning". https://arxiv.org/abs/2203.14465 — 原始论文
2. [论文] Hosseini et al. (2024). "V-STaR: Training Verifiers for Self-Taught Reasoners". https://arxiv.org/abs/2402.06457 — DPO 验证器
3. [论文] Zelikman et al. (2024). "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking". https://arxiv.org/abs/2403.09629 — 每词元内部推理链
4. [论文] Lightman et al. (2023). "Let's Verify Step by Step". https://arxiv.org/abs/2305.20050 — 过程奖励模型
5. [论文] DeepSeek-R1. https://arxiv.org/abs/2501.12948 — 可验证任务上的 RL，STaR 规模化到前沿训练

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
