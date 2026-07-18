# LLM 功能 A/B 测试——GrowthBook、Statsig 和"感觉"问题

> 传统 A/B 测试不是为非确定性 LLM 构建的。关键区别：评估回答"模型能完成任务吗？"A/B 测试回答"用户在意吗？"两者都需要。2026 年要测试什么：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）——顺序测试、CUPED、一体化。**GrowthBook**——开源（MIT）、仓库原生、贝叶斯+频率学派+顺序引擎、CUPED、SRM 检查。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 13（可观测性）、阶段 17 · 20（渐进式部署）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分评估（"模型能完成任务吗"）和 A/B 测试（"用户在意吗"）
- [ ] 列举三个可测试轴（提示词、模型、参数）并为每个选择指标
- [ ] 解释 CUPED、顺序测试和 Benjamini-Hochberg 多重比较校正
- [ ] 根据仓库 SQL 姿态和企业收购立场选择 Statsig 或 GrowthBook

---

## 1. 问题

你手工调优了一个系统提示词。感觉更好。你发布了。转化率变化在噪声范围内。你怪指标。或者你发布了新模型，转化率没动——是模型退化了还是变化太小检测不到？你不知道，因为你没有做 A/B 就发布了。

评估回答模型能否在标注集上完成任务。它们不回答用户是否偏好输出。只有受控的在线实验能回答那个问题——前提是实验有足够的统计效力、控制了非确定性、并修正了多重比较。

---

## 2. 概念

### 2.1 评估 vs A/B 测试

**评估——** 离线、标注集、裁判（规则或 LLM-as-judge 或人类）。回答："在固定分布上，输出正确/有帮助/安全吗？"

**A/B 测试——** 在线、真实用户、随机化。回答："新变体是否移动了用户层面的重要指标？"

两者都需要。评估在暴露前捕获回归；A/B 在之后确认产品影响。

### 2.2 测试什么

1. **提示词工程** — 措辞、系统提示词结构、示例。指标：任务成功率、用户留存、每请求成本
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + P99 延迟。多目标
3. **生成参数** — temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）

### 2.3 CUPED——方差缩减

使用实验前数据的受控实验。在比较实验期数据之前，回归掉实验前方差。典型方差缩减：30-70%。有效样本量免费增加。

### 2.4 顺序测试

经典 A/B 假设固定样本量。顺序测试（"偷看决定"）在重复观察下控制假阳性率。始终有效的顺序过程（mSPRT、Howard 置信序列）允许在明确胜出时提前停止。

### 2.5 多重比较校正

在 95% 置信度下运行 20 个 A/B 测试会产生一个假阳性。Bonferroni 校正收紧每次测试的 α；Benjamini-Hochberg 控制假发现率。GrowthBook 两者都实现了。

### 2.6 非确定性使统计效力复杂化

相同提示词产生不同的输出。传统效力计算假设 IID 观测。LLM 非确定性下，有效样本量低于名义值。将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 2.7 Statsig vs GrowthBook

**Statsig：** 被 OpenAI 以 $1.1B 收购（2025 年 9 月）。托管 SaaS。顺序测试、CUPED、留存人群。一体化：Feature Flags + 实验 + 可观测性。

**GrowthBook：** 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。多引擎：贝叶斯、频率学派、顺序。CUPED、SRM、Bonferroni、BH 校正。自托管或托管云。

根据仓库 SQL 姿态和"被 OpenAI 收购"对你的组织是否重要来选择。

---

## 3. 从零实现

### 第 1 步：顺序测试模拟

```python
import random


def sequential_test(p_true=0.05, p_control=0.04, n_trials=200, threshold=1.96):
    """简化版顺序测试——累积统计量在边界时提前停止。"""
    successes = 0
    for i in range(1, n_trials + 1):
        if random.random() < p_true:
            successes += 1

        # 简化的 z-score
        p_hat = successes / i
        se = (p_hat * (1 - p_hat) / i) ** 0.5
        if se > 0:
            z = (p_hat - p_control) / se
        else:
            z = 0

        if z > threshold:
            return {"decided": True, "n": i, "result": "treatment_wins",
                    "lift": (p_true - p_control) / p_control}

    return {"decided": False, "n": n_trials, "result": "inconclusive"}


if __name__ == "__main__":
    random.seed(42)
    for p_true in [0.04, 0.045, 0.05, 0.055, 0.06]:
        r = sequential_test(p_true=p_true)
        print(f"p_true={p_true:.3f}  n={r['n']:>4d}  {r['result']:>20s}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 A/B 测试平台对照

| 平台 | 许可 | 特性 | 最佳场景 |
|---|---|---|---|
| Statsig | 商业（OpenAI 收购） | 顺序测试、CUPED、一体化 | 想要捆绑产品 |
| GrowthBook | MIT 开源 | 仓库原生、多引擎、CUPED | 仓库 SQL 团队 |
| 内部方案 | 自建 | 完全控制 | 大型 ML 团队 |

---

## 5. 工程最佳实践

### 5.1 评估和 A/B 都需要

评估在暴露前捕获回归。A/B 在暴露后确认产品影响。两者缺一不可。

### 5.2 非确定性要求更大的样本量

LLM 非确定性使有效样本量低于名义值。将所需样本量乘以 1.3-1.5 倍。

### 5.3 中文场景特别建议

- **中文 A/B 测试的特殊性。** 中文用户的反馈行为可能不同——中文用户更少点击"踩"按钮，但更可能提交客服工单。指标设计需要适配中文用户行为
- **GrowthBook 在国内可用。** GrowthBook 是开源的，可以自托管。直接连接国内的数据仓库（阿里云 MaxCompute、华为云 DWS）
- **国内 LLM 的 A/B 测试成本。** 国内 LLM API 的价格差异大——A/B 测试时需要同时运行两套 API，成本翻倍。确保预算覆盖测试期的额外成本

---

## 6. 常见错误

### 错误 1："感觉更好"就发布

**现象：** 团队调优了提示词，觉得更好就发布了。一周后转化率下降 1.2%，但没人注意到。

**原因：** 没有做 A/B 测试——"感觉"不是统计证据。

**修复：** 永远做 A/B 测试。即使变化看起来很小，也需要统计显著性来确认。

### 错误 2：不修正多重比较

**现象：** 运行 20 个 A/B 测试，其中一个声称"统计显著"。但实际上是因为多次比较导致的假阳性。

**原因：** 20 个测试在 95% 置信度下会产生一个假阳性。没有用 Bonferroni 或 Benjamini-Hochberg 校正。

**修复：** 使用 GrowthBook 的内置校正（Bonferroni 或 BH FDR）。

---

## 7. 面试考点

### Q1：评估和 A/B 测试的区别是什么？为什么两者都需要？（难度：⭐⭐）

**参考答案：**
评估是离线的——在标注数据集上运行候选模型/提示词，回答"输出正确/有帮助吗？"。A/B 测试是在线的——在真实用户上随机分配，回答"用户是否在意这个变化？"两者都需要：评估在暴露前捕获回归（避免发布退化版本），A/B 在暴露后确认产品影响（避免"感觉更好"但实际上没有效果的发布）。只做评估不 A/B = 不知道用户是否真的在意。只 A/B 不评估 = 可能已经发布了退化版本。

### Q2：CUPED 如何在 A/B 测试中减少方差？（难度：⭐⭐⭐）

**参考答案：**
CUPED（Controlled-experiments Using Pre-Experiment Data）利用实验前的数据来减少方差。原理：将实验前用户的指标（如历史转化率）作为协变量，在比较实验组和对照组之前回归掉这部分方差。效果：方差缩减 30-70%，有效样本量免费增加。对于 LLM A/B 测试特别有价值——因为 LLM 非确定性增加了额外方差，CUPED 可以部分抵消这个效应。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 评估 | "离线测试" | 标注集上的模型能力评估 |
| A/B 测试 | "实验" | 在线用户上的随机化对比 |
| CUPED | "方差缩减" | 使用实验前数据的回归来减少方差 |
| 顺序测试 | "可偷看的测试" | 始终有效的过程，允许提前停止 |
| 多重比较 | "族错误" | 多次测试膨胀假阳性 |
| Bonferroni | "严格校正" | 除以测试次数 |
| SRM | "分配错误" | 样本比例不匹配；分配哈希有 bug |
| Statsig | "OpenAI 所有" | 商业一体化平台，2025 年收购 |
| GrowthBook | "开源那个" | MIT 仓库原生平台 |

---

## 📚 小结

评估回答"模型能完成任务吗？"A/B 测试回答"用户在意吗？"两者都需要。三个可测试轴：提示词（措辞）、模型（选择）、参数（temperature/top-p）。非确定性要求更大样本量（1.3-1.5 倍）。CUPED 减少方差 30-70%。Statsig（OpenAI 收购）和 GrowthBook（MIT 开源）是两个主要选择——根据仓库 SQL 姿态和企业收购立场选择。**"感觉更好"不是发布理由——A/B 测试才是。**

---

## ✏️ 练习

1. 运行 `code/main.py`。预期 5% 提升、基线 3% 转化率时，达到 80% 统计效力需要多少样本量？
2. 为一个医疗受监管的本地客户选择 Statsig 还是 GrowthBook。
3. 设计一个 A/B 测试：GPT-4 vs GPT-3.5 在每解决工单成本上的表现。主要指标、护栏指标、次要指标分别是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 顺序测试模拟器 | `code/main.py` | 固定边界 vs 顺序边界的对比 |
| A/B 测试方案 | `outputs/skill-ab-plan.md` | 根据功能变更选择平台、门控、样本量 |

---

## 📖 参考资料

1. [博客] GrowthBook — How to A/B Test AI. https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/
2. [博客] Statsig — Beyond Prompts: Data-Driven LLM Optimization. https://www.statsig.com/blog/llm-optimization-online-experimentation
3. [论文] Deng et al. — CUPED. https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
4. [论文] Howard — Confidence Sequences. https://arxiv.org/abs/1810.08240
