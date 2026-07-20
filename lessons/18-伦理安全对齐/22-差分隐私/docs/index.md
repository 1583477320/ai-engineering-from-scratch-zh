# LLM 的差分隐私

> DP-SGD 仍然是标准——注入噪声的梯度更新提供形式化的 (ε, δ) 保证。在计算、内存和效用上的开销是实质性的；参数高效 DP 微调（LoRA + DP-SGD）是 2025 年的通用配置。两组证据紧张对立：基于金丝雀的成员推断（Duan et al., 2024）报告对语言模型的成功有限；训练数据提取（Carlini et al., 2021; Nasr et al., 2025）恢复了大量逐字记忆。2025 年 3 月的解决：差距在于测量内容——插入的金丝雀 vs "最可提取"数据。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 01 · 09（信息论）、阶段 10 · 01（大模型训练）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义 (ε, δ)-差分隐私并陈述 DP-SGD 配方
- [ ] 解释 2024-2025 年的紧张对立：金丝雀 MIA vs 训练数据提取给出不同图景
- [ ] 描述 PMixED 以及为什么推理时隐私预测是 DP 训练的替代方案
- [ ] 描述差分隐私反转通过 LLM 反馈的攻击

---

## 1. 问题

LLM 会记忆。Carlini et al. 2021 显示生产语言模型按要求逐字复现训练文本。DP 是形式化防御：训练使得输出对任何单个训练样本的贡献在证明上不敏感。2024-2025 年的证据表明 DP-SGD 是必要的，但已部署的 ε 值可能不匹配威胁模型。

---

## 2. 概念

### 2.1 (ε, δ)-差分隐私

随机算法 M 是 (ε, δ)-DP 的，如果对于任何相差一个样本的两个数据集和任何事件 S：
`P(M(D) in S) ≤ e^ε × P(M(D') in S) + δ`

### 2.2 DP-SGD

Abadi et al. 2016。标准配方：
1. 采样小批量
2. 计算每样本梯度
3. 每样本梯度裁剪到阈值 C
4. 裁剪后的梯度求和并添加标准差为 σ × C 的高斯噪声
5. 用噪声和更新参数

### 2.3 2024-2025 年的紧张对立

- **金丝雀 MIA（Duan et al. 2024）：** 插入唯一金丝雀，测量成员推断攻击是否能识别它们。报告有限成功。暗示 MIA 很难
- **训练数据提取（Carlini 2021, Nasr et al. 2025）：** 用前缀提示模型，测量是否恢复训练中的逐字文本。报告大量记忆。暗示 MIA 在相关意义上很容易

2025 年 3 月解决：两者测量不同事物。MIA 问"样本 e 在 D 中吗？"（金丝雀）。提取问"D 中我能恢复什么？""最可提取"样本才是隐私关心的。

### 2.4 DP 训练的替代方案

- **PMixED（arXiv:2403.15638）。** 推理时隐私预测。对下一个词元分布的专家混合——每个专家看到训练数据的一个分片；聚合添加噪声以实现 DP
- **DP 合成数据生成（Google Research 2024）。** 用 DP-SGD 做 LoRA 微调，采样合成数据，在合成数据上训练下游分类器

两者绕过了完全 DP 训练的效用成本，代价是不同的威胁模型。

### 2.5 通过 LLM 反馈的差分隐私反转

新兴 2025 攻击。使用 DP 训练模型的置信度分数作为 oracle 重新识别个体。即使输出不泄露，置信度分布也可以。防御：不暴露置信度，或在暴露前截断/量化。

---

## 3. 从零实现

```python
import random
import math


def dp_sgd_step(gradient, clip_norm=1.0, noise_multiplier=1.0):
    """单步 DP-SGD：裁剪 + 加噪。"""
    # 梯度裁剪
    grad_norm = math.sqrt(sum(g**2 for g in gradient))
    if grad_norm > clip_norm:
        gradient = [g * clip_norm / grad_norm for g in gradient]

    # 添加高斯噪声
    noisy_grad = [g + random.gauss(0, noise_multiplier * clip_norm)
                  for g in gradient]

    return noisy_grad


def compute_epsilon(steps, noise_multiplier, delta=1e-5):
    """简化版 (ε, δ) 计算。"""
    eps = steps * (1.6 * noise_multiplier) ** 2
    return eps, delta


# 演示
gradient = [0.5, -0.3, 0.8, -0.2]
noisy = dp_sgd_step(gradient, clip_norm=1.0, noise_multiplier=1.0)
eps, delta = compute_epsilon(steps=100, noise_multiplier=1.0)

print(f"原始梯度: {[f'{g:.2f}' for g in gradient]}")
print(f"DP-SGD 后: {[f'{g:.2f}' for g in noisy]}")
print(f"(ε, δ) = ({eps:.2f}, {delta:.1e})")
print(f"噪声引入了约 {abs(sum(noisy) - sum(gradient))/abs(sum(gradient)):.1%} 的偏差")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 DP 工具

| 工具 | 类型 | 特点 |
|---|---|---|
| Opacus | DP-SGD 训练 | Facebook 开源，PyTorch 集成 |
| Google DP | DP 库 | Google 开源，全面 |
| TF Privacy | DP 训练 | TensorFlow 集成 |
| IBM Diffprivlib | DP 工具箱 | 基础 DP 原语 |

---

## 5. 工程最佳实践

### 5.1 DP-SGD 的 ε 值没有"安全"默认

不同的威胁模型、数据敏感度、效用目标需要不同的 ε。不要假设 ε=1 是安全的。

### 5.2 LoRA + DP-SGD 是 2025 年配置

完全 DP-SGD 对前沿模型是禁止的。LoRA 限制梯度更新到小适配器——DP 保证适用于适配器，基础模型保持固定。

### 5.3 置信度泄露是新兴攻击

DP 训练模型的置信度分数可以作为 oracle 重新识别个体。防御：不暴露置信度，或在暴露前截断/量化。

---

## 6. 常见错误

### 错误 1：假设 ε 值小就安全

**现象：** 使用 ε=1 进行 DP-SGD 训练，认为这是"安全"的。

**原因：** ε 的安全阈值取决于威胁模型、数据敏感度和效用目标——没有普遍的"安全"默认值。

**修复：** 根据具体场景评估所需的 ε，而不是假设一个固定值。

### 错误 2：忽略置信度泄露

**现象：** DP 训练后声称隐私保护，但模型的置信度分数可以被用来重新识别个体。

**修复：** 在部署时截断或量化置信度分数——这是 DP 训练之外的额外要求。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| DP | "(ε, δ)-差分隐私" | 形式化隐私：输出分布对相邻数据集变化足够接近 |
| DP-SGD | "注入噪声的 SGD" | 梯度裁剪 + 高斯噪声添加；标准 DP 训练 |
| LoRA + DP-SGD | "高效私有微调" | 低秩适配器上的 DP-SGD；2025 年标准配置 |
| MIA | "成员推断" | 判断样本是否在训练数据中的攻击 |
| 金丝雀 | "插入的水印样本" | 用于测量 DP 泄露的唯一训练样本 |
| PMixED | "私有推理混合" | 推理时 DP——通过专家混合在下一个词元分布上 |
| DP 反转 | "置信度泄露攻击" | 使用模型置信度作为重新识别的 oracle |

---

## 📚 小结

DP-SGD 仍然是标准——注入噪声的梯度更新提供形式化的 (ε, δ) 保证。2024-2025 年的紧张对立（金丝雀 MIA vs 训练数据提取）在 2025 年 3 月得到解决——两者测量不同事物。LoRA + DP-SGD 是 2025 年配置。DP 的替代方案（PMixED、合成数据）绕过了完全 DP 训练的效用成本。置信度泄露是新兴攻击——DP 训练之外需要额外防御。

---

## ✏️ 练习

1. 运行 `code/main.py`。扫描 σ 在 {0.5, 1.0, 2.0} 时的 (ε, δ)-准确性权衡。找出效用崩溃的点。
2. 实现金丝雀插入和 log-loss 测试。测量 DP-SGD 在 σ=1.0 前后的检测率。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| DP-SGD 模拟 | `code/main.py` | 噪声注入和 ε-δ 计算 |
| DP 审计 | `outputs/skill-dp-audit.md` | 差分隐私声明审计 |

---

## 📖 参考资料

1. [论文] Abadi et al. — DP-SGD. arXiv:1607.00133
2. [论文] Carlini et al. — Extracting Training Data. arXiv:2012.07805
3. [论文] Duan et al. — Canary MIA on LLMs. arXiv:2402.07841
4. [论文] Kowalczyk et al. — Auditing DP for LLMs. arXiv:2503.06808
5. [论文] PMixED. arXiv:2403.15638
