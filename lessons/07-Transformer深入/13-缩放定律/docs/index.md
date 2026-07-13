# 缩放定律

> Chinchilla 说：给模型 20 倍参数，性能提升 2 倍。给模型 20 倍数据，性能提升 2 倍。参数和数据要平衡增长——而不是只要大模型。

**类型：** 概念课
**语言：** Python
**前置知识：** 第 7 阶段 . 05（完整 Transformer）
**预计时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 . 14（Transformer 毕业设计）— 缩放定律指导模型规模选择

---

## 🎯 学习目标

- [ ] 理解 Kaplan 缩放定律——损失与模型大小、数据量、计算量的幂律关系
- [ ] 解释 Chinchilla 的修正——为什么 GPT-3 用 175B 参数 + 300B 词元是次优的
- [ ] 说明 2026 年"过度训练"策略的动机——Llama 系列用更小模型 + 更多数据
- [ ] 使用 NumPy 模拟缩放定律，验证损失与规模的幂律关系
- [ ] 比较不同模型的推理 FLOPs，解释为什么 Llama 3 8B 在成本上完胜 GPT-3

---

## 1. 问题

给定固定计算预算——应该训练一个更大的模型（少轮次），还是训练一个小模型（多轮次）？2020 年之前，没有人能给出有根据的答案。GPT-3 用 175B 参数 + 300B 词元训练——这比必要的参数量多了 4 倍。这些多出来的参数在推理时持续消耗算力，但性能提升微乎其微。

如果你在 2024 年部署一个 GPT-3 规模的模型，用户每调用一次，GPU 要完成 7.17e14 次浮点运算。而 Llama 3 8B 完成同样任务只需 3.28e13 次——快 22 倍，性能相当。理解缩放定律，你就理解了为什么"更大的模型"并不总是"更好的选择"。

---

## 2. 概念

### 2.1 Kaplan 缩放定律（2020）

损失（交叉熵）与三个因素的幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

- **N** = 参数量（模型大小）
- **D** = 数据量（训练词元数）
- **C** = 计算量（FLOPs）

**关键推论：** 模型大小增加 10 倍，损失降低约 20%；数据量增加 10 倍，损失降低约 12%。**参数和数据要平衡增长**。

### 2.2 Chinchilla 的修正（2022）

Kaplan 的实验受限于早期硬件——模型只训练了少量轮次。DeepMind 修正了这一点：

**Chinchilla 结论：** 给定固定计算预算，最优策略是参数量和数据量同比例增长。GPT-3（175B 参数，300B 词元）应该只用约 40B 参数训练 300B 词元——**参数过大多了 4 倍，性能只高了 20%**。

### 2.3 2026 年的"过度训练"

| 模型 | 参数量 | 训练词元 | Chinchilla 比率 |
|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7x（参数过多） |
| Llama 3 | 8B | 15T | 1875x（数据过多） |
| Llama 3 405B | 405B | 15T | 37x（数据过多） |
| Qwen2.5 | 7B | 18T | 2571x（数据过多） |

**2026 年的主流策略是"过度训练"——用更小模型 + 更多数据。** 原因：推理时参数量决定成本（FLOPs = 2 x 参数量 x 序列长度）。8B 参数 x 18T 词元的模型在推理时比 175B x 300B 的模型快 20 倍——但性能差不多。

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 3. 从零实现

### 第 1 步：Kaplan 幂律函数

```python
import numpy as np

def kaplan_loss_by_params(num_params):
    """L(N) = (N_c / N)^alpha，参数增加 10x -> 损失 ~20%。"""
    return (8.8e13 / num_params) ** 0.076

def kaplan_loss_by_data(num_tokens):
    """L(D) = (D_c / D)^alpha，数据增加 10x -> 损失 ~12%。"""
    return (5.4e12 / num_tokens) ** 0.095

# 验证：参数量从 10M 到 1T
params = np.array([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
losses = kaplan_loss_by_params(params)
for n, l in zip(params, losses):
    print(f"参数 {n:.0e} -> 损失 {l:.4f}")
```

### 第 2 步：Chinchilla 最优分配

```python
def chinchilla_optimal(compute_flops):
    """给定 FLOPs，返回最优参数量 N* 和训练词元数 D*。"""
    N = (compute_flops / 120.0) ** 0.5  # C = 120 * N^2
    D = 20.0 * N
    return N, D

# GPT-3 的计算预算
gpt3_N, gpt3_D = 175e9, 300e9
C = 6 * gpt3_N * gpt3_D  # ~3.15e23 FLOPs
N_opt, D_opt = chinchilla_optimal(C)
print(f"最优参数: {N_opt:.2e}, 最优词元: {D_opt:.2e}")
print(f"GPT-3 参数多 {gpt3_N / N_opt:.0f} 倍")
```

### 第 3 步：推理成本对比

```python
def inference_flops(num_params, seq_len=2048):
    """推理 FLOPs = 2 x 参数量 x 序列长度。"""
    return 2.0 * num_params * seq_len

gpt3_flops = inference_flops(175e9)
llama8_flops = inference_flops(8e9)
print(f"GPT-3 推理: {gpt3_flops:.2e} FLOPs")
print(f"Llama 3 8B: {llama8_flops:.2e} FLOPs")
print(f"速度提升: {gpt3_flops / llama8_flops:.0f} 倍")
```

---

## 4. 工业工具

### 4.1 训练 FLOPs 估算工具

```python
# 使用 Chinchilla 公式估算训练成本
def estimate_training_cost(num_params_B, num_tokens_T, gpu_tflops=312, gpu_count=1):
    """估算训练所需 GPU 小时数。"""
    C = 6 * num_params_B * 1e9 * num_tokens_T * 1e12  # 总 FLOPs
    gpu_flops_per_sec = gpu_tflops * 1e12
    total_flops_per_sec = gpu_flops_per_sec * gpu_count
    seconds = C / total_flops_per_sec
    hours = seconds / 3600
    return hours

# Llama 3 8B 训练成本估算
hours = estimate_training_cost(8, 15, gpu_tflops=990, gpu_count=16384)
print(f"Llama 3 8B 训练约需 {hours:.0f} GPU 小时")
```

### 4.2 常用缩放估算框架

| 工具 | 用途 | 适用场景 |
|---|---|---|
| Chinchilla 公式 | 最优 N/D 分配 | 模型规划阶段 |
| 海岸线估算（Scaling Laws for Downstream Tasks） | 下游任务损失 | 评估微调收益 |
| 大语言模型规模计算器（llm-scaling） | 训练成本估算 | 预算规划 |

### 4.3 性能对比

| 实现方式 | 精度 | 速度 | 适用场景 |
|---|---|---|---|
| 本课 NumPy 版 | 近似 | 极快 | 学习理解 |
| Chinchilla 公式 | 拟合精度 | 瞬时 | 模型规划 |
| 小规模实验外推 | 中等 | 分钟级 | 初步验证 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

GPT-3（175B）是 Kaplan 缩放定律的产物——OpenAI 认为"越大越好"。但 Chinchilla 证明这是错的：GPT-3 的参数量比最优值多了 4 倍。这直接导致了 Llama 3 的策略转变——Meta 用 8B 参数 + 15T 词元训练，推理成本仅为 GPT-3 的 1/22。

### 5.2 LLM 时代什么变了？

Chinchilla 最优比率是 $D^*/N^* \approx 20$，但 Llama 3 8B 的实际比率是 $15T/8B = 1875$。这是因为**推理成本主导了部署决策**——训练是一次性成本，推理是持续成本。2026 年的"过度训练"本质上是用一次性训练成本换取持续的推理成本节省。

### 5.3 什么没变？

幂律关系本身没有变。参数量和数据量对损失的贡献仍然是幂律的。变的是优化目标——从"给定训练预算最小化损失"变成"给定总成本（训练 + 推理）最小化损失"。理解原始的 Kaplan/Chinchilla 公式，才能理解为什么现代模型偏离了"最优"。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你同时使用 ChatGPT（GPT-4）和 Claude（基于 Llama 架构优化）时，你可能注意到 Claude 的响应速度更快。部分原因就是缩放策略的差异——更小的模型 + 更多数据 = 推理更快。理解缩放定律，你就理解了为什么"大模型"不等于"好模型"。

---

## 6. 工程最佳实践

### 6.1 模型规模选择

| 场景 | 推荐策略 | 理由 |
|---|---|---|
| 学习/实验 | 1B-3B 参数 | 训练快、成本低、足够验证思路 |
| 生产部署（延迟敏感） | 8B-14B + 大量数据 | 推理快、成本可控 |
| 生产部署（质量优先） | 70B+ + 足够数据 | 更好的推理能力 |

### 6.2 中文场景特别建议

- 中文训练词元的"信息密度"低于英文（一个中文字 ≈ 2-3 个英文词元），因此中文模型需要更多词元才能达到同等损失水平
- 使用 `tokenizer` 计算中文文本的词元数，确保训练时的数据量估算准确
- Qwen2.5 7B 使用 18T 词元训练，其中中文占比约 30%——这是中文过度训练的典型案例

### 6.3 踩坑经验

- 不要直接套用 Chinchilla 公式决定生产模型规模——推理成本才是真正的瓶颈
- 训练前用小规模实验（如 1B 参数）验证缩放趋势，再外推到目标规模
- 注意"数据重复"的影响——Chinchilla 假设数据不重复，但实际训练中重复数据的收益递减

---

## 7. 常见错误

### 错误 1：认为参数量越大越好

**现象：** 训练了一个 175B 参数的模型，发现推理延迟无法接受，但性能提升有限。

**原因：** Kaplan 缩放定律表明参数量的收益是递减的——10 倍参数只带来 20% 的损失改善。Chinchilla 证明在固定计算预算下，过度增加参数量是次优的。

**修复：**
```python
# 错误：盲目追求大参数量
model_params = 175e9  # GPT-3 规模

# 正确：根据 Chinchilla 最优分配
C = 6 * 175e9 * 300e9  # GPT-3 的计算预算
N_opt = (C / 120.0) ** 0.5
print(f"最优参数量: {N_opt:.2e}")  # 远小于 175B
```

### 错误 2：忽略推理成本

**现象：** 训练时关注训练 FLOPs，部署后发现推理成本远超预期。

**原因：** 训练 FLOPs $\approx 6ND$，推理 FLOPs $\approx 2NL$（每个词元）。训练是一次性的，推理是持续的。一个 175B 模型推理 2048 词元需要 7.17e14 FLOPs——每次调用都是这个成本。

**修复：**
```python
# 错误：只看训练成本
train_flops = 6 * 175e9 * 300e9  # ~3.15e23

# 正确：同时评估推理成本
train_cost = 6 * 175e9 * 300e9
infer_cost_per_token = 2 * 175e9  # 3.5e11 FLOPs/词元
print(f"训练总 FLOPs: {train_cost:.2e}")
print(f"推理每词元: {infer_cost_per_token:.2e}")
```

### 错误 3：混淆 Chinchilla 比率的实际含义

**现象：** 看到 Llama 3 8B 的 Chinchilla 比率是 1875x，认为它违反了缩放定律。

**原因：** Chinchilla 最优是针对"给定训练计算预算"的。Llama 3 的目标不是最小化训练损失，而是最小化"训练 + 推理"的总成本。高 Chinchilla 比率意味着更多的训练成本，但换来更低的推理成本。

**修复：**
```python
# 错误：认为 1875x 违反了 Chinchilla
chinchilla_ratio = 15e12 / 8e9  # 1875x
print(f"这违反了 Chinchilla！")

# 正确：理解"过度训练"是优化目标不同
# Chinchilla 最优：最小化训练损失
# 过度训练：最小化总成本（训练 + 推理）
train_cost = 6 * 8e9 * 15e12
infer_savings = (2 * 175e9) / (2 * 8e9)  # 22x 推理加速
print(f"推理成本节省 {infer_savings:.0f} 倍，值得多花训练成本")
```

---

## 8. 面试考点

### Q1：Kaplan 缩放定律的核心结论是什么？（难度：）

**参考答案：**
语言模型的交叉熵损失与参数量、数据量、计算量呈幂律关系：$L \propto N^{-0.076}$，$L \propto D^{-0.095}$，$L \propto C^{-0.050}$。这意味着增加 10 倍参数量只带来约 20% 的损失改善——收益是递减的。参数和数据需要平衡增长，不能只增加其中一个。

### Q2：Chinchilla 修正了 Kaplan 的什么结论？为什么？（难度：）

**参考答案：**
Kaplan 认为"越大越好"——增加参数量总是值得的。Chinchilla 证明这是错的：Kaplan 的实验中模型只训练了少量轮次，没有达到数据饱和。当模型训练足够多轮次后，Chinchilla 发现最优策略是参数量和训练词元数同比例增长（$D^*/N^* \approx 20$）。GPT-3 的参数量比 Chinchilla 最优多了约 4 倍。

### Q3：为什么 2026 年的模型普遍"过度训练"？（难度：）

**参考答案：**
Chinchilla 最优是针对"给定训练计算预算最小化损失"的。但实际部署中，推理成本（持续的）比训练成本（一次性的）更重要。Llama 3 8B 用 15T 词元训练（Chinchilla 最优约 8B 词元），训练成本增加了约 1875 倍，但推理成本降低了 22 倍。对于高流量服务，推理节省远超训练多花的成本。

### Q4：如何估算一个模型的训练 FLOPs 和推理 FLOPs？（难度：）

**参考答案：**
训练 FLOPs $\approx 6ND$（$N$ = 参数量，$D$ = 训练词元数，6 = 前向 2 + 反向 4）。推理 FLOPs $\approx 2NL$（$L$ = 序列长度，2 = 每个词元需要约 2N 次运算）。例如 GPT-3：训练 FLOPs $\approx 6 \times 175B \times 300B \approx 3.15 \times 10^{23}$，推理 FLOPs（2048 词元）$\approx 2 \times 175B \times 2048 \approx 7.17 \times 10^{14}$。

### Q5：如果你要训练一个 7B 参数的模型，应该如何决定训练词元数？（难度：）

**参考答案：**
需要考虑三个因素：(1) Chinchilla 最优：$D^* \approx 20N = 140B$ 词元；(2) 推理成本约束：如果模型会高流量部署，值得过度训练到 1T+ 词元以降低推理延迟；(3) 数据可用性：高质量中文数据有限时，过多词元可能导致数据重复，收益递减。实践中，Llama 3 7B 用 2T 词元，Qwen2.5 7B 用 18T 词元——具体选择取决于部署场景和数据质量。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 缩放定律 | "模型越大越好" | 损失与参数量/数据量/计算量的幂律关系——收益递减，需要平衡增长 |
| Chinchilla 最优 | "最优模型大小" | 固定计算预算下，参数量和训练词元数同比例增长时损失最小（$D^*/N^* \approx 20$） |
| 过度训练 | "训练太多轮次" | 用远多于 Chinchilla 最优的数据训练小模型——以训练成本换推理成本 |
| 推理 FLOPs | "推理有多快" | 2 x 参数量 x 序列长度——决定生产部署的延迟和成本 |
| 训练 FLOPs | "训练花了多少算力" | 6 x 参数量 x 训练词元数——前向（2）+ 反向（4）传播的总计算量 |
| 幂律关系 | "线性关系" | $y = a \cdot x^b$ 形式的关系——在对数坐标下呈直线，$b$ 是幂律指数 |

---

## 📚 小结

Kaplan 缩放定律揭示了损失与规模的幂律关系：$L \propto N^{-0.076}$，$D^{-0.095}$。Chinchilla 修正了"越大越好"的结论——给定计算预算，参数和数据同比例增长最优。2026 年的主流是"过度训练"——8B 参数 x 15T 词元（Llama 3）在推理时比 175B x 300B（GPT-3）便宜 22 倍，性能相当。参数决定推理成本，数据决定训练质量——两者不能替代对方。

---

## ✏️ 练习

1. 【理解】用自己的话解释"幂律关系"在缩放定律中的含义。为什么增加 10 倍参数只带来 20% 的损失改善？写 150 字以内的说明，让一个没有 ML 背景的程序员也能听懂。

2. 【实现】修改 `code/main.py` 中的 `chinchilla_optimal` 函数，使其接受一个 `target_loss` 参数，返回能达到该损失的最小参数量和训练词元数。（提示：用二分搜索或牛顿法求解 $L(N, D) = target$）

3. 【实验】估算以下模型的训练 FLOPs 和推理 FLOPs（2048 词元）：(a) Llama 3 70B（70B 参数，15T 词元）；(b) Qwen2.5 72B（72B 参数，18T 词元）。哪个模型的训练/推理成本比更高？

4. 【思考】假设你要训练一个中文领域的 3B 参数模型，训练数据有 500B 高质量中文词元。根据 Chinchilla 公式，这个数据量是否足够？如果不够，你会如何获取更多数据？考虑数据重复的影响。

5. 【设计】为一个日活 100 万的聊天应用设计模型选型方案。用户平均每次对话 500 词元。比较以下三个方案的月度推理成本：(a) GPT-3 175B；(b) Llama 3 8B；(c) Llama 3 70B。假设 H100 GPU 推理速度为 1000 TFLOPs，每小时成本 $2。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 缩放定律 NumPy 实现 | `code/main.py` | Kaplan 幂律、Chinchilla 最优分配、推理成本对比 |
| 公式速查手册 | `outputs/scaling-laws-guide.md` | 所有缩放定律公式的汇总和实际模型对照表 |

---

## 📖 参考资料

1. [论文] Kaplan et al. "Scaling Laws for Neural Language Models". arXiv:2001.08361, 2020.
2. [论文] Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla). arXiv:2203.15556, 2022.
3. [论文] Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models". arXiv:2307.09288, 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
