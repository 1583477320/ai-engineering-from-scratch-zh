# 缩放定律公式速查手册

> 缩放定律（Scaling Laws）告诉你：给定固定计算预算，如何分配参数量和数据量才能让模型损失最小。

---

## 1. Kaplan 缩放定律（2020）

OpenAI Kaplan 等人发现语言模型的交叉熵损失与三个因素呈**幂律关系**：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

| 符号 | 含义 | 拟合值 |
|---|---|---|
| N | 模型参数量 | — |
| D | 训练词元数 | — |
| C | 总计算量（FLOPs） | — |
| $\alpha_N$ | 参数量幂律指数 | 0.076 |
| $\alpha_D$ | 数据量幂律指数 | 0.095 |
| $\alpha_C$ | 计算量幂律指数 | 0.050 |
| N_c, D_c, C_c | 饱和常数 | 8.8e13, 5.4e12, 1.6e10 |

**核心推论：**

- 参数量增加 10 倍，损失降低约 20%
- 数据量增加 10 倍，损失降低约 12%
- 参数和数据需要**平衡增长**，不能只增加其中一个

---

## 2. Chinchilla 缩放定律（2022）

DeepMind Hoffmann 等人修正了 Kaplan 的结论：

$$
L(N, D) = L_\infty + A \cdot N^{-\alpha} + B \cdot D^{-\beta}
$$

| 符号 | 含义 | 拟合值 |
|---|---|---|
| $L_\infty$ | 不可削减损失 | 3.0 |
| A | 参数量系数 | 406.4 |
| B | 数据量系数 | 410.7 |
| $\alpha$ | 参数量指数 | 0.34 |
| $\beta$ | 数据量指数 | 0.28 |

**Chinchilla 最优分配公式：**

给定计算预算 $C$（单位：FLOPs），最优参数量和训练词元数：

$$
N^* = \sqrt{\frac{C}{120}}, \quad
D^* = 20 \cdot N^*
$$

其中训练 FLOPs $\approx 6ND$（前向 + 反向传播）。

**Chinchilla 最优比率：** $D^* / N^* \approx 20$，即训练词元数应约为参数量的 20 倍。

---

## 3. 训练计算量估算

$$
C \approx 6 \cdot N \cdot D
$$

- $N$ = 模型参数量
- $D$ = 训练词元数
- 系数 6 = 前向传播（2）+ 反向传播（4）

---

## 4. 推理计算量估算

$$
\text{Inference FLOPs} \approx 2 \cdot N \cdot L
$$

- $N$ = 模型参数量
- $L$ = 序列长度
- 推理时每个词元的计算量约为 2N FLOPs

**关键洞察：** 推理成本由参数量决定，训练质量由数据量决定——两者不能相互替代。

---

## 5. 实际模型对照表

| 模型 | 参数量 | 训练词元 | Chinchilla 比率 | 推理 FLOPs（2048 词元） |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7x（参数过多） | 7.17e14 |
| Llama 3 8B | 8B | 15T | 1875x（数据过多） | 3.28e13 |
| Llama 3 70B | 70B | 15T | 214x（数据过多） | 2.87e14 |
| Llama 3 405B | 405B | 15T | 37x（数据过多） | 1.66e15 |
| Qwen2.5 7B | 7B | 18T | 2571x（数据过多） | 2.87e13 |

**2026 年主流策略是"过度训练"**——用远多于 Chinchilla 最优的数据训练小模型。

---

## 6. 公式速查

```
Kaplan 损失：        L(N) = (N_c / N)^alpha
Chinchilla 损失：    L(N, D) = 3.0 + 406.4 * N^(-0.34) + 410.7 * D^(-0.28)
Chinchilla 最优：    N* = sqrt(C / 120),  D* = 20 * N*
训练 FLOPs：         C ≈ 6 * N * D
推理 FLOPs：         2 * N * L
```

---

## 参考文献

1. Kaplan et al. "Scaling Laws for Neural Language Models". arXiv:2001.08361, 2020.
2. Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla). arXiv:2203.15556, 2022.
