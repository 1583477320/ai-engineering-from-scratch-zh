# 完整 Transformer

> 从词元嵌入到注意力到前馈网络到残差连接——组装一个可以训练的 Transformer 块。这是阶段 07 的毕业设计。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 02-04
**时间：** ~120 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零组装一个完整的 Transformer 块——嵌入 + 位置编码 + 多头注意力 + 前馈网络 + 残差连接 + 层归一化
- [ ] 构建编码器和解码器栈——包括交叉注意力和因果掩码
- [ ] 在简单分类任务上验证模型可以训练——理解从理论到实现的关键细节

---

## 1. 问题

前面三课分别构建了自注意力、多头注意力和位置编码。现在需要把它们组装成一个可以训练的完整架构——这是从"理解组件"到"构建系统"的关键跃迁。

---

## 2. 概念——Transformer 块的解剖

```
输入嵌入 + 位置编码
        ↓
┌─────────────────────┐
│ 多头自注意力         │
│ 残差连接 + 层归一化  │
│ 前馈网络 (FFN)       │
│ 残差连接 + 层归一化  │
└─────────────────────┘
        ↓
N 层堆叠
        ↓
输出
```

**每个组件的作用：**

| 组件 | 作用 |
|---|---|
| 嵌入 + 位置编码 | 将词元 ID 转换为带位置信息的向量 |
| 多头注意力 | 让每个词元关注其他词元——学习关系 |
| 前馈网络 (FFN) | 两层线性层 + GELU — 对每个位置独立处理 |
| 残差连接 | `x + attention(x)` — 梯度直通，防止梯度消失 |
| 层归一化 | 稳定训练——归一化每个层的输出 |

---

## 3. 从零实现

```python
import numpy as np

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        # 多头注意力
        self.mha = MultiHeadSelfAttention(d_model, n_heads, seed)
        # 前馈网络：d_model → d_ff → d_model
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / (d_model + d_ff)), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (d_ff + d_model)), (d_ff, d_model))
        self.d_model = d_model

    def forward(self, x):
        # 自注意力 + 残差连接
        attn_out, _ = self.mha.forward(x)
        x = self.layernorm(x + attn_out)
        # 前馈网络 + 残差连接
        ffn_out = np.maximum(0, x @ self.W1) @ self.W2  # GELU 近似
        x = self.layernorm(x + ffn_out)
        return x

    def layernorm(self, x):
        """层归一化：减均值，除标准差。"""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mean) / std
```

### 从零到可训练的清单

| 组件 | 状态 | 验证 |
|---|---|---|
| 词元嵌入 | ✓ 阶段 05 · 03 | 形状 (vocab, d_model) |
| 位置编码 | ✓ 阶段 07 · 04 | 可以加到嵌入上 |
| 多头自注意力 | ✓ 阶段 07 · 03 | 输出形状 (batch, seq, d_model) |
| FFN | ✓ 本课 | GELU 激活 |
| 残差连接 | ✓ 本课 | x + attn(x) |
| 层归一化 | ✓ 本课 | 稳定训练 |

---

## 4. 工具——PyTorch 实现

```python
import torch
import torch.nn as nn

class TransformerBlockPytorch(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力 + 残差 + 层归一化
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        # FFN + 残差 + 层归一化
        ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)

# 一个 6 层、512 维、8 头的 Transformer 编码器
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
    num_layers=6
)
x = torch.randn(2, 20, 512)  # (batch, seq_len, d_model)
out = encoder(x)
print(f"输入: {x.shape}, 输出: {out.shape}")
# 输入: torch.Size([2, 20, 512]), 输出: torch.Size([2, 20, 512])
```

---

## 5. 常见错误

### 错误 1：层归一化放在注意力之前

**现象：** 训练不稳定，loss 震荡。

**原因：** Pre-LN（先归一化再注意力）比 Post-LN（先注意力再归一化）在深层网络上更稳定。现代 Transformer 默认 Pre-LN。

### 错误 2：因果掩码缺失

**现象：** 解码器在训练时可以看到未来位置。

**原因：** 解码器必须是自回归的——位置 i 只能看位置 0..i。没有掩码的解码器训练会在推理时产生不一致的结果。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 前馈网络 | "中间的小网络" | 两层线性层 + GELU，对每个位置独立处理 |
| 残差连接 | "快捷连接" | x + attention(x)——梯度直通，防止深层网络梯度消失 |
| 层归一化 | "每层归一化" | 减均值除标准差，稳定训练 |
| 因果掩码 | "解码器掩码" | 位置 i 只能看位置 0..i，防止"看到未来" |

---

## 📚 小结

完整 Transformer 块 = 多头注意力 + FFN + 残差连接 + 层归一化。从零组装到训练，关键细节是：残差让梯度直通、层归一化稳定训练、因果掩码保持自回归性质。PyTorch 的 `TransformerEncoderLayer` 封装了这一切——理解每个组件的作用比记住 API 更重要。

---

## ✏️ 练习

1. 从零实现完整的 Transformer 编码器（6层 × 512 维 × 8 头），在玩具分类任务上验证可以训练
2. 比较 Pre-LN vs Post-LN：在相同超参下训练，记录训练损失曲线——哪个更稳定？

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurPR, 2017.
2. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
