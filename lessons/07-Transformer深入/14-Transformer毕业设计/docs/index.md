# Transformer 毕业设计

> 从零构建一个完整的 Transformer——编码器、解码器、训练循环、推理解码。这是阶段 07 的毕业考试。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 07 · 02-13 | **时间：** ~180 分钟

---

## 🎯 学习目标

- [ ] 构建一个可以训练的 Encoder-Decoder Transformer——用于简单的翻译或摘要任务
- [ ] 实现完整的训练循环——数据加载、优化器、损失计算、反向传播
- [ ] 实现贪婪解码和束搜索——对比两种推理策略的输出质量

---

## 1. 问题

前面 12 课分别构建了注意力、位置编码、BERT、GPT、T5 等组件。现在需要把它们组装成一个完整的、可以在真实任务上训练和推理的系统——这是从"理解组件"到"交付产品"的最终跃迁。

---

## 2. 概念

### 2.1 完整 Encoder-Decoder 架构

```
源序列 → [嵌入 + 位置编码] → [Encoder × N层] → 编码器输出
                                                     ↓
目标序列 → [嵌入 + 位置编码] → [Decoder × N层（因果掩码 + 交叉注意力）] → 输出
```

### 2.2 训练 vs 推理

| | 训练 | 推理 |
|---|---|---|
| 解码器输入 | 目标序列（教师强制） | 上一步的输出 |
| 掩码 | 因果掩码 | 因果掩码 |
| 注意力 | 交叉注意力看完整编码器输出 | 同上 |
| 损失 | 交叉熵 | 无（贪心/束搜索） |

---

## 3. 从零实现

### Step 1：数据加载 + 分词

```python
# 简单的英文-法文句子对
pairs = [
    ("the cat sat on the mat .", "le chat s'est assis sur le tapis ."),
    ("the dog ran across the room .", "le chien a couru a travers la piece ."),
]

# BPE tokenizer（简化版）
def simple_tokenizer(text):
    return text.lower().split()
```

### Step 2：构建并训练

```python
# 使用 PyTorch 的 Transformer
from torch import nn, optim

encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
    num_layers=2
)
decoder = nn.TransformerDecoder(
    nn.TransformerDecoderLayer(d_model=64, nhead=4, batch_first=True),
    num_layers=2
)

# 训练循环
for epoch in range(100):
    # 编码源序列
    enc_output = encoder(src_embed)
    # 解码（教师强制）
    dec_output = decoder(tgt_embed, enc_output)
    # 交叉熵损失
    loss = criterion(dec_output.view(-1, vocab_size), tgt_ids.view(-1))
    loss.backward()
    optimizer.step()
```

### Step 3：贪婪解码

```python
def greedy_decode(encoder, decoder, src, max_len=50):
    """逐词元解码——每步取概率最高的词。"""
    enc_out = encoder(src)
    decoder_input = torch.tensor([[BOS_TOKEN]])
    
    for _ in range(max_len):
        output = decoder(decoder_input, enc_out)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        if next_token == EOS_TOKEN:
            break
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]])], dim=1)
    
    return decoder_input[0].tolist()
```

### Step 4：束搜索

```python
def beam_search(encoder, decoder, src, beam_width=5, max_len=50):
    """束搜索——保持 beam_width 个最优候选序列。"""
    enc_out = encoder(src)
    # ... 实现细节见 code/
```

---

## 4. 工具——PyTorch 完整实现

```python
import torch
import torch.nn as nn

# 使用 PyTorch 内置的 Transformer
model = nn.Transformer(
    d_model=64, nhead=4,
    num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=256,
    batch_first=True
)

# 训练一个简单的序列到序列任务
src = torch.randint(0, 100, (32, 10))  # 源序列
tgt = torch.randint(0, 100, (32, 10))  # 目标序列
output = model(src, tgt[:, :-1], tgt_mask=...)
loss = nn.CrossEntropyLoss()(output.reshape(-1, 100), tgt[:, 1:].reshape(-1))
```

---

## 5. 常见错误

### 错误 1：解码器训练时用了教师强制但推理时不用

**现象：** 训练 loss 很低但推理输出乱码。

**原因：** 训练时解码器输入是目标序列（教师强制），推理时是上一步输出——训练和推理分布不同。用 `teacher_forcing_ratio` 从 1.0 逐步降到 0.0 来缓解。

### 错误 2：忘记添加 EOS token

**现象：** 解码器永远不会停止生成——无限循环。

**原因：** 没有终止条件——解码器不断预测下一个词直到达到 max_len。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 教师强制 | 训练时用目标序列作为解码器输入，加速收敛 |
| 束搜索 | 保持 K 个候选序列，避免贪心解码的局部最优 |
| 交叉注意力 | 解码器关注编码器的所有位置——将"理解"和"生成"连接 |
| 早期停止 | 当解码器输出 EOS token 时停止生成 |

---

## 📚 小结

阶段 07 的毕业设计——从零构建完整的 Encoder-Decoder Transformer。包含嵌入、位置编码、编码器、解码器、训练循环和推理解码。这是从"理解组件"到"构建完整系统"的最终跃迁。理解每个组件的作用，比记住 API 更重要。

---

## ✏️ 练习

1. 在玩具翻译数据集上训练你的 Transformer，达到困惑度 < 10
2. 对比贪婪解码和束搜索（beam_width=5）的输出质量——记录 BLEU 分数差异
3. 用你的 Transformer 翻译 10 句中文→英文，打印结果

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". 2017.
2. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
