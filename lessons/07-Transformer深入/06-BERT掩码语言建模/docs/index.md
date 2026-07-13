# BERT——掩码语言建模

> BERT 不是生成下一个词，而是预测被遮住的词。这个简单的训练目标教会了模型深度双向的语言理解。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 BERT 的掩码语言模型（MLM）训练目标——为什么遮住 15% 的词元比预测下一个词更高效
- [ ] 理解 BERT 的双向编码器架构——与 GPT 的单向解码器的根本区别
- [ ] 实现 BERT 的预训练流程——掩码选择 + 分类头 + 微调

---

## 1. 问题

GPT 预测下一个词——每个位置只能看到左边的上下文。这在生成任务上很好，但在理解任务上是浪费：**语言的真正含义由左右两边的上下文共同决定。**

BERT 的答案：不预测下一个词，而是**遮住 15% 的词元，让模型预测被遮住的是什么**。训练时看完整句子；推理时用 `[CLS]` 表示做分类。

---

## 2. 概念

### 2.1 MLM（掩码语言模型）

```
输入: "The cat [MASK] on the mat"
目标: 预测 [MASK] = "sat"

训练时随机遮住 15% 的词元：
- 80% 替换为 [MASK]
- 10% 替换为随机词
- 10% 保持不变
```

**为什么是 15%？** 太少→训练信号太弱。太多→上下文被破坏，模型学不到有用的东西。15% 是经验值——平衡了训练效率和上下文保留。

### 2.2 BERT 的双向编码

```
GPT（单向）:   [CLS] he → [CLS] he said → [CLS] he said that
               每一步只能看左边

BERT（双向）:  [CLS] he said that → 词元 "said" 同时看到 "he" 和 "that"
               所有词元同时看到左右上下文
```

这就是"预训练"和"微调"的核心差异——BERT 用无标签文本学习双向表示，然后用少量标注数据微调到下游任务。

---

## 3. 从零实现

### Step 1：掩码选择

```python
import random

def create_mlm_input(tokens, vocab_size, mask_token_id=103, mask_prob=0.15):
    """创建掩码输入：80% 替换为 [MASK]，10% 随机替换，10% 不变。"""
    labels = [-100] * len(tokens)  # -100 = 不计算损失
    output = list(tokens)
    
    for i in range(len(tokens)):
        if random.random() < mask_prob:
            labels[i] = tokens[i]  # 标签是原始词元
            r = random.random()
            if r < 0.8:
                output[i] = mask_token_id  # 80% 用 [MASK]
            elif r < 0.9:
                output[i] = random.randint(0, vocab_size - 1)  # 10% 随机
            # 10% 保持不变
    return output, labels
```

### Step 2：BERT 前向传播

```python
class BERTForMLM:
    def __init__(self, vocab_size, d_model=768, n_layers=12):
        # 词元嵌入 + 位置嵌入
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(512, d_model)
        # Transformer 编码器栈
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=12, dim_feedforward=3072, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # MLM 分类头
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # 嵌入
        x = self.embeddings(input_ids) + self.position_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        # Transformer 编码
        h = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        # MLM 预测
        logits = self.mlm_head(h)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss
```

### Step 3：微调用于下游分类

```python
def finetune_for_classification(num_labels=3):
    """冻结 BERT 主体，微调分类头。"""
    model = BERTForMLM(...)
    # 冻结 Transformer 层
    for param in model.encoder.parameters():
        param.requires_grad = False
    # 新增分类头
    classifier = nn.Linear(768, num_labels)
    return model, classifier
```

---

## 4. 常见错误

### 错误 1：掩码比例太高

**现象：** 模型在下游任务上表现差——上下文被严重破坏。

**原因：** 掩码超过 15% 时，词元之间的上下文信号被过度稀释。模型被迫在几乎没有上下文的情况下预测——这在推理时不存在的情况。15% 是最佳平衡点。

### 错误 2：微调时解冻了所有层

**现象：** 小数据集上过拟合严重。

**原因：** BERT 有 110M 参数，小数据集无法提供足够的训练信号。冻结底层，只微调上层（或分类头）更安全。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| MLM | "掩码语言模型" | 预测被遮住的词元——BERT 的预训练目标 |
| [CLS] | "分类 token" | BERT 输入的第一个特殊词元，其最终表示用于整个序列的分类 |
| 双向编码 | "同时看两边" | 与 GPT 的单向解码相反——BERT 可以同时看到左右上下文 |
| 微调 | "在下游任务上训练" | 冻结预训练层，训练分类头——少量数据即可 |

---

## 📚 小结

BERT 用掩码语言模型预训练——遮住 15% 的词元，预测被遮住的是什么。这个目标让模型学会了双向的语言理解。微调时冻结 Transformer 层，只训练分类头——少量标注数据即可达到好效果。与 GPT 的单向生成相比，BERT 在分类/问答/信息提取任务上更自然。

---

## ✏️ 练习

1. 实现 `create_mlm_input` 的完整版本，包括训练时的 `labels` 构建
2. 在 MNLI 数据集上用 BERT 做零样本分类——对比有/无微调的准确率

---

## 📖 参考资料

1. [论文] Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers". NAACL, 2019.
2. [论文] Lee et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach". 2019.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
