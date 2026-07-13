# BERT——掩码语言建模

> BERT 不是生成下一个词，而是预测被遮住的词。这个简单的训练目标教会了模型深度双向的语言理解。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）
**时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 07（GPT 因果语言建模）— 对比 BERT 双向编码与 GPT 单向解码

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

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# 加载预训练的 BERT 模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

# 输入文本，手动加 [MASK] 让 BERT 预测
text = "我 [MASK] 吃苹果"
inputs = tokenizer(text, return_tensors="pt")

# 预测被遮住的词
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 找到 [MASK] 的位置
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(f"预测结果: {predicted_token}")  # "想"、"要"、"喜欢" 等
```

### 4.2 PyTorch 实现

```python
import torch.nn as nn

# 标准 BERT 编码器层
encoder_layer = nn.TransformerEncoderLayer(
    d_model=768, nhead=12, dim_feedforward=3072,
    batch_first=True, activation="gelu"
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

# MLM 分类头
class MLMHead(nn.Module):
    def __init__(self, d_model=768, vocab_size=30522):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.decoder(self.layernorm(self.activation(self.dense(x))))
```

### 4.3 性能对比

| 模型 | 参数量 | 层数 | 头数 | d_model | 预训练数据 |
|---|---|---|---|---|---|
| BERT-base | 110M | 12 | 12 | 768 | 3.3B 词 |
| BERT-large | 340M | 24 | 16 | 1024 | 3.3B 词 |
| RoBERTa-base | 125M | 12 | 12 | 768 | 160GB 文本 |
| DistilBERT | 66M | 6 | 12 | 768 | 3.3B 词（蒸馏） |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

BERT 虽然称为"大语言模型"，但它严格来说是一个**编码器模型**（Encoder-only）。它不像 GPT 那样生成文本，而是为文本生成表示。

BERT 系列模型（包括 RoBERTa、DistilBERT、ALBERT）在以下场景仍然广泛使用：
- 文本分类（情感分析、意图识别）
- 命名实体识别（NER）
- 问答系统
- 文本相似度计算
- 信息提取

GPT 系列模型（ChatGPT、Claude、Llama 3）是**解码器模型**（Decoder-only），擅长生成文本。但如果你需要做分类或信息提取，BERT 通常更快、更轻量、更可控。

### 5.2 LLM 时代什么变了？

**从编码器到解码器。** BERT 发布时（2018 年），编码器-解码器架构是主流。但 GPT-3 之后（2020 年），纯解码器架构成为标准。原因：解码器既可以做生成，也可以做理解（通过改造）。编码器只能做理解。

**从微调到提示词工程。** BERT 时代，你需要微调模型才能在具体任务上使用。现在，你可以直接用提示词让大语言模型完成任务——无需微调。

**从单任务到通用。** BERT 需要为每个任务准备标注数据并微调。大语言模型可以用同一个模型做翻译、摘要、分类、问答——无需每个任务一个模型。

### 5.3 什么没变？

**双向编码的价值没变。** 即使在大语言模型时代，双向编码在理解任务上仍有价值。BERT 的分类准确率在同等参数规模下仍然优于 GPT。

**预训练 + 微调的范式没变。** BERT 开创的预训练 + 微调范式是现代大语言模型的基础。GPT 也是先预训练再微调（只是微调方式变成了指令微调和 RLHF）。

**掩码策略被继承。** 大语言模型在训练时也使用类似的掩码策略——SpanBERT 的连续掩码、T5 的跨度预测等都源于 BERT 的 MLM。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中输入"请问法国的首都是什么？"，它不会像 BERT 那样分析整个问题——而是逐个词元生成回答。

这就是编码器和解码器的根本区别：BERT 一次性看完整句话，GPT 一个个生成词元。

但你也可以在 ChatGPT 上做"掩码任务"——比如输入"法国的首都是[MASK]"，它会用生成能力代替掩码预测。不过注意：它生成的是完整序列，不是 BERT 那种分类预测。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 分类 / NER / QA | BERT-base / RoBERTa | 开源，110M 参数，推理速度快 |
| 深度学习 | DistilBERT | BERT 的蒸馏版本，参数量减半，速度翻倍 |
| 生产环境 | ONNX / TensorRT | 将 BERT 导出为 ONNX，推理加速 2-4x |

### 6.2 中文场景特别建议

- 使用 `bert-base-chinese` 时注意：它是字级别的，不产生子词
- 中文 MLM 任务中，掩码一个字的上下文比掩码一个词更难预测——考虑使用全词掩码
- 中文 BERT 的 `[CLS]` 表示可以用于文本分类，但需要微调

### 6.3 踩坑经验

- 微调时不要解冻所有层——小数据集上过拟合严重
- 冻结底层只想微调时，注意设置 `param.requires_grad = False`
- 掩码比例不要超过 15%——上下文被破坏后模型学不到有用的东西
- `attention_mask` 和 `token_type_ids` 不能混淆——一个用于填充，一个用于区分句子
- BERT 的 `max_length` 是 512——如果句子更长，需要截断或使用其他模型

---

## 7. 常见错误

### 错误 1：掩码比例太高

**现象：** 模型在下游任务上表现差——上下文被严重破坏。

**原因：** 掩码超过 15% 时，词元之间的上下文信号被过度稀释。模型被迫在几乎没有上下文的情况下预测——这在推理时不存在的情况。15% 是最佳平衡点。

**修复：**
```python
# ❌ 错误写法：掩码比例太高
mask_prob = 0.5  # 50% 的词元被遮住——上下文严重破坏

# ✓ 正确写法：15% 掩码比例
mask_prob = 0.15  # 15% 的词元被遮住——平衡上下文和训练信号
```

### 错误 2：微调时解冻了所有层

**现象：** 小数据集上过拟合严重。

**原因：** BERT 有 110M 参数，小数据集无法提供足够的训练信号。冻结底层，只微调上层（或分类头）更安全。

**修复：**
```python
# ❌ 错误写法：解冻所有层
for param in model.parameters():
    param.requires_grad = True  # 小数据集上过拟合

# ✓ 正确写法：冻结编码器，只微调分类头
for param in model.encoder.parameters():
    param.requires_grad = False
```

### 错误 3：未正确设置 ignore_index

**现象：** 损失包含非掩码位置的预测误差——训练信号太强，模型无法专注于掩码预测。

**原因：** 交叉熵损失的 `ignore_index` 参数未设置或设置错误。非掩码位置（标签为 -100）不应计算损失。

**修复：**
```python
# ❌ 错误写法：未设置 ignore_index
loss = nn.CrossEntropyLoss()(logits, labels)  # 所有位置都计算损失

# ✓ 正确写法：设置 ignore_index
loss = nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)  # 只计算掩码位置的损失
```

### 错误 4：忘记 [CLS] 的输出用于分类

**现象：** 分类时使用全部词元的平均表示而不是 [CLS]——性能下降。

**原因：** BERT 的设计中，[CLS] 的最终表示专门用于整个序列的分类。使用平均表示会丢失 BERT 的设计优势。

**修复：**
```python
# ❌ 错误写法：使用平均表示
cls_representation = h.mean(dim=1)  # 平均所有位置的表示

# ✓ 正确写法：使用 [CLS] 表示
cls_representation = h[:, 0, :]  # 取第一个位置 ([CLS]) 的表示
```

### 错误 5：位置编码选择错误

**现象：** 模型在长序列上性能下降。

**原因：** BERT 使用可学习位置编码——限制在训练时的最大长度（通常 512）。如果推理时遇到更长的序列，位置编码失效。

**修复：**
```python
# ❌ 错误写法：可学习编码固定在训练长度
position_embeddings = nn.Embedding(512, d_model)  # 超过 512 报错

# ✓ 正确写法：使用可以外推的编码
pe = sinusoidal_position_encoding(1024, d_model)  # 正弦编码可以外推到更长的序列
```

---

## 8. 面试考点

### Q1：BERT 和 GPT 的训练目标有什么区别？（难度：⭐⭐）

**参考答案：**
BERT 使用掩码语言模型（MLM）——遮住 15% 的词元，模型预测被遮住的是什么。GPT 使用因果语言模型（CLM）——预测下一个词元。

核心差异：BERT 是双向的——每个词元可以看到左右两边的上下文。GPT 是单向的——每个词元只能看到左边的上下文。BERT 擅长理解任务（分类、NER、QA），GPT 擅长生成任务（对话、翻译、摘要）。

### Q2：为什么 BERT 选择掩码 15% 而不是 50%？（难度：⭐⭐）

**参考答案：**
掩码比例是一个权衡。太少的掩码（如 5%）产生太少的训练信号——模型每次只能学几个词元。太多的掩码（如 50%）破坏上下文——模型没有足够的信息预测被遮住的词元。

15% 是经验选择的结果。在这个比例下，模型有足够的训练信号（每 100 个词元学 15 个），同时上下文大部分保留（85% 的词元可见）。

### Q3：BERT 中 [CLS] 的作用是什么？（难度：⭐⭐）

**参考答案：**
[CLS] 是 BERT 输入的第一个特殊词元，它的最终表示用于整个序列的分类。在预训练时，[CLS] 的表示也被用于下一句预测（NSP）任务。

[CLS] 的设计思路是：通过自注意力机制，[CLS] 可以与所有词元交互——它的最终表示包含了整个序列的信息。

### Q4：微调 BERT 时应该冻结哪些层？（难度：⭐⭐⭐）

**参考答案：**
微调策略取决于数据集大小：
- 小数据集（<1K 样本）：冻结全部 Transformer 层，只微调分类头
- 中等数据集（1K-10K 样本）：冻结底部 6 层，微调顶部 6 层 + 分类头
- 大数据集（>10K 样本）：全参数微调

### Q5：BERT 的缺点是什么？RoBERTa 如何改进？（难度：⭐⭐⭐）

**参考答案：**
BERT 的缺点：
1. 可学习位置编码限制了最大长度（512）
2. 下一句预测（NSP）任务对模型提升有限
3. 训练数据量不够大

RoBERTa 的改进：
1. 删除 NSP 任务，只保留 MLM
2. 使用更大规模的数据（160GB vs 3.3B 词）
3. 动态掩码策略（每次训练轮次重新生成掩码）
4. 更大批次大小 + 更多训练步数

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| MLM | "掩码语言模型" | 预测被遮住的词元——BERT 的预训练目标 |
| [CLS] | "分类 token" | BERT 输入的第一个特殊词元，其最终表示用于整个序列的分类 |
| 双向编码 | "同时看两边" | 与 GPT 的单向解码相反——BERT 可以同时看到左右上下文 |
| 微调 | "在下游任务上训练" | 冻结预训练层，训练分类头——少量数据即可 |
| NSP | "下一句预测" | BERT 的辅助预训练目标，预测两个句子是否相邻 |
| 全词掩码 | "把整个词遮住" | 对中文 BERT 的改进——如果一个字被选中，遮住整个词 |
| RoBERTa | "优化版 BERT" | 删除 NSP、动态掩码、更大数据——效果更好 |
| DistilBERT | "轻量版 BERT" | 知识蒸馏——参数减半，速度翻倍，性能保留 97% |

---

## 📚 小结

BERT 用掩码语言模型预训练——遮住 15% 的词元，预测被遮住的是什么。这个目标让模型学会了双向的语言理解。微调时冻结 Transformer 层，只训练分类头——少量标注数据即可达到好效果。与 GPT 的单向生成相比，BERT 在分类/问答/信息提取任务上更自然。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 BERT 使用 15% 的掩码比例而不是更高或更低。写 200 字以内的说明。

2. **【实现】** 实现 `create_mlm_input` 的完整版本，包括训练时的 `labels` 构建。

3. **【实现】** 实现从零到微调的完整流程：预训练 BERT（小规模数据）→ 保存检查点 → 微调分类头 → 评估。

4. **【实验】** 在 MNLI 数据集上用 BERT 做零样本分类——对比有/无微调的准确率。

5. **【思考】** 阅读 RoBERTa 论文，用你自己的话解释为什么删除 NSP 任务、动态掩码、更大数据能提升性能。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| BERT MLM 实现 | `code/main.py` | 掩码策略、BERT 模型、预测流程 |
| BERT 微调指南 | `outputs/bert-finetuning-guide.md` | 冻结策略、超参数、常见任务 |

---

## 📖 参考资料

1. [论文] Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers". NAACL, 2019.
2. [论文] Lee et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach". 2019.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
