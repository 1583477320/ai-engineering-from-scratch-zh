# T5/BART——编码器-解码器架构

> BERT 理解文本，GPT 生成文本，T5/BART 同时理解并生成。编码器-解码器架构是 NLP 中"最全能"的选择。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）、06（BERT）、07（GPT）
**时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 09（视觉 Transformer）— 对比文本与视觉的 Transformer 应用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 比较 BERT（编码器）、GPT（解码器）、T5（编码器+解码器）的架构差异
- [ ] 解释 T5 的"Text-to-Text"统一框架——为什么翻译、摘要、QA 都用同一个模型
- [ ] 理解 BART 与 T5 的差异——BART 用去噪自编码预训练，T5 用 span corruption

---

## 1. 问题

BERT 擅长理解（分类、问答），但不能生成。GPT 擅长生成，但生成是单向的。T5 和 BART 将两者结合：**编码器理解输入，解码器生成输出**。翻译、摘要、问答——所有"输入序列→输出序列"的任务都用同一个架构。

---

## 2. 概念

### 2.1 三种 Transformer 变体

| 架构 | 组件 | 生成 | 代表模型 |
|---|---|---|---|
| 仅编码器 | 只有 Transformer 编码器 | ❌ | BERT, RoBERTa |
| 仅解码器 | 只有 Transformer 解码器 | ✅ | GPT, LLaMA |
| 编码器+解码器 | 编码器 + 解码器 + 交叉注意力 | ✅ | T5, BART, BLOOM |

### 2.2 T5——"Text-to-Text"

T5 的核心思想：**所有任务都是文本到文本。**

```
翻译:  "translate English to French: The cat sat." → "Le chat s'est assis."
摘要:  "summarize: [长文本]" → "简短摘要"
分类:  "classify: [文本]" → "positive"
QA:    "answer: 问题 + 上下文" → "答案"
```

一个模型，一个框架，所有 NLP 任务。

### 2.3 BART vs T5

| | BART | T5 |
|---|---|---|
| 预训练 | 去噪自编码（破坏文本→恢复） | 文本到文本（span corruption） |
| 架构 | BERT 编码器 + GPT 解码器 | 标准编码器-解码器 |
| 强项 | 文本生成、摘要、翻译 | 多任务统一 |
| 论文 | Lewis et al. 2020 | Raffel et al. 2020 |

---

## 3. 从零实现

### 编码器-解码器架构

```python
class EncoderDecoderTransformer:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_encoder_layers=3, n_decoder_layers=3):
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_encoder_layers)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, d_ff, n_decoder_layers)

    def encode(self, input_ids):
        return self.encoder(input_ids)

    def decode(self, decoder_ids, encoder_output):
        # 自注意力（因果掩码）+ 交叉注意力（看编码器输出）
        return self.decoder(decoder_ids, encoder_output)

    def forward(self, input_ids, decoder_ids):
        encoder_output = self.encode(input_ids)
        logits = self.decode(decoder_ids, encoder_output)
        return logits
```

**关键区别：** 解码器的交叉注意力关注编码器的输出——这让生成时可以"回看"输入的全部信息。

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 HuggingFace T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载 T5 模型
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 翻译任务
input_text = "translate English to French: The cat sat on the mat."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# "Le chat s'est assis sur le tapis."
```

### 4.2 HuggingFace BART

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# 加载 BART 模型
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 摘要任务
text = "Your long article here..."
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

### 4.3 性能对比

| 模型 | 参数量 | 架构 | 适用任务 |
|---|---|---|---|
| T5-small | 60M | 编码器-解码器 | 翻译、摘要、分类 |
| T5-base | 220M | 同上 | 多任务统一 |
| T5-large | 770M | 同上 | 高质量生成 |
| BART-base | 140M | BERT+GPT | 生成任务更强 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

T5 和 BART 在大语言模型时代仍然有独特的价值。虽然 GPT 系列是生成任务的标配，但 T5 在需要"理解输入后生成"的场景（翻译、摘要、结构化生成）中仍然高效。

Google 的 T5 系列和 Meta 的 BART 系列在许多 NLP 基准测试中仍然是强基线。它们的优势在于：比相同参数规模的 GPT 生成质量更高（因为有编码器理解输入），同时比 GPT 更快（编码器一次前向，解码器逐步生成）。

### 5.2 LLM 时代什么变了？

**从独立模型到统一模型。** T5 的"Text-to-Text"思想被大语言模型继承——ChatGPT 可以用同一个模型做翻译、摘要、分类、问答。但 T5 是通过为每个任务加前缀实现的，而 GPT 是通过指令微调实现的。

**编码器-解码器 vs 仅解码器。** 大语言模型普遍使用仅解码器架构（GPT、LLaMA）。编码器-解码器在生成质量上更好，但推理速度更慢（两次前向）。速度战胜了质量。

### 5.3 什么没变？

**交叉注意力没变。** 即使在仅解码器架构中，视觉语言模型（如 GPT-4V）也使用交叉注意力让模型"看"图像。

**Text-to-Text 思想没变。** 所有大语言模型都是文本到文本的——输入文本，输出文本。分类任务也被转化为文本生成任务。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中要求"翻译这个句子"时，你正在使用 T5 开创的"Text-to-Text"思想——模型把"翻译"看作文本到文本的任务。

T5 和 ChatGPT 的核心区别：T5 需要为每个任务加前缀（如"translate: "），而 ChatGPT 通过指令微调理解自然语言指令。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 翻译 | T5 / mT5 | 多语言支持好 |
| 摘要 | BART / PEGASUS | 摘要质量高 |
| 结构化生成 | T5 | Text-to-Text 框架 |
| 生产环境 | ONNX 导出 | 推理加速 2-4x |

### 6.2 中文场景特别建议

- 中文翻译使用 mT5（多语言 T5）——支持 101 种语言
- 中文摘要使用 BART——对中文生成任务效果更好

### 6.3 踩坑经验

- 编码器和解码器的输入长度不同——注意设置 `max_length`
- 交叉注意力的计算开销大——避免过长的编码器输入
- T5 的前缀（如 "translate: "）必须与训练时的格式一致

---

## 7. 常见错误

### 错误 1：编码器和解码器混淆

**现象：** 将解码器输入直接送给编码器——模型输出异常。

**原因：** 编码器处理输入序列（无掩码），解码器处理输出序列（有因果掩码）。两者架构不同。

**修复：**
```python
# ❌ 错误写法：解码器输入送给编码器
enc_output = encoder(decoder_input)

# ✓ 正确写法：编码器处理输入，解码器处理输出
enc_output = encoder(input_ids)
dec_output = decoder(decoder_ids, enc_output)
```

### 错误 2：忘记交叉注意力的掩码

**现象：** 解码器所有位置都关注编码器的全部位置——无法生成。

**原因：** 交叉注意力中解码器关注编码器——解码器的每个位置都可以看到编码器的全部位置。这是正确的——不需要掩码。

### 错误 3：T5 前缀格式错误

**现象：** T5 生成结果与期望不符。

**原因：** T5 依赖任务前缀确定任务类型。前缀格式必须与训练时一致。

**修复：**
```python
# ❌ 错误写法：无前缀
input_text = "The cat sat on the mat."

# ✓ 正确写法：正确的前缀
input_text = "translate English to French: The cat sat on the mat."
```

---

## 8. 面试考点

### Q1：编码器-解码器架构和仅解码器架构有什么区别？（难度：⭐⭐）

**参考答案：**
编码器-解码器架构有一个编码器（双向注意力）和一个解码器（因果注意力 + 交叉注意力）。编码器理解输入，解码器生成输出。仅解码器架构只有一个解码器（因果注意力）——同时处理输入和生成输出。

### Q2：交叉注意力的作用是什么？（难度：⭐⭐）

**参考答案：**
交叉注意力连接编码器和解码器——解码器的每一步都可以关注编码器的所有位置。这让生成时可以"回看"输入的全部信息。没有交叉注意力，解码器只能靠自己生成——失去了输入信息。

### Q3：T5 为什么用"Text-to-Text"框架？（难度：⭐⭐⭐）

**参考答案：**
T5 将所有 NLP 任务统一为文本到文本格式——输入文本，输出文本。翻译、摘要、分类、问答都用同一个模型，只需要改变输入的前缀。这简化了模型设计和部署——一个模型做所有任务。

### Q4：BART 和 T5 的预训练目标有什么区别？（难度：⭐⭐⭐）

**参考答案：**
BART 使用去噪自编码预训练——破坏输入文本（掩码、删除、排列等），让模型恢复原始文本。T5 使用 span corruption——将连续的词元替换为哨兵词元，让模型生成被替换的跨度。

BART 在生成任务上更强，T5 在多任务统一上更好。

### Q5：为什么大语言模型普遍使用仅解码器架构而不是编码器-解码器？（难度：⭐⭐⭐）

**参考答案：**
1. 推理速度：编码器-解码器需要两次前向（编码 + 解码），仅解码器只需要一次
2. 简单性：仅解码器架构更简单，更容易扩展
3. 统一性：仅解码器可以同时处理理解（通过 prompt）和生成任务
4. 研究趋势：GPT-3 的成功验证了仅解码器在足够大时的有效性

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 编码器-解码器 | "Transformer 两半" | 编码器理解输入，解码器生成输出，交叉注意力连接两者 |
| T5 | "文本到文本" | 所有 NLP 任务都统一为输入文本→输出文本格式 |
| BART | "BERT + GPT" | BERT 风格编码器 + GPT 风格解码器，预训练为去噪自编码 |
| 交叉注意力 | "解码器看编码器" | 解码器每一步关注编码器的所有位置——生成时可以"回看"输入 |
| Text-to-Text | "文本到文本" | 所有 NLP 任务都视为文本生成问题 |
| Span Corruption | "跨度破坏" | T5 的预训练目标——破坏连续文本，模型恢复 |
| 去噪自编码 | "破坏后恢复" | BART 的预训练目标——破坏输入，恢复原始文本 |
| Seq2Seq | "序列到序列" | 输入序列→输出序列的通用框架 |

---

## 📚 小结

T5 和 BART 是"最全能"的 Transformer 变体——编码器理解、解码器生成。T5 用"文本到文本"框架统一所有任务；BART 用去噪预训练在生成任务上更强。两者在翻译、摘要、QA 上仍是强基线——虽然 LLM 正在取代它们，但它们更快、更小、在数据有限时更稳。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释编码器-解码器架构与仅解码器架构的区别。写 200 字以内的说明。

2. **【实现】** 用 T5 实现一个简单的翻译任务——理解编码器-解码器在 Seq2Seq 上的工作方式。

3. **【实验】** 比较 BERT、GPT、T5 在相同硬件上的参数量和推理速度——画出权衡曲线。

4. **【实现】** 从零实现一个简化的编码器-解码器模型（在 code/main.py 中），验证交叉注意力的工作方式。

5. **【思考】** 阅读 T5 论文的摘要部分，用你自己的话解释"Text-to-Text"框架为什么能统一所有 NLP 任务。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 编码器-解码器实现 | `code/main.py` | 编码器、解码器、交叉注意力的完整实现 |
| 三种架构对比指南 | `outputs/encoder-decoder-comparison.md` | BERT / GPT / T5 的详细对比 |

---

## 📖 参考资料

1. [论文] Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5). 2020.
2. [论文] Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". 2020.
3. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
