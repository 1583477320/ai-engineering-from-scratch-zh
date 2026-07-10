---
name: embeddings-picker
description: 为文本流水线选择词嵌入方案。
phase: 5
lesson: 04
---

给定任务和数据描述，你输出：

1. 嵌入策略。GloVe（通用英文）、FastText（形态丰富/新词/拼写错误）、BPE/子词 tokenizer（输入要进 Transformer）、TF-IDF（线性模型 + 可解释性优先）。
2. 预训练来源。英文：GloVe 300d / fastText `cc.en.300.bin`。中文：Chinese-Word-Vectors / fastText `cc.zh.300.bin`。
3. OOV 处理。FastText 子词组合（推荐），或字符 n-gram 回退，或 `[UNK]` 统一映射。
4. 一个可复现性的坑。Tokenizer 和模型必须配对——微调预训练 Transformer 时，永远不要换 tokenizer。

拒绝在无 OOV 容错需求时推 FastText（GloVe 更快更小）。对中文输入，提示需要先分词的依赖。
