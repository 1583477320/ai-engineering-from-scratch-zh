---
name: tokenizer-picker
description: 为语言模型或文本流水线选择子词分词方案。
phase: 5
lesson: 19
---

给定场景（从头训练 LLM / 微调现有模型 / 仅做分类检索），你输出：

1. 分词策略。BPE（GPT/Llama 生态）、Unigram（T5/mBART——子词正则化优势）、WordPiece（BERT 生态）。从头训练推荐 Unigram（子词正则化对数据增强有价值）。
2. 词表大小。英文单语：32K。多语言：64K-128K。中文：32K-50K（中文 token 信息密度更高）。
3. 训练库。SentencePiece（从头训练——Unicode 原生，中文友好）。tiktoken（仅推理——使用预训练词表）。HF Tokenizers（BPE/WordPiece 训练 + 编码）。
4. 一个可复现性的坑。Tokenizer 和模型权重是锁死的——微调预训练模型时永远不换 tokenizer。从头训练时 tokenizer 训练语料必须和模型预训练语料一致。

拒绝为微调预训练模型推荐训练自定义 tokenizer。对中文/日文等无空格语言标记 SentencePiece 的 `▁` 空白编码是预期行为，不是错误。
