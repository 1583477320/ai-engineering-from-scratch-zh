---
name: text-encoder-picker
description: 给定约束集选择合适的文本编码架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），你输出：

1. 编码架构。TextCNN（边缘设备）、BiLSTM（流式/小数据）、BiLSTM-CRF（序列标注）、Transformer fine-tune（数据充足）、或"冻结 Transformer + 小分类头"（折中）。
2. 嵌入输入。随机初始化、GloVe/FastText 冻结、或上下文 Transformer 嵌入。中文推荐 FastText 或 BERT 中文字级别嵌入 + 小 CNN。
3. 5 行训练配方。优化器、学习率、batch size、epoch、正则化。
4. 一个监控信号。RNN/CNN：按序列长度分桶检查准确率。Transformer fine-tune：前 100 步检查训练 loss 是否崩溃。

拒绝在 < 500 标注条下推荐 Transformer fine-tune（除非 TextCNN/BiLSTM baseline 已饱和）。标记边缘部署需要架构优先决策。
