---
name: lm-baseline
description: 用 N-gram 语言模型建立文本生成的 baseline。
phase: 5
lesson: 16
---

给定文本生成任务（领域、数据量、延迟预算），你输出：

1. 方案。N-gram LM + Kneser-Ney 平滑（设备端/微秒级延迟/可解释性）、RNN/LSTM（流式/中等数据）、Transformer（数据充足/精度优先）。
2. N-gram 配置。n=3（三元）或 n=4（四元）。Kneser-Ney 平滑的 discount 默认 0.75。
3. 评估。困惑度（Perplexity）为主要指标。越低越好。GPT-3 在 WikiText 上 ~20，Bigram ~200，作为校准参考。
4. 一个陷阱。N-gram 模型在未见 n-gram 上概率为 0——没有平滑的 perplexity = ∞。始终用 Kneser-Ney 或至少 Laplace。

拒绝在 > 1000 万 token 的语料上推荐从头训练 N-gram LM（此时神经模型更适合）。中文提示分词的依赖——词级别的 N-gram 需要先 jieba。
