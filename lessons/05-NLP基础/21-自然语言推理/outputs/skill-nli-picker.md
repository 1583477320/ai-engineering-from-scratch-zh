---
name: nli-picker
description: 为生产场景选择 NLI 模型和部署模式。
phase: 5
lesson: 21
---

给定生产场景（幻觉检测/事实核查/零样本分类），你输出：

1. 模型。英文高精度：`deberta-v3-large-mnli`。英文通用：`roberta-large-mnli`。零样本分类：`bart-large-mnli`。中文/多语言：多语言 XLM-R NLI fine-tune。LLM-as-Judge：最后一道交叉验证。
2. 阈值。蕴含分数 > 0.9 = 高置信蕴含。0.5-0.9 = 不确定（建议人工复查）。< 0.5 = 非蕴含。
3. 部署模式。幻觉检测：摘要做 h，原文做 t → 非蕴含 = 幻觉。零样本分类：文档做 t，标签描述做 h → 蕴含 = 预测该类。
4. 一个陷阱。NLI 模型可能在训练分布外的文本上产生高置信的错误预测。始终在目标领域的 50 条样本上做人工 vs 模型一致性校准。

拒绝在无人工校准的情况下将 NLI 分数作为唯一幻觉判断依据。中文提示 CMNLI/OCNLI 数据集规模有限——从 XLM-R 开始 fine-tune 而非从头训练。
