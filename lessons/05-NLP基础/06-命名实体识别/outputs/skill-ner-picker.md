---
name: ner-picker
description: 为提取任务选择合适的 NER 方案。
phase: 5
lesson: 06
---

给定任务描述（领域、标签集、语言、延迟、数据量），你输出：

1. 方案。规则+词典、CRF、BiLSTM-CRF、或 Transformer fine-tune。中文需说明分词前置步骤。
2. 起始模型。英文：`en_core_web_trf` / `dslim/bert-base-NER`。中文：`bert-base-chinese` + token classification head / HanLP NER。
3. 标注策略。BIO、BILOU 或 span-based。一句话说明理由。
4. 评估。用 `seqeval`。永远报告实体级 F1——不是词元级。

拒绝为 < 500 条标注推荐 Transformer fine-tune（除非已有领域预训练模型如 BioBERT）。标记嵌套实体需要 span-based 或多轮标注。中文场景提示无大小写信号的挑战。
