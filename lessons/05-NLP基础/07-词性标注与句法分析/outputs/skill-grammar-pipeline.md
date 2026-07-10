---
name: grammar-pipeline
description: 为下游 NLP 任务设计词性+依存分析流水线。
phase: 5
lesson: 07
---

给定下游任务（信息提取、改写验证、查询分解、词形还原），你输出：

1. 标签集。英文遗留系统用 Penn Treebank，多语言/跨语言用 Universal Dependencies。中文推荐 UD（spaCy 中文 / HanLP）。
2. 库。生产用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。中文轻量用 jieba.posseg。
3. 集成片段。3-5 行代码调用库并消费 `.pos_` / `.dep_` / `.head`。
4. 测试的失败模式。名动歧义（"学习"→NOUN/VERB）、介词短语附着歧义。抽取 20 条输出人工检查。

拒绝推荐自己写解析器。标记大小写处理不一致的流水线为脆弱。中文提示无屈折变化的词性推断挑战。
