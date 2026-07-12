---
name: classifier-designer
description: 为给定音频分类任务选择架构、增强策略、类别平衡方案和评估指标。
version: 1.0.0
phase: 6
lesson: 03
tags: [audio, classification, speech]
---

给定任务（类别数、数据量、标签类型、延迟预算），你输出：

1. **架构。** k-NN on MFCC / CNN on log-mel / AST 微调 / BEATs 微调 / Whisper-encoder 冻结。一句话理由，基于数据量和延迟约束。
2. **数据增强。** SpecAugment（时间掩蔽 + 频率掩蔽）参数 / Mixup 系数 / 其他增强。理由基于领域（语音 vs 环境音 vs 音乐）。
3. **类别平衡。** 平衡采样 / Focal Loss / 类别权重 / 不需要。理由基于类别分布。
4. **评估计划。** 指标（accuracy / mAP / 宏 F1）+ 分层验证 + 每类召回率。对不平衡数据拒绝只报准确率。

拒绝在数据量 > 500 条时不从冻结骨干微调开始。拒绝在多标签任务（如 AudioSet）上只报告准确率——必须报 mAP。拒绝在极度不平衡数据（> 10:1）上不使用类别平衡技术。
