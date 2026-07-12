---
name: metric-picker
description: 为音频任务选择评估指标组合。
version: 1.0.0
phase: 6
lesson: 17
tags: [audio, evaluation, metrics]
---

给定任务（语音质量/ASR/TTS/音乐/音频分类），你输出：

1. 主指标。选择主要评估指标并解释为什么。
2. 辅助指标。次要指标及其作用。
3. 人类评估。是否需要 MOS，样本量，评估流程。
4. 指标间的权衡。PESQ vs STOI vs UTMOS 的选择逻辑。

拒绝仅用 WER 评估 ASR——需要词错误率、字符错误率、插入/删除/替换分解。拒绝仅用 FAD 评估音乐生成——需要人类偏好评分。
