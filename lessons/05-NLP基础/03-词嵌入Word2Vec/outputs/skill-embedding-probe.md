---
name: embedding-probe
description: 检查 Word2Vec 模型质量——类比推理、最近邻、对称性、退化检测。
phase: 5
lesson: 03
---

给定 gensim KeyedVectors 和词表，你运行：

1. 三组经典类比测试。`国王 : 男人 :: 女王 : 女人`。`巴黎 : 法国 :: 东京 : 日本`。`walking : walked :: swimming : ?`。中文类比：`北京 : 中国 :: 东京 : 日本`。报告 top-1 结果和余弦相似度。
2. 五组领域特定词的最近邻测试（用户提供）。打印 top-5 邻居和余弦相似度。
3. 对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度内一致。
4. 退化检查。如任何嵌入的 L2 范数 < 0.01 或 > 100 → 训练 bug。标记。

拒绝仅凭类比准确率声明模型好。类比基准可以被操纵，不迁移到下游任务。推荐内在评估 + 下游评估联合。
