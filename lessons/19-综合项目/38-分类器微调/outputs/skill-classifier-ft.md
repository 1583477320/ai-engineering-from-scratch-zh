# 分类器微调技能

## 目标
通过头部替换实现分类器微调，对比冻结身体和全量微调。

## 构建检查清单

- [ ] LMBody（token+position+transformer块）
- [ ] Classifier（body+均值池化+线性头部）
- [ ] freeze_body/unfreeze_body
- [ ] train_classifier统一训练循环
- [ ] 评测（精确率/召回率/F1/混淆矩阵）
- [ ] 预训练身体+两种训练策略对比
- [ ] 合成spam/ham数据集
