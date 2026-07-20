# 直接偏好优化DPO技能

## 目标
从零实现DPO损失，在偏好对上训练策略。

## 构建检查清单

- [ ] InstructionTokenizer（INST/RESP）
- [ ] TinyGPT（decoder-only）
- [ ] 12个偏好三元组（prompt/chosen/rejected）
- [ ] seq_log_prob（序列级对数概率）
- [ ] dpo_loss（sigmoid + 对数比率差）
- [ ] 参考模型（冻结）+ 策略模型（可训练）
- [ ] warmup预训练
- [ ] train_dpo循环（每epoch记录margin）
- [ ] 损失下降+margin上升断言
