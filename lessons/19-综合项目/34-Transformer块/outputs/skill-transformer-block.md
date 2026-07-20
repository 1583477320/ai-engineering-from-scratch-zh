# Transformer块技能

## 目标
实现Pre-LN和Post-LN Transformer块并展示训练稳定性差异。

## 构建检查清单

- [ ] LayerNorm（可学习scale+shift）
- [ ] MultiHeadAttention（融合QKV+因果掩码+残差）
- [ ] FeedForward（D→4D→D+GELU）
- [ ] TransformerBlock（Pre-LN/Post-LN切换）
- [ ] BlockStack（N层堆叠+最终LayerNorm）
- [ ] 梯度范数对比（Pre-LN >> Post-LN）
- [ ] 形状断言（输入=输出形状）
