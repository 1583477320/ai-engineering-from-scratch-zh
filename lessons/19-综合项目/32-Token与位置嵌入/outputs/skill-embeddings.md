# Token与位置嵌入技能

## 目标
构建token嵌入+学习位置嵌入+正弦位置嵌入并合成为Transformer输入。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | Token嵌入正确 | (B,T)输入→(B,T,D)输出 |
| 20 | 位置嵌入形状 | 学习位置嵌入和正弦位置嵌入都产生(T,D) |
| 20 | 合成 | 逐元素相加，(B,T,D)形状断言 |
| 20 | 邻位余弦相似度 | 学习嵌入随机性vs正弦嵌入平滑衰减 |
| 15 | 参数计数 | 正弦嵌入0参数断言 |

## 构建检查清单

- [ ] TokenEmbedding(nn.Module)
- [ ] LearnedPositionalEmbedding(nn.Module)
- [ ] SinusoidalPositionalEmbedding(nn.Module)
- [ ] EmbeddingComposer(nn.Module)
- [ ] 形状断言((B,T,D))
- [ ] 参数计数(正弦=0)
- [ ] 邻位余弦相似度曲线
- [ ] 长度外推(学习嵌入在max_context_length外出错)
