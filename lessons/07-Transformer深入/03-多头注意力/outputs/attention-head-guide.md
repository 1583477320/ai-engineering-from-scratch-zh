# 注意力头分析指南

> 分析每个注意力头的学习模式，帮助理解模型关注了什么。

## 使用方法

```python
import numpy as np
from code.main import MultiHeadSelfAttention, AttentionHeadAnalyzer

# 创建示例
rng = np.random.default_rng(42)
sentence = ["我", "喜欢", "吃", "苹果"]
d_model = 16
X = rng.normal(0, 1, (len(sentence), d_model))

# 多头注意力
mha = MultiHeadSelfAttention(d_model, n_heads=4, seed=42)
output, all_weights = mha.forward(X)

# 分析每个头
analyzer = AttentionHeadAnalyzer(sentence)
analysis = analyzer.analyze(all_weights)

for a in analysis:
    print(f"头 {a['head'] + 1}:")
    print(f"  平均注意力范围: {a['avg_range']:.2f}")
    print(f"  最大注意力跨度: {a['max_span']}")
    print(f"  注意力熵: {a['entropy']:.3f}")
```

## 输出示例

```
头 1:
  平均注意力范围: 2.50
  最大注意力跨度: 3
  注意力熵: 1.234

头 2:
  平均注意力范围: 1.75
  最大注意力跨度: 2
  注意力熵: 0.987
```

## 指标解释

| 指标 | 含义 | 高值 | 低值 |
|------|------|------|------|
| 平均注意力范围 | 每个位置平均关注多少个其他位置 | 长距离依赖 | 局部关注 |
| 最大注意力跨度 | 最远的关注距离 | 捕获远距离关系 | 只关注邻近位置 |
| 注意力熵 | 分布的分散程度 | 分散关注多个位置 | 集中关注少数位置 |

## 常见模式

1. **语法头**：关注主语-谓语、动词-宾语关系
2. **语义头**：关注语义相似的词元
3. **位置头**：关注邻近位置的词元
4. **长距离头**：关注远距离的词元（如句子首尾）
