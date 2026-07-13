# 注意力权重可视化工具

> 可视化任意注意力权重矩阵，帮助理解模型关注模式。

## 使用方法

```python
import numpy as np
from code.main import (
    SelfAttention,
    MultiHeadSelfAttention,
    create_causal_mask,
    softmax
)

def visualize_attention(weights, labels, title="注意力权重"):
    """可视化注意力权重矩阵。

    Args:
        weights: 形状 (n, n) 的注意力权重
        labels: 词元标签列表
        title: 图表标题
    """
    n = len(labels)
    print(f"\n{title}")
    print("-" * (8 + n * 8))

    # 表头
    header = "查询\\键"
    for label in labels:
        header += f"{label:>8}"
    print(header)
    print("-" * (8 + n * 8))

    # 每行
    for i, query_label in enumerate(labels):
        row = f"{query_label:>8}"
        for j in range(n):
            w = weights[i, j]
            if w > 0.3:
                row += f"  \033[1m{w:.3f}\033[0m"
            else:
                row += f"  {w:.3f}"
        print(row)


# 演示
if __name__ == "__main__":
    # 创建示例
    rng = np.random.default_rng(42)
    sentence = ["我", "喜欢", "吃", "苹果"]
    d_model = 16
    X = rng.normal(0, 1, (len(sentence), d_model))

    # 双向注意力
    attn = SelfAttention(d_model, dk=4, dv=4, seed=42)
    output, weights双向 = attn.forward(X)
    visualize_attention(weights双向, sentence, "双向自注意力")

    # 因果掩码
    mask = create_causal_mask(len(sentence))
    output_causal, weights因果 = attn.forward(X, mask=mask)
    visualize_attention(weights因果, sentence, "因果自注意力（解码器模式）")
```

## 输出示例

```
双向自注意力
----------------------------------------------
查询\键       我    喜欢       吃      苹果
----------------------------------------------
      我   0.452    0.231    0.187    0.130
    喜欢   0.198    0.412    0.256    0.134
       吃   0.156    0.289    0.398    0.157
     苹果   0.143    0.198    0.267    0.392

因果自注意力（解码器模式）
----------------------------------------------
查询\键       我    喜欢       吃      苹果
----------------------------------------------
      我   1.000    0.000    0.000    0.000
    喜欢   0.312    0.688    0.000    0.000
       吃   0.245    0.398    0.357    0.000
     苹果   0.198    0.267    0.312    0.223
```

## 关键观察

1. **双向注意力**：每个位置都可以关注所有位置，权重矩阵是对称的
2. **因果掩码**：上三角被遮挡（权重为 0），每个位置只关注自己和之前的位置
3. **对角线**：在因果掩码中，最后一个位置的对角线值通常最高（因为它是唯一能看到所有信息的位置）
