# 余弦学习率预热配方

## 公式

$$
\text{lr}(t) = \text{lr}_{\text{min}} + \frac{1}{2}(\text{lr}_{\text{max}} - \text{lr}_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right)\right)
$$

## 三阶段

1. **预热**（0 → warmup_steps）：学习率从 0 线性提升到 lr_max
2. **余弦衰减**（warmup_steps → total_steps）：余弦曲线从 lr_max 降到 lr_min
3. **地板**（total_steps 后）：固定在 lr_min

## 典型配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| lr_max | 3e-4 到 3e-3 | 根据模型大小和批次大小扫描 |
| lr_min | lr_max × 1%-10% | 保持微小学习能力 |
| warmup_steps | min(1000, total_steps // 50) | 预热步数经验法则 |

## PyTorch 实现

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000,
)
```
