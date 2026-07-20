# 结果评估器配方

## 裁决决策表

1. terminal != "ok" → failed
2. |improvement| < 2% → noise
3. p > 0.05 → noise
4. improvement > 0 → improved
5. else → regressed

## 改进计算

```python
if direction == "higher_is_better":
    improvement = (candidate - baseline) / abs(baseline)
else:
    improvement = (baseline - candidate) / abs(baseline)
```
