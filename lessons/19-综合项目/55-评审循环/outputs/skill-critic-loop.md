# 评审循环配方

## 五维评分

clarity, novelty, evidence, methodology, related_work (0-10)

## 收敛规则

1. 所有维度 ≥ target (8.0) → target
2. 连续两轮提升 < epsilon (0.1) → plateau
3. 轮次 ≥ max (5) → budget

## 迹线格式

```json
{"round": N, "scores": {...}, "suggestion_count": N}
```
