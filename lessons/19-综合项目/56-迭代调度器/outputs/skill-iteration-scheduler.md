# 迭代调度器配方

## UCB1 公式

```
ucb = mean_reward + sqrt(2) × sqrt(ln(total_runs) / runs(branch))
```

## 预算保护

- max_experiments: 总实验上限
- max_seconds: 墙上时钟上限

## 扇出条件

平均奖励 ≥ paper_threshold (0.7) → 触发论文写作
