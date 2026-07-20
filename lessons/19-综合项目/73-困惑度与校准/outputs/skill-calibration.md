# 困惑度与校准配方

## 困惑度

```
perplexity = exp(sum(nll) / total_tokens)
```

## ECE

将置信度分桶 → 每桶平均(置信度-准确率)差距 → 按桶大小加权求和

## Brier

```
brier = mean((p_i - y_i)^2)
```

## 可靠性图

三个数组：每桶平均置信度、每桶平均准确率、每桶计数
