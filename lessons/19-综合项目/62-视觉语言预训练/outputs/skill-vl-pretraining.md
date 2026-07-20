# 视觉语言预训练配方

## 损失组合

```
total = contrastive + lm_weight × lm_loss
```

## InfoNCE

双向交叉熵：`S = I T^T / tau`
对角线为正样本，非对角为负样本

## 温度 tau

使用 `log_tau` 参数 + `exp()` 确保正数
初始化: `tau = 0.07`
