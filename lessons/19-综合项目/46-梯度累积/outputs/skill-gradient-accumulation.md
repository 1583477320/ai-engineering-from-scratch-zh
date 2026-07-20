# 梯度累积配方

## 有效批次恒等式

```
有效批次 = 微批次大小 × 累积步数 × 数据并行设备数
```

## 实现模式

```python
accum_steps = effective_batch // (micro_batch * world_size)
zero_grads(model)
for i, (x, y) in enumerate(micro_batches):
    is_last = i == accum_steps - 1
    if not is_last and use_ddp:
        with model.no_sync():
            (loss_fn(model(x), y) / accum_steps).backward()
    else:
        (loss_fn(model(x), y) / accum_steps).backward()
optimizer.step()
```

## 关键规则

- 损失必须在每个微批次上除以 `accum_steps`
- 优化器步骤只在最后一个微批次后执行一次
- 优化器状态（动量、Adam 矩）的更新频率由有效步骤决定，不是微批次
- 单设备：这是簿记。多设备：用 `no_sync` 避免 N-1 次全规约
