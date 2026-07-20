# 梯度裁剪与混合精度训练配方

## 配置

```yaml
max_norm: 1.0
amp_dtype: bfloat16  # CPU 用 bfloat16, GPU 用 float16
device_type: cuda
```

## 训练步骤接线

```python
scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type, dtype=amp_dtype):
        loss = loss_fn(model(batch.inputs), batch.targets)
    if not torch.isfinite(loss).all():
        continue  # 跳过非有限损失
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if has_non_finite_grad(model.parameters()):
        scaler.update()
        continue  # 跳过非有限梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
```

## 日志 CSV 格式

`step, lr, grad_l2_pre_clip, grad_l2_post_clip, loss, skipped, skip_reason, scaler_scale`

## 告警规则

- 1,000 步滚动跳过率超过 5% → 分页告警
