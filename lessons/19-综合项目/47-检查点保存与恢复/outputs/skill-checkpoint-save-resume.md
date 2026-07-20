# 检查点保存与恢复配方

## 载荷结构

```python
{
    "schema": "ckpt.v1",
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "state": {"step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "losses": losses},
    "rng": {"python": ..., "numpy": ..., "torch_cpu": ..., "torch_cuda": ...},
    "wall_saved_at": timestamp,
}
```

## 原子保存

```python
tmp = tempfile.NamedTemporaryFile(delete=False, dir=target_dir, prefix="ckpt.", suffix=".tmp")
torch.save(payload, Path(tmp.name))
os.replace(Path(tmp.name), target_path)
```

## 分片检查点

- 参数键轮询分配到 N 个分片
- 每个分片写为 `model.shard-NNN.pt`
- 元文件 `meta.pt` 保存优化器 + 调度器 + 训练状态 + RNG
- `index.json` 记录分片路径和 sha256 哈希
- 加载器先验证哈希，再合并分片

## 中途恢复

- 保存 `(epoch, batch_in_epoch)` + RNG 状态
- 加载后快进 RNG 到当前批次位置，继续训练
