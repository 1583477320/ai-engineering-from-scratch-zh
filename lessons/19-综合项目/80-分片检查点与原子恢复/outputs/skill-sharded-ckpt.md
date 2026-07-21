# 分片检查点配方

## 清单字段

- world_size：防止不同设备数恢复
- sha256：捕获损坏
- offset/numel：重建扁平参数张量

## 原子写入

```
write .tmp → fsync → os.replace → atomic
```

## 故障防御

设备数变化、分片缺失、哈希不匹配 → 快速失败
