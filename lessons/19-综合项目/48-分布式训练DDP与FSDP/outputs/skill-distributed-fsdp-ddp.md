# 分布式训练 DDP 与 FSDP 配方

## 进程组初始化

```python
import torch.distributed as dist
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
```

生产环境使用 `torchrun` + `nccl` 后端。

## DDP 核心实现

### 构建时广播参数
```python
for tensor in list(module.parameters()) + list(module.buffers()):
    dist.broadcast(tensor.data, src=0)
```

### 反向传播后全规约梯度
```python
for p in module.parameters():
    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
    p.grad.data.div_(world_size)
```

## FSDP 参数分片

- 每个设备持有参数的 1/N 切片
- 前向时 `all_gather` 重建完整张量，使用后释放
- 每层前向传播的 all_gather 可与前一层的计算重叠

## 选择策略

| 场景 | 推荐 |
|------|------|
| 模型可放入单 GPU，要加速 | DDP (`DistributedDataParallel`) |
| 模型无法放入单 GPU | FSDP (`FullyShardedDataParallel`) |
| 多 GPU 且模型较大 | FSDP + 计算通信重叠 |
