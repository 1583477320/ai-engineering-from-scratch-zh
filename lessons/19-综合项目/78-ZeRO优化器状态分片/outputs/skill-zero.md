# ZeRO 配方

## 三阶段

| 阶段 | 分片 | 内存 |
|:----|:-----|:-----|
| ZeRO-1 | 优化器 | 4P+12P/N |
| ZeRO-2 | 优化器+梯度 | 2P+14P/N |
| ZeRO-3 | 全部 | ~16P/N |

## 通信

Stage 1 = reduce_scatter + allgather = allreduce = DDP 通信量
