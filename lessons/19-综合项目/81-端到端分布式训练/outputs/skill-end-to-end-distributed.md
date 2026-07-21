# 端到端分布式训练配方

## 组合

DDP(广播+同步) + ZeRO-1(reduce_scatter+allgather) + 分片检查点

## 四不变量

1. 损失单调递减
2. 参数范数一致
3. 内存符合 ZeRO-1 公式
4. 检查点字节等价

## 生产替换

CPU gloo → NCCL；模拟多进程 → torchrun
