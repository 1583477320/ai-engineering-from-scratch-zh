# DDP 配方

## 三步

1. 广播参数（初始化）
2. allreduce 梯度（反向后）
3. 优化器步进

## 关键模式

- 桶分组：~25MB/桶
- no_sync：梯度累积时跳过 allreduce
- find_unused_parameters：条件前向时开启
