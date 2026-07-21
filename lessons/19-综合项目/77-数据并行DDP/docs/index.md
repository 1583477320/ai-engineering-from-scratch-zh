# 综合项目77——数据并行 DDP（Data Parallel DDP From Scratch）

> DistributedDataParallel 是 allreduce 上的钩子。包装模型，从 rank 0 广播初始参数，安装反向传播钩子在每次参数梯度上发起 allreduce，其余就是梯度下降。整个模式 200 行。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第76节
**预计时间：** 90分钟

---

## 学习目标

- 构建广播初始参数、反向传播后 allreduce 梯度的 DDP 封装
- 用 `torch.multiprocessing.spawn` 在 gloo 后端启动 N 个 CPU 设备
- 通过在相同数据上顺序训练证明参数等价性
- 讨论桶分组和通信重叠作为生产 DDP 的两个关键改进

---

## 1. 问题

十亿参数模型在单 GPU 上训练需要数周。数据并行将批次分割到 N 个设备，每个设备在自己的切片上计算前向和反向传播。没有梯度同步，N 个副本在第 2 步就分歧——模型不再是"在更多数据上训练的一个模型"，而是碰巧共享初始权重的 N 个独立模型。

DDP 的工艺是让梯度同步相对于计算几乎免费。PyTorch DDP 通过桶分组、反向传播与 allreduce 重叠、NVLink 上使用 NCCL 来实现。我们在 CPU gloo 上实现同样的三个改进。

---

## 2. 核心概念

### 2.1 DDP 需要的三个操作

| 阶段 | 操作 | 原因 |
|:-----|:-----|:-----|
| 初始化 | 从 rank 0 广播参数 | 每设备以相同参数开始 |
| 反向传播后 | 每个梯度桶 allreduce | 优化器在平均梯度上步进 |
| 有时 | 广播缓冲区 | BatchNorm 运行统计保持同步 |

### 2.2 为什么用平均而非求和

allreduce-SUM 除以 world_size 得到平均梯度。平均值不随 world_size 变化——在一个设备上调好的学习率在四个设备上也适用。不除以 world_size 会在每次改变集群大小时重新调学习率。

### 2.3 桶分组

一个 transformer 有数千个参数张量。每个张量一次 allreduce 付 gloo 延迟基线数千次。DDP 将梯度分组为 ~25MB 桶，每个桶一次 allreduce。延迟被桶分摊。

### 2.4 种子模式

每个设备调用 `torch.manual_seed(seed + rank)` 做打乱，但 `torch.manual_seed(seed)` 做参数初始化。统一种子意味着相同批次顺序（破坏数据并行），设备特定种子意味着初始化偏差。

---

## 3. 从零实现

```python
"""数据并行 DDP——广播+allreduce+桶。"""
import torch, torch.nn as nn, torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp


class MiniMLP(nn.Module):
    def __init__(self, dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class SimpleDDP(nn.Module):
    def __init__(self, module, world_size):
        super().__init__()
        self.module = module
        self.world_size = world_size
        if dist.is_initialized() and world_size > 1:
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        if not dist.is_initialized() or self.world_size == 1:
            return
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data.div_(self.world_size)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    return loss.item()


def worker(rank, world_size, seed):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    model = MiniMLP()
    ddp = SimpleDDP(model, world_size)
    opt = optim.SGD(ddp.parameters(), lr=0.01)

    torch.manual_seed(seed + rank)
    for step in range(20):
        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        loss = train_step(ddp, opt, x, y)
        ddp.sync_grads()
        if rank == 0 and step % 5 == 0:
            print(f"  step {step}: loss={loss:.6f}")

    dist.destroy_process_group()


def main():
    import os
    world_size = 2
    mp.spawn(worker, args=(world_size, 42), nprocs=world_size, join=True)
    print("✓ DDP 训练完成")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 通信 | 特点 |
|:----|:-----|:-----|
| PyTorch DDP | NCCL/gloo | 标准实现 |
| HuggingFace Accelerate | torchrun | 启动器简化 |
| Megatron-LM | NCCL | 张量+数据并行 |

---

## 5. 工程最佳实践

- **find_unused_parameters=True**：条件跳过参数时必须设置
- **static_graph=True**：前向稳定时预计算桶调度，节省每步几毫秒
- **no_sync 上下文**：梯度累积时跳过非最终微批次的 allreduce
- **中文场景建议**：CPU gloo 调试，生产用 `torchrun` + NCCL

---

## 6. 常见错误

- **桶分组遗漏**：每个张量一次 allreduce 会阻塞网络
- **no_sync 上下文未使用**：忘记上下文管理器导致 K 次无意义 allreduce
- **未在所有设备上初始化相同的参数**：每个设备用相同种子初始化

---

## 7. 面试考点

**Q1：DDP 的 allreduce 为什么用平均而非求和？**（难度：⭐⭐）

**参考答案：** 平均梯度不随设备数变化——在 1 设备上调好的学习率在 4 设备上也有效。求和会导致每步梯度放大 N 倍，每次改变集群大小都需要重新调学习率。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| DDP | 广播参数 + allreduce 梯度的封装 |
| 桶分组 | 将 N 个小 allreduce 合并为一个大的 |
| 通信重叠 | 在后一层计算时发起前一层的 allreduce |
| no_sync | 梯度累积时跳过反向传播后的 allreduce |

---

## 📚 小结

DDP 是分布式训练中最简单的并行策略——广播参数，allreduce 梯度，优化器步进。你实现了完整循环并证明了参数等价性。下一节将梯度 allreduce 替换为 reduce_scatter 实现 ZeRO。

---

## ✏️ 练习

1. 【实验】添加可配置梯度桶大小，测量每参数一次 allreduce vs 桶分组的加速比
2. 【实现】实现 `no_sync()` 上下文管理器，验证 K 微批次累积与单进程基线匹配

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| DDP 实现 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Li et al. "PyTorch Distributed". 2020. https://arxiv.org/abs/2006.15704
2. [官方文档] PyTorch DDP. https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
