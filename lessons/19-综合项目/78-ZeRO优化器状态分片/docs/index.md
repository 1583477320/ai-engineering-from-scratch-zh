# 综合项目78——ZeRO 优化器状态分片（ZeRO Optimizer State Sharding）

> Adam 每参数存储两个矩估计。7B 参数模型有 56GB 优化器状态。ZeRO Stage 1 将其分片到 N 个设备；每设备持有 1/N。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第76节
**预计时间：** 90分钟

---

## 学习目标

- 分片优化器状态（一阶矩、二阶矩、fp32 主副本）到 N 个设备
- 使用 reduce_scatter 传递每个设备的梯度和，allgather 广播更新后的参数
- 计算 Stage 1/2/3 的内存节省表
- 解释 Stage 1 与 Stage 3 的选择

---

## 1. 问题

朴素 DDP 复制一切：参数、梯度和优化器状态全在每个设备上完整存在。对于 fp16 的 7B 模型，每个设备有 14GB 参数 + 14GB 梯度 + 28GB 优化器状态。优化器状态是最大的项，且最容易分片——它只在步进时使用，前向和反向传播中不用。

ZeRO Stage 1 分片优化器状态。每设备持有 1/N 的 Adam 矩。反向传播后，用 reduce_scatter 传递每设备的分片梯度和，用 allgather 广播更新后的参数分片。

---

## 2. 核心概念

### 2.1 ZeRO 各阶段

| 阶段 | 分片什么 | 每设备内存 | 每步通信 |
|:-----|:--------|:---------|:---------|
| DDP | 无 | 参数+梯度+优化器 | 1× allreduce |
| ZeRO-1 | 优化器状态 | 参数+梯度+优化器/N | 1× reduce_scatter + 1× allgather |
| ZeRO-2 | 优化器+梯度 | 参数+梯度/N+优化器/N | 1× reduce_scatter + 1× allgather |
| ZeRO-3 | 全部 | 参数/N+梯度/N+优化器/N | 1× allgather/层 + 1× reduce_scatter/层 |

### 2.2 内存数学

| 项 | 朴素 | ZeRO-1 | 原因 |
|:--|:-----|:-------|:-----|
| fp16 参数 | 2P | 2P | 前向需要 |
| fp16 梯度 | 2P | 2P | 反向需要 |
| fp32 主副本 | 4P | 4P/N | 优化器使用 |
| fp32 一阶矩 | 4P | 4P/N | 优化器使用 |
| fp32 二阶矩 | 4P | 4P/N | 优化器使用 |
| **总计** | **16P** | **4P + 12P/N** | |

N=8: 16P → 5.5P，节省 65%。

### 2.3 为什么 reduce_scatter 优于 allreduce

allreduce 给每设备完整和。如果只需要分片 r，那 (N-1)/N 的归约在 r 上是浪费的。reduce_scatter 精确传递每设备拥有的分片；通信量与 allreduce 相同，但第二半被后续的参数 allgather 替换。

---

## 3. 从零实现

```python
"""ZeRO Stage 1 优化器状态分片。"""
import torch, torch.nn as nn, math


class ZeroOptimizer:
    def __init__(self, params, lr=0.01, world_size=1, rank=0):
        flat = torch.cat([p.data.view(-1) for p in params])
        total = flat.numel()
        per_shard = total // world_size
        self.offset = rank * per_shard
        self.end = self.offset + per_shard if rank < world_size - 1 else total
        self.shard_size = self.end - self.offset

        self.master = flat[self.offset:self.end].clone().float()
        self.m = torch.zeros(self.shard_size)
        self.v = torch.zeros(self.shard_size)
        self.lr = lr
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0

    def step(self, flat_grad):
        grad = flat_grad[self.offset:self.end].float()
        self.t += 1
        self.m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = self.lr * m_hat / (v_hat.sqrt() + self.eps)
        self.master.sub_(update)
        return self.master.clone()


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def demo():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    world_size = 2
    optimizers = [ZeroOptimizer(model.parameters(), lr=0.01, world_size=world_size, rank=r) for r in range(world_size)]

    for step in range(10):
        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        for rank in range(world_size):
            out = model(x) + rank * 0.0001
            loss = F.mse_loss(out, y)
            loss.backward()
            flat_grad = flatten_params(model)
            shard = optimizers[rank].step(flat_grad)
    print(f"ZeRO-1: 每设备持有 {optimizers[0].shard_size}/{32*64+64*32} 参数优化器状态")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 阶段 | 特点 |
|:----|:-----|:-----|
| DeepSpeed ZeRO | 1/2/3 | 参考实现 |
| PyTorch FSDP | 2/3 | PyTorch 原生 |
| HuggingFace Accelerate | 全部 | 统一配置 |

---

## 5. 工程最佳实践

- Stage 1 是近乎免费的胜利——通信与 DDP 相同，内存线性减少
- 混合精度是核心——不用混合精度就浪费了 fp32 主副本的分片
- **中文场景建议**：ZeRO-1 配置简单，推荐作为默认起点

---

## 6. 常见错误

- **未更新所有设备的优化器状态**：Adam 矩必须在分片上步进
- **reduce_scatter 后忘记 allgather**：参数不一致导致下一步损失异常
- **Stage 3 未逐层 allgather**：前向传播需要完整层参数

---

## 7. 面试考点

**Q1：为什么 ZeRO Stage 1 的通信量与 DDP 相同？**（难度：⭐⭐⭐）

**参考答案：** DDP 的 allreduce 等价于 reduce_scatter + allgather。ZeRO-1 将 allreduce 拆为 reduce_scatter（梯度分片和）+ allgather（更新参数分片广播）。总通信量 = 一次 reduce_scatter + 一次 allgather = 一次 allreduce = DDP 通信量。内存减少，通信不变。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| ZeRO-1 | 分片优化器状态 |
| 主副本 | 优化器更新的 fp32 参数 |
| reduce_scatter | 每设备只接收自己分片的梯度和 |
| allgather | 广播更新后的参数分片给所有设备 |

---

## 📚 小结

ZeRO 通过分片优化器状态将内存从 16P 降至 4P+12P/N。你实现了 Stage 1 的核心循环。下一节构建流水线并行。

---

## ✏️ 练习

1. 【实验】计算并打印 ZeRO-1 vs DDP 的内存表，验证公式
2. 【实现】扩展到 ZeRO-2：分片梯度，reduce_scatter 后归零非分片部分

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| ZeRO Stage 1 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models". SC 2020. https://arxiv.org/abs/1910.02054
2. [官方文档] DeepSpeed ZeRO. https://www.deepspeed.ai/tutorials/zero/
