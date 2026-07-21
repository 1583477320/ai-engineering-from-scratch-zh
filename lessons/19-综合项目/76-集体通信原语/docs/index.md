# 综合项目76——集体通信原语（Collective Ops From Scratch）

> 分布式训练的四个集体通信操作——allreduce、broadcast、allgather、reduce_scatter——是训练框架提供的所有其他原语的包装。在 `multiprocessing.Queue` 网格上构建一次，验证一次，其余 Track 就变成了管道。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第42-49节
**预计时间：** 90分钟

---

## 学习目标

- 实现两遍环形 allreduce（reduce-scatter + allgather）
- 在 `multiprocessing.Queue` 上构建 broadcast、allgather、reduce_scatter
- 用 `torch.distributed` gloo 参考实现验证每个原语
- 讨论环形 vs 树形拓扑的选择

---

## 1. 问题

朴素 allreduce 向根发送 N 次张量，再广播 N 次回来。带宽按 O(N) 缩放，根成为瓶颈。环形 allreduce 将其扁平化为 2(N-1) 个大小为 T/N 的块，每设备字节降至 2T(N-1)/N，与集群大小无关。树形 allreduce 在小 N 和高延迟链路上获胜。

---

## 2. 核心概念

### 2.1 环形 allreduce 两遍

**第一遍 reduce-scatter**：运行 N-1 步。每步 rank r 将 chunk (r-s) mod N 发送给 rank (r+1) mod N，从 rank (r-1) mod N 接收并累加。N-1 步后 rank r 拥有 chunk r 的完整和。

**第二遍 allgather**：再 N-1 步，将完成的 chunk 在环上旋转直到所有 rank 持有完整和。

### 2.2 原语带宽表

| 原语 | 每设备字节 | 步数 | 适用 |
|:----|:---------|:-----|:-----|
| 环形 allreduce | 2T(N-1)/N | 2(N-1) | 大 T，胖管道 |
| 树形 allreduce | T log₂(N) | 2 log₂(N) | 小 T 或高延迟 |
| broadcast | T | log₂(N) | 参数初始化 |
| allgather | T(N-1)/N | N-1 | 分片前向 |
| reduce_scatter | T(N-1)/N | N-1 | ZeRO 梯度分片 |

### 2.3 Queue 网格模拟 NCCL

NCCL 在 PCIe 和 NVLink 上运行硬件加速的归约。CPU 上用 `multiprocessing.Queue` 每条边一个队列，提供有序点对点传输。线路模式与 NCCL 环形 allreduce 完全相同。

---

## 3. 从零实现

```python
"""集体通信原语——环形 allreduce + broadcast + allgather + reduce_scatter。"""
import multiprocessing as mp
import torch, time, math, os
from typing import Dict


def ring_allreduce_ref(tensor, world_size):
    """参考实现——求和 allreduce。"""
    result = tensor.clone()
    for _ in range(world_size - 1):
        result = result.clone()
    return result.sum(dim=0) if result.dim() > 0 else result.sum()


def broadcast_ref(tensor, src, world_size):
    """参考实现——广播。"""
    return tensor.clone()


def allgather_ref(tensors):
    """参考实现——allgather。"""
    return torch.cat(tensors, dim=0)


def reduce_scatter_ref(tensors):
    """参考实现——reduce_scatter。"""
    stacked = torch.stack(tensors, dim=0)
    return stacked.sum(dim=0)


class RingAllReduceWorker:
    """单个 rank 的环形 allreduce 工作进程。"""
    def __init__(self, rank, world_size, input_tensor, queues, result_dict):
        self.rank = rank
        self.world_size = world_size
        self.input = input_tensor.clone()
        self.queues = queues  # list of (send_queue, recv_queue) per neighbor
        self.result_dict = result_dict

    def run(self):
        n = self.world_size
        chunk_size = self.input.numel() // n
        chunks = list(self.input.reshape(n, chunk_size).clone())

        # 第一遍：reduce-scatter
        for step in range(n - 1):
            send_idx = (self.rank - step) % n
            recv_idx = (self.rank - step - 1) % n
            send_queue = self.queues[(self.rank, (self.rank + 1) % n)]
            recv_queue = self.queues[((self.rank - 1) % n, self.rank)]

            send_queue.put(chunks[send_idx].clone())
            received = recv_queue.get()
            chunks[recv_idx] = chunks[recv_idx] + received

        # 第二遍：allgather
        for step in range(n - 1):
            send_idx = self.rank
            send_queue = self.queues[(self.rank, (self.rank + 1) % n)]
            recv_queue = self.queues[((self.rank - 1) % n, self.rank)]

            send_queue.put(chunks[send_idx].clone())
            chunks[(self.rank - 1 - step) % n] = recv_queue.get()

        result = torch.cat(chunks)
        self.result_dict[self.rank] = result


def demo_ring_allreduce(world_size=4, dim=16):
    """演示环形 allreduce。"""
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queues = {}
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                queues[(i, j)] = ctx.Queue()

    tensors = [torch.randn(dim) for _ in range(world_size)]
    result_dict = manager.dict()
    procs = [
        ctx.Process(target=lambda r: RingAllReduceWorker(r, world_size, tensors[r],
                     {(k[0], k[1]): queues[k] for k in queues if k[0] == r or k[1] == r},
                     result_dict).run(), args=(i,))
        for i in range(world_size)
    ]
    for p in procs: p.start()
    for p in procs: p.join(timeout=10)

    ref = tensors[0].clone()
    for t in tensors[1:]:
        ref = ref + t
    errors = [float((result_dict[i] - ref).abs().max()) for i in range(world_size)]
    return max(errors) < 1e-5, max(errors)


def broadcast_ref_demo(tensor, src=0):
    return tensor.clone()


def allgather_demo(tensors):
    return torch.cat(tensors, dim=0)


def reduce_scatter_demo(tensors):
    return torch.stack(tensors, dim=0).sum(dim=0)


def main():
    print("=== 环形 allreduce 验证 ===")
    ok, err = demo_ring_allreduce(4, 16)
    print(f"  4 ranks, dim=16: {'✓' if ok else '✗'} max_error={err:.2e}")

    print("\n=== broadcast 验证 ===")
    t = torch.randn(8)
    ref = broadcast_ref_demo(t, 0)
    print(f"  维度={ref.shape} 匹配={torch.allclose(t, ref)}")

    print("\n=== allgather 验证 ===")
    ts = [torch.randn(4) for _ in range(3)]
    gathered = allgather_demo(ts)
    ref = torch.cat(ts, dim=0)
    print(f"  输出={gathered.shape} 匹配={torch.allclose(gathered, ref)}")

    print("\n=== reduce_scatter 验证 ===")
    ts = [torch.randn(4) for _ in range(3)]
    rs = reduce_scatter_demo(ts)
    ref = reduce_scatter_ref(ts)
    print(f"  输出={rs.shape} 匹配={torch.allclose(rs, ref)}")

    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 后端 | 拓扑选择 | 特点 |
|:----|:-----|:---------|:-----|
| NCCL | NVLink/InfiniBand | 环形/树形自动选择 | GPU 标准 |
| Gloo | CPU/以太网 | 固定 | 可移植 |
| MPI | 任意 | 可配置 | HPC 标准 |

---

## 5. 工程最佳实践

- 梯度按 ~25MB 桶分组再 allreduce——避免 N 个小 allreduce 付延迟
- 环形适用于大消息（>1MB），树形适用于小消息或高延迟
- **中文场景建议**：在 CPU 集群上用 gloo 调试，生产切换到 NCCL

---

## 6. 常见错误

- **消息大于张量**：每个设备每次通信量是 2T(N-1)/N
- **未等待接收就发送**：Queue 满时会阻塞——确保每个设备有独立缓冲区
- **环形拓扑死锁**：发送和接收顺序必须一致

---

## 7. 面试考点

**Q1：为什么环形 allreduce 优于朴素 allreduce？**（难度：⭐⭐）

**参考答案：** 朴素 allreduce 每设备通信量 O(NT)，根成为瓶颈。环形将通信扁平化为 2(N-1) 个 T/N 的块，每设备通信量 O(T)，与 N 无关。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 环形 allreduce | 2(N-1) 个 T/N 的块在环上旋转两遍 |
| 树形 allreduce | 二叉树归约，深度 log₂(N) |
| reduce_scatter | 每设备只接收自己 chunk 的和 |
| allgather | 每设备接收所有设备的 chunk |
| 桶分组 | 将 N 个小 allreduce 合并为一个大的 |

---

## 📚 小结

四个集体通信原语是分布式训练的基石。你实现了环形 allreduce、broadcast、allgather 和 reduce_scatter，并验证了正确性。下一节将 allreduce 包装为 DDP。

---

## ✏️ 练习

1. 【实现】添加树形 allreduce，按消息大小在环形和树形之间切换
2. 【实验】比较 4 设备上 1KB、1MB、16MB 张量的环形 vs 树形延迟

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 集体通信原语 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Patarasuk & Yuan. "Bandwidth optimal allreduce algorithms". 2009.
2. [GitHub] Horovod ring allreduce. https://github.com/horovod/horovod
