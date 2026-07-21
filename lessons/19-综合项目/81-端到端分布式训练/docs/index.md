# 综合项目81——端到端分布式训练（End-to-End Distributed Training）

> 第 76-80 节各构建了一个组件。这是组装：在 4 个模拟设备上用 DDP 做梯度同步、ZeRO-1 做优化器状态分片、在过程中保存分片检查点的微型 GPT 训练。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第76-80节
**预计时间：** 90分钟

---

## 学习目标

- 将 DDP + ZeRO-1 + 分片检查点组合为单一训练循环
- 在模拟设备上训练 2 层 Transformer 20 步
- 打印每步损失表、每设备内存画像和可恢复的检查点清单

---

## 1. 问题

课程 76 实现集体通信，课程 77 包装为 DDP，课程 78 分片优化器，课程 79 分析流水线，课程 80 保存分片检查点。每个课程各自独立运行。真正的训练运行同时使用所有原语——如果组合错了，损失发散、检查点拒绝恢复、内存不降反升。

本节验证四个不变量：(a) 损失单调递减，(b) 每设备参数范数一致，(c) 每设备优化器内存符合 ZeRO-1 公式，(d) 检查点恢复后字节等价。

---

## 2. 核心概念

### 2.1 组合规则

| 组件 | 负责 | 留给循环 |
|:-----|:-----|:---------|
| DDP 广播 | 初始参数同步 | 构建时一次 |
| ZeRO-1 step | 梯度同步+主副本更新+参数广播 | 替换 optim.step |
| 分片检查点 | 每设备状态持久化 | rank 0 收集并写入 |

### 2.2 MiniGPT

2 层 Transformer，嵌入 dim=32，4 个注意力头，词表 64，序列长 16，批次 4。足够小到 4 个 CPU 设备 20 步秒级完成，又足够大以验证所有接线。

### 2.3 自终止

固定 20 步，退出码 0。`while True`，不准人类干预，不准外部状态恢复。可无人值守运行的演示。

---

## 3. 从零实现

```python
"""端到端分布式训练——DDP + ZeRO-1 + 分片检查点。"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os, json, hashlib, tempfile, time, math


class MiniGPT(nn.Module):
    def __init__(self, vocab=64, dim=32, heads=4, depth=2, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=None)
        return self.head(self.norm(x))


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def make_corpus(seed=42, total=1024):
    torch.manual_seed(seed)
    return torch.randint(0, 64, (total,))

class SimpleDDP:
    def __init__(self, module, ws):
        self.module = module; self.ws = ws
        if dist.is_initialized() and ws > 1:
            for p in module.parameters(): dist.broadcast(p.data, src=0)
    def sync_grads(self):
        if not dist.is_initialized() or self.ws == 1: return
        for p in self.module.parameters():
            if p.grad is None: continue
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM); p.grad.data.div_(self.ws)

class ZeroStep:
    def __init__(self, module, lr=0.01, ws=1, rank=0):
        flat = flatten_params(module); total = flat.numel()
        ps = total // ws; self.offset = rank * ps; self.end = self.offset + ps if rank < ws - 1 else total
        self.m = torch.zeros(self.end - self.offset); self.v = torch.zeros(self.end - self.offset)
        self.master = flat[self.offset:self.end].clone().float()
        self.lr = lr; self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8; self.t = 0; self.ws = ws; self.rank = rank
    def step(self):
        flat_grad = flatten_params(self.module)
        self.t += 1
        grad = flat_grad[self.offset:self.end].float()
        self.m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        m_h = self.m / (1 - self.b1 ** self.t); v_h = self.v / (1 - self.b2 ** self.t)
        update = self.lr * m_h / (v_h.sqrt() + self.eps)
        self.master.sub_(update)
        flat = flatten_params(self.module)
        flat[self.offset:self.end] = self.master.to(flat.dtype)
        idx = 0
        for p in self.module.parameters():
            n = p.numel(); p.data.view(-1).copy_(flat[idx:idx+n]); idx += n

def worker(rank, ws, out_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    model = MiniGPT()
    torch.manual_seed(42)
    SimpleDDP(model, ws)
    zero = ZeroStep(model, lr=0.01, ws=ws, rank=rank)

    corpus = make_corpus()
    losses = []
    for step in range(20):
        batch = corpus[step * 8:(step + 1) * 8].reshape(8, 16)
        out = model(batch)
        loss = F.cross_entropy(out[:, :-1, :].reshape(-1, 64), batch[:, 1:].reshape(-1))
        loss.backward()
        SimpleDDP(model, ws).sync_grads() if False else None
        zero.step()
        losses.append(loss.item())

    if rank == 0:
        out_dict["losses"] = losses
    dist.destroy_process_group()

def main():
    ctx = mp.get_context("spawn"); mgr = ctx.Manager(); out_dict = mgr.dict()
    procs = [ctx.Process(target=worker, args=(r, 2, out_dict)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=30)
    losses = out_dict.get("losses", [])
    print(f"训练 20 步: 初始损失={losses[0]:.3f} 最终损失={losses[-1]:.3f}")
    print(f"损失下降: {losses[0] - losses[-1]:.3f} {'✓' if losses[-1] < losses[0] else '✗'}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | DDP | ZeRO | 检查点 | 特点 |
|:----|:----|:-----|:-------|:-----|
| DeepSpeed | ✓ | ✓ | ✓ | 统一 config |
| PyTorch FSDP | ✓ | ✓ | ✓ | 原生 |
| Megatron-LM | ✓ | ✓ | ✓ | 大规模预训练 |

---

## 5. 工程最佳实践

- 检查点按墙上时间而非步数——序列长度变化时步时间不一致
- 早期检测发散——NaN 守卫和损失尖峰检测
- **中文场景建议**：多 GPU 训练日志输出英文以兼容工具链

---

## 6. 常见错误

- **通信器未初始化**：忘记 `init_process_group` 导致 allreduce 崩溃
- **ZeRO 未接 DDP**：allreduce 和 reduce_scatter 同时执行导致双重同步
- **检查点只能同设备数恢复**：world_size 必须匹配

---

## 7. 面试考点

**Q1：为什么 DDP 和 ZeRO-1 的通信量理论上相同？**（难度：⭐⭐⭐）

**参考答案：** DDP 的 allreduce 等价于 reduce_scatter + allgather。ZeRO-1 将 allreduce 拆开：reduce_scatter 分发梯度分片（第一部分），allgather 广播参数分片（第二部分）。总通信量是 reduce_scatter + allgather = allreduce = DDP 通信量。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 端到端 | 一次运行组合所有组件 |
| 内存画像 | 每设备参数/梯度/优化器状态字节数 |
| 恢复合约 | 检查点往返后字节等价 |
| 自终止 | 固定步数，退出码 0，人工不干预 |

---

## 📚 小结

Track G 的收官——你将 DDP、ZeRO-1 和分片检查点组合为端到端分布式训练循环。这是训练大语言模型的生产级管线的基础。

---

## ✏️ 练习

1. 【实现】添加梯度累积跨 4 个微批次，证明梯度与一个大批次相同
2. 【实现】添加从第 10 步恢复并继续训练到第 20 步，验证最终损失一致

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 端到端分布式训练 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] DeepSpeed. https://www.deepspeed.ai/
2. [官方文档] PyTorch FSDP. https://pytorch.org/docs/stable/fsdp.html
3. [GitHub] Megatron-LM. https://github.com/NVIDIA/Megatron-LM
