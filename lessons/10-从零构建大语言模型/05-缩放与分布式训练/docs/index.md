# 缩放与分布式训练

> 你 124M 的模型在单 GPU 上训练。现在试试 70 亿参数。模型放不进显存。数据在单机上需要几周。分布式训练在规模化时不是可选项——它是唯一的路径。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练 MiniGPT）| **时间：** ~120 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 04（预训练 MiniGPT）— 从单 GPU 到多 GPU | 阶段 10 · 11（量化）— 训练后的模型压缩

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释三种并行策略——数据并行、张量并行、流水线并行
- [ ] 计算给定模型大小的显存预算——权重+优化器+梯度+激活值
- [ ] 说明 FSDP/DeepSpeed ZeRO 如何跨 GPU 分片模型状态
- [ ] 实现 PyTorch DDP 数据并行训练
- [ ] 选择正确的并行策略——根据模型大小和 GPU 数量

---

## 1. 问题

一个 7B 参数模型用 FP16 需要 14GB 仅存储权重。Adam 优化器为每个参数存储两个额外副本（一阶和二阶矩估计）。那是另外 28GB。反向传播的梯度再加 14GB。还没算任何一个激活值，你就已经到 56GB 了。

这是 LLM 训练的核心工程挑战：**单个 GPU 装不下模型。**

解决方案：把模型拆分到多个 GPU 上。但怎么拆？每种拆法有自己的权衡。

---

## 2. 概念

### 2.1 显存预算计算

对于一个参数量为 Φ 的模型（以参数个数计），使用 Adam 优化器和混合精度训练：

| 组成部分 | 显存 |
|---------|------|
| 模型权重（FP16） | 2Φ bytes |
| 优化器状态（Adam 二阶矩） | 4Φ bytes |
| 梯度（FP16） | 2Φ bytes |
| 激活值 | ~6Φ bytes（取决于 batch size） |
| **总计** | ~12Φ bytes |

| 模型大小 | 权重 | 优化器+梯度 | 激活值 | 总显存 |
|---------|------|-----------|--------|--------|
| 124M | 250MB | 1.2GB | 1.5GB | ~3GB |
| 1.3B | 2.6GB | 10GB | 8GB | ~20GB |
| 7B | 14GB | 56GB | 42GB | ~110GB |
| 70B | 140GB | 560GB | 420GB | ~1TB |

### 2.2 三种并行策略

**数据并行（Data Parallel）**

每个 GPU 持有完整模型副本。数据被分片——每个 GPU 处理不同的 batch，然后同步梯度。最简单，但每个 GPU 必须装得下完整模型。

```
GPU 0: 完整模型 + batch_0 → 梯度_0 →┐
GPU 1: 完整模型 + batch_1 → 梯度_1 →├→ AllReduce 平均梯度
GPU 2: 完整模型 + batch_2 → 梯度_2 →┘
GPU 3: 完整模型 + batch_3 → 梯度_3 →┘
```

**张量并行（Tensor Parallel）**

将单个层的权重矩阵拆分到多个 GPU 上。每个 GPU 计算部分结果，然后 AllReduce 合并。适合单层太大的情况。

```
GPU 0: W[:, :half]   → 部分结果 →┐
GPU 1: W[:, half:]   → 部分结果 →├→ 拼接完整结果
```

**流水线并行（Pipeline Parallel）**

将模型的不同层放在不同的 GPU 上。每个 GPU 只处理部分层。数据像流水线一样流过。

```
GPU 0: 层 0-3   → 输出 → GPU 1: 层 4-7 → 输出 → GPU 2: 层 8-11
```

### 2.3 FSDP / DeepSpeed ZeRO

FSDP（PyTorch）和 ZeRO（DeepSpeed）的核心思想：**将模型状态分片存储到所有 GPU 上，需要时再聚合。**

| ZeRO Stage | 分片内容 | 每 GPU 显存 | 通信量 |
|-----------|---------|------------|--------|
| Stage 0 | 无分片（传统 DDP） | 12Φ/GPU | 低 |
| Stage 1 | 优化器状态 | 8Φ/GPU | 低 |
| Stage 2 | +梯度 | 4Φ/GPU | 中 |
| Stage 3 | +权重 | 2Φ/GPU | 高 |

**ZeRO Stage 3（FSDP）**可以将 70B 模型装进 8 个 80GB GPU——而不用任何张量并行。

### 2.4 实际选择

| 模型大小 | GPU 数量 | 推荐策略 |
|---------|---------|---------|
| < 1B | 1-2 | DDP（数据并行） |
| 1B-10B | 4-8 | FSDP / ZeRO Stage 2 |
| 10B-70B | 8-64 | ZeRO Stage 3 + 张量并行 |
| > 70B | 64+ | 3D 并行（数据+张量+流水线） |

---

## 3. 从零实现

### PyTorch DDP 数据并行

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """初始化分布式训练。"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    model = GPT(config).to(rank)
    model = DDP(model, device_ids=[rank])

    for batch in dataloader:
        input_ids = batch["input_ids"].to(rank)
        labels = batch["labels"].to(rank)
        _, loss = model(input_ids, labels)
        loss.backward()
        # DDP 自动 AllReduce 梯度
        optimizer.step()
        optimizer.zero_grad()
```

### FSDP 配置

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# 将模型分片到所有 GPU
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Stage 3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
)
```

---

## 4. 工具

### 4.1 PyTorch DDP

```python
# torchrun --nproc_per_node=4 train.py
import torch.distributed as dist
dist.init_process_group("nccl")
```

### 4.2 DeepSpeed

```json
{
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

### 4.3 FSDP2（PyTorch 2.x）

```python
# PyTorch 2.x 的 FSDP2——更简单
model = torch.distributed.fsdp.FullyShardedDataParallel(model)
```

### 4.4 工具对比

| 工具 | 适用场景 | 特点 |
|------|---------|------|
| DDP | 单节点多 GPU | 最简单，每个 GPU 完整模型 |
| FSDP | 多节点 / 大模型 | PyTorch 原生，分片模型状态 |
| DeepSpeed | 超大模型 | ZeRO Stage 3 + CPU offload |
| Megatron-LM | 超大规模 | 张量并行 + 流水线并行 |

---

## 5. LLM 视角

### 5.1 分布式训练在大语言模型中的体现

| 模型 | 参数量 | GPU 数量 | 并行策略 |
|------|--------|---------|---------|
| GPT-2 | 124M | 1 | 无 |
| Llama 2 7B | 7B | 256×A100 | DDP + FSDP |
| Llama 3 405B | 405B | 16K×H100 | 3D 并行 |
| GPT-4 | 1.8T（推测） | 25K×A100 | 3D + Expert 并行 |
| DeepSeek-V3 | 671B | 2048×H800 | MoE + 张量并行 |

### 5.2 训练 vs 推理的显存需求

训练比推理需要多得多的显存——因为需要存储梯度和优化器状态。推理只需要权重和 KV 缓存。这就是为什么量化（下一课）对推理如此重要——但对训练帮助有限。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 时，它背后的模型可能在数千个 GPU 上训练。分布式训练让这个规模成为可能——没有 FSDP/ZeRO，70B 模型需要 110GB 显存，而最大的单 GPU 只有 80GB。

---

## 6. 工程最佳实践

### 6.1 选择并行策略

```
模型能放进单 GPU？
├─ 是 → DDP（数据并行）
├─ 否，但 4-8 GPU 能装 → FSDP / ZeRO Stage 2
├─ 否，需要多节点 → ZeRO Stage 3 + 张量并行
└─ 100B+ → 3D 并行（数据+张量+流水线）
```

### 6.2 混合精度训练

- **BF16**（推荐）：与 FP32 同范围，更稳定
- **FP16**：范围小，需要梯度缩放
- **FP32 混合**：某些层用 FP32，其余用 FP16

### 6.3 踩坑经验

- **AllReduce 通信瓶颈**：多节点训练时，梯度同步可能成为瓶颈——增加 gradient accumulation 步数
- **显存 OOM**：先试减小 batch size → 启用梯度检查点 → 切换到 ZeRO Stage 3
- **训练不稳定**：BF16 比 FP16 更稳定——优先使用 BF16

---

## 7. 常见错误

### 错误 1：DDP 中忘记设置 device_ids

**现象：** 模型只在 GPU 0 上运行。

```python
# ❌ 错误
model = DDP(model)

# ✓ 正确
model = DDP(model, device_ids=[rank])
```

### 错误 2：没有在 DDP 中用 DistributedSampler

**现象：** 每个 GPU 看到相同的数据。

```python
# ❌ 错误
sampler = RandomSampler(dataset)

# ✓ 正确
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
```

### 错误 3：训练结束后忘记清理分布式

```python
# 训练结束后
dist.destroy_process_group()
```

---

## 8. 面试考点

### Q1：7B 模型在单 GPU 上能训练吗？需要什么条件？（难度：⭐⭐）

**参考答案：**
不能直接训练。7B FP16 权重 = 14GB，Adam 优化器 = 28GB，梯度 = 14GB，总计 56GB，远超单 GPU 的 80GB。解决方案：(1) 使用 ZeRO Stage 2——分片优化器和梯度，每 GPU 只需 ~20GB；(2) CPU offload——将部分状态卸载到 CPU 内存；(3) 梯度检查点——用计算换显存——前向时不保存激活值，反向时重新计算。

### Q2：FSDP 和 DDP 的核心区别是什么？（难度：⭐⭐⭐）

**参考答案：**
DDP（Distributed Data Parallel）在每个 GPU 上存储完整的模型副本——每个 GPU 需要装得下整个模型。FSDP（Fully Sharded Data Parallel）将模型的权重、梯度和优化器状态分片到所有 GPU 上——每个 GPU 只存储 1/N 的模型状态。训练时需要 AllGather 收集完整权重做前向/反向，然后重新分片。FSDP 的通信量更大（AllGather vs AllReduce），但可以训练单 GPU 装不下的模型。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 数据并行 | "每个 GPU 一份模型" | 每个 GPU 持有完整模型，数据分片，梯度 AllReduce 同步 |
| 张量并行 | "每层拆到多个 GPU" | 将单层的权重矩阵拆分到多个 GPU，各自计算部分结果 |
| 流水线并行 | "层分到不同 GPU" | 模型的不同层放在不同 GPU 上，数据像流水线一样流过 |
| FSDP | "PyTorch 的 ZeRO" | 全分片数据并行——权重、梯度、优化器状态都分片到所有 GPU |
| ZeRO | "DeepSpeed 的分片" | 零冗余优化器——Stage 1/2/3 逐步分片更多状态 |
| 梯度检查点 | "用计算换显存" | 前向时不保存激活值，反向时重新计算——显存减半但速度慢 30% |
| BF16 | "Brain Float 16" | 16 位浮点数，范围与 FP32 相同但精度低——训练更稳定 |

---

## 📚 小结

分布式训练是 LLM 规模化的关键。三种并行：数据并行（每个 GPU 一份模型）、张量并行（每层拆分）、流水线并行（层分到不同 GPU）。FSDP/ZeRO 将模型状态分片——ZeRO Stage 3 可以在 8 个 80GB GPU 上训练 70B 模型。混合精度（BF16）减少显存且加速计算。选择策略：小模型用 DDP，中等用 FSDP，超大用 3D 并行。

---

## ✏️ 练习

1. **【计算】** 计算 Llama 2 7B 训练所需的显存——权重、优化器、梯度、激活值各多少？在 8 个 A100 80GB 上用 ZeRO Stage 2 是否足够？
2. **【实验】** 用 torchrun 在 2 个 GPU 上运行 DDP 训练——对比单 GPU 训练速度。
3. **【思考】** 如果你有 64 个 A100，训练 70B 模型，你会如何分配并行策略？画出 GPU 分配图。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| DDP 训练脚本 | `code/ddp_train.py` | PyTorch DDP 多 GPU 训练示例 |
| 显存计算器 | `code/memory_calculator.py` | 计算给定模型的显存需求 |

---

## 📖 参考资料

1. [论文] Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models". SC, 2020. https://arxiv.org/abs/1910.02054
2. [官方文档] PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
3. [官方文档] DeepSpeed: https://www.deepspeed.ai/
4. [论文] Narayanan et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM". SC, 2021. https://arxiv.org/abs/2104.04473

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
