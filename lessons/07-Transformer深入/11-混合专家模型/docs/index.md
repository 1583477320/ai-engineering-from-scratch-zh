# 混合专家模型（MoE）

> 参数多但每次只激活一小部分——MoE 让模型在不增加计算成本的前提下增大参数量。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）
**时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 12（KV 缓存与 FlashAttention）— 对比 MoE 与 KV 缓存两种不同的推理优化策略

---

## 🎯 学习目标

- [ ] 理解 MoE 的核心思想——多个"专家"FFN，路由器选择少数专家
- [ ] 解释 MoE 如何实现"参数多但 FLOPs 低"——计算效率的关键
- [ ] 说明 Mixtral、Qwen2.5-MoE 的路由策略——Top-K 选择

---

## 1. 问题

Transformer 的前馈网络（FFN）是参数密集的——通常占模型参数的 2/3。扩大 FFN 会线性增加计算量。MoE 的答案：**将一个大 FFN 替换为 N 个小的"专家"FFN，每次只激活 K 个。**

```
标准 FFN:  1 个大 FFN（所有参数每次都激活）
MoE:       N 个专家 FFN（每次只激活 K 个，路由器决定）
```

**效果：** N=8 个专家、K=2 → 参数量增大 8 倍，但 FLOPs 只增加 25%。

---

## 2. 概念

### 2.1 路由器

```python
def router(x, n_experts=8, top_k=2):
    """路由器：输入 x → 每个专家的得分 → 选 top-K 个专家。"""
    router_logits = x @ router_weights  # (n_experts,)
    probs = softmax(router_logits)
    top_indices = torch.topk(probs, top_k).indices  # 选 top-K
    return top_indices
```

### 2.2 MoE 层

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        # 路由器决定每个位置激活哪些专家
        router_logits = self.router(x)
        indices = torch.topk(router_logits, self.top_k, dim=-1).indices
        # 只计算被激活专家的输出——其余专家跳过
        output = sum(self.experts[k](x) for k in range(self.top_k))
        return output / self.top_k  # 平均激活
```
完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 HuggingFace MoE 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 Mixtral-8x7B
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Mixtral 8x7B：47B 总参数，12B 激活
# 效果接近 70B 稠密模型
```

### 4.2 性能对比

| 模型 | 总参数量 | 激活参数量 | 效果对标 |
|---|---|---|---|
| Mixtral 8x7B | 47B | 12B | ~70B 稠密模型 |
| Qwen2.5-MoE | 14B | 4B | ~7B 稠密模型 |
| DeepSeek-V2 | 236B | 21B | ~70B 稠密模型 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

MoE 是当前大语言模型扩大参数量的主要方式。GPT-4 被广泛认为是 MoE 架构（虽然 OpenAI 未官方确认）。Mixtral-8x7B 是最成功的开源 MoE 模型——47B 参数但只激活 12B，效果接近 70B 稠密模型。

### 5.2 LLM 时代什么变了？

**从稠密到稀疏。** 稠密模型每个参数都参与计算。MoE 只激活一部分——用更少的计算量达到更大的参数量效果。

**参数效率成为关键。** 大语言模型的推理成本与激活参数量成正比。MoE 让模型在不增加推理成本的前提下增大参数量。

### 5.3 什么没变？

**FFN 的核心作用没变。** MoE 本质上是把一个大 FFN 拆成多个小 FFN——每个专家仍然是前馈网络。

**注意力机制没变。** MoE 只改变了 FFN 层——注意力层仍然是标准的多头注意力。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 时，如果背后是 MoE 模型，每次请求只激活 2-4 个专家——而不是全部。这就是为什么 MoE 模型可以同时服务大量用户——每个请求的计算量比稠密模型小。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 开源推理 | Mixtral 8x7B | 效果好，社区支持 |
| 高质量生成 | DeepSeek-V2 | 参数效率最高 |
| 训练新模型 | 从稠密预训练→MoE 微调 | 最稳定的训练方式 |

### 6.2 中文场景特别建议

- Qwen2.5-MoE 是中文 MoE 的首选——14B 总参数，4B 激活
- Mixtral 对中文支持有限——中文场景优先选择 Qwen 或 DeepSeek

### 6.3 踩坑经验

- MoE 训练不稳定——辅助负载均衡损失是关键
- 专家死亡是常见问题——需要仔细调参
- 通信开销在分布式训练中是瓶颈

---

## 7. 常见错误

### 错误 1：专家死亡

**现象：** 某些专家从未被选中——参数量增大但效果没有提升。

**原因：** 路由器的初始权重可能偏向少数专家——一旦这些专家被频繁选中，它们会越来越好，其他专家越来越差（马太效应）。

**修复：**
```python
# ❌ 不加辅助损失
loss = cross_entropy(logits, labels)

# ✓ 加入辅助负载均衡损失
aux_loss = compute_auxiliary_loss(indices, n_experts)
loss = cross_entropy(logits, labels) + alpha * aux_loss
```

### 错误 2：负载不均衡

**现象：** 少数专家处理大部分输入——推理时无法利用并行。

**原因：** 路由器没有强制负载均衡——某些专家总是被优先选择。

**修复：**
- 使用辅助负载均衡损失
- 调整路由权重的初始化

### 错误 3：分布式训练通信开销大

**现象：** MoE 模型在多 GPU 上训练速度很慢。

**原因：** 专家可能分布在不同 GPU 上——每个 token 需要路由到对应 GPU 的专家——通信成本高。

---

## 8. 面试考点

### Q1：MoE 如何实现"参数多但 FLOPs 低"？（难度：⭐⭐）

**参考答案：**
MoE 将一个大 FFN 替换为 N 个小的"专家"FFN，每次只激活 K 个。总参数量是 N 个小 FFN 的参数之和，但每次推理只使用 K 个专家——计算量与单个大 FFN 接近。

### Q2：路由器是如何工作的？（难度：⭐⭐）

**参考答案：**
路由器是一个线性层——输入 token 的表示，输出每个专家的得分。用 softmax 归一化后选 top-K 个专家。被选中的专家的输出按权重加权求和。

### Q3：什么是专家死亡？如何解决？（难度：⭐⭐⭐）

**参考答案：**
专家死亡是指某些专家从未被选中——参数虽然存在但从未参与计算。解决方案：1）辅助负载均衡损失——鼓励均匀路由；2）随机路由——在 top-K 之外随机选一个专家；3）定期重初始化未使用的专家。

### Q4：Mixtral 8x7B 的参数量和计算量分别是多少？（难度：⭐⭐）

**参考答案：**
Mixtral 8x7B：总参数量 47B，激活参数量 12B（每次 8 个专家中选 2 个）。效果接近 70B 稠密模型——参数效率是 70B/12B ≈ 5.8x。

### Q5：MoE 的主要挑战是什么？（难度：⭐⭐⭐）

**参考答案：**
1. 负载均衡：确保专家被均匀使用
2. 通信开销：分布式训练中专家跨 GPU 通信
3. 训练稳定性：路由器的训练比稠密模型更不稳定
4. 推理优化：需要支持稀疏计算的推理框架

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 专家 (Expert) | "小 FFN" | N 个独立的小 FFN——每个有不同的参数，处理不同的输入子集 |
| 路由器 (Router) | "选择器" | 决定每个输入激活哪些专家的可学习门控 |
| 稀疏激活 | "只用一部分" | 每次只激活 K 个专家（K << N）——计算效率的关键 |
| 专家死亡 | "不用的专家" | 某些专家从未被选中——参数浪费 |
| 负载均衡 | "均匀使用" | 鼓励路由器均匀地分配输入到各专家 |
| 辅助损失 | "鼓励均匀的损失" | 额外的损失项——惩罚负载不均衡 |
| 稠密模型 | "标准 Transformer" | 所有参数每次都激活——与 MoE 相对 |
| Top-K 路由 | "选最好的 K 个" | 路由器选得分最高的 K 个专家 |

---

## 📚 小结

MoE 的核心：参数多但激活少。8 个专家、每次激活 2 个 → 参数量 8x，计算量仅 1.25x。Mixtral 8x7B（47B 参数，12B 激活）用 1.2B FLOPs 达到了 70B 模型的性能。2026 年混合 SSM+MoE 是前沿实验室的主流架构。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 MoE 如何实现"参数多但计算量小"。写 200 字以内的说明。

2. **【实现】** 实现一个 4 专家、Top-2 路由的 MoE 层，在 MNIST 上对比有/无 MoE 的参数量和准确率。

3. **【实现】** 画出路由器权重矩阵——每个位置激活了哪些专家？是否存在"专家死亡"（某些专家从未被选中）？

4. **【实验】** 实现辅助负载均衡损失——对比有/无辅助损失时的专家使用分布。

5. **【思考】** 阅读 Switch Transformer 论文的摘要，用你自己的话解释为什么稀疏激活能提升模型效率。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| MoE 完整实现 | `code/main.py` | 路由器、专家 FFN、负载均衡损失 |
| MoE 模型对比指南 | `outputs/moe-comparison.md` | 参数量和计算效率对比 |

---

## 📖 参考资料

1. [论文] Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". 2022.
2. [模型] Mistral-7B / Mixtral-8x7B. https://github.com/mistralai/mistral-src
3. [论文] Lepikhin et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding". 2021.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
