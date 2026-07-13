# 混合专家模型（MoE）

> 参数多但每次只激活一小部分——MoE 让模型在不增加计算成本的前提下增大参数量。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 07 · 05（完整 Transformer）| **时间：** ~45 分钟

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

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 专家 (Expert) | N 个小的 FFN——每个有不同的参数，处理不同的输入子集 |
| 路由器 (Router) | 决定每个输入激活哪些专家的可学习门控 |
| 稀疏激活 | 每次只激活 K 个专家（K << N）——计算效率的关键 |

---

## 📚 小结

MoE 的核心：参数多但激活少。8 个专家、每次激活 2 个 → 参数量 8x，计算量仅 1.25x。Mixtral 8x7B（47B 参数，12B 激活）用 1.2B FLOPs 达到了 70B 模型的性能。2026 年混合 SSM+MoE 是前沿实验室的主流架构。

---

## ✏️ 练习

1. 实现一个 4 专家、Top-2 路由的 MoE 层，在 MNIST 上对比有/无 MoE 的参数量和准确率
2. 画出路由器权重矩阵——每个位置激活了哪些专家？是否存在"专家死亡"（某些专家从未被选中）？

---

## 📖 参考资料

1. [论文] Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". 2022.
2. [模型] Mistral-7B / Mixtral-8x7B. https://github.com/mistralai/mistral-src

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
