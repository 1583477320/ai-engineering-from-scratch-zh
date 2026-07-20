# 综合项目40——直接偏好优化（DPO 从零实现）

> 奖励模型和 PPO 是经典 RLHF 栈。DPO 将该栈折叠为单一监督损失，直接在偏好对上拟合策略。本课程从奖励差分恒等式推导 DPO 损失，实现参考模型+策略模型对，计算序列级对数概率，并在偏好数据集上训练小型 Transformer。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将 DPO 损失推导为缩放对数比率差上的 sigmoid
- 构建冻结参考模型+可训练策略模型对
- 计算两个模型下的序列级对数概率，掩码提示词词元
- 在偏好三元组上训练策略并观察 chosen 对数概率相对 rejected 上升
- 用损失数学、梯度符号和参考不变性测试固定行为

---

## 1. 问题

你有一个 SFT（监督微调）模型。它遵循指令，但输出参差不齐。你还有一小部分偏好对：同一提示词，人工标记一个补全为 chosen（优选），另一个为 rejected（劣选）。

经典 RLHF 答案是两阶段管道：先训练奖励模型，再用 PPO 优化策略。DPO 将两阶段替换为单一监督损失。奖励模型从不显式存在。

关键洞察：对数比率差（策略与参考的对数概率之差）可以直接用于最大化偏好的似然，无需先训练奖励模型。

---

## 2. 核心概念

### 2.1 DPO 损失推导

从 Bradley-Terry 偏好模型出发：

$$
P(y_w > y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

其中 $\sigma$ 是 sigmoid 函数，$r$ 是奖励函数。

最优策略的闭式解使对数比率差与奖励差成正比。取负对数似然：

$$
L_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right)\right]
$$

无需单独的奖励模型。KL 约束嵌入闭式推导中。

### 2.2 梯度符号

对 $\log \pi_\theta(y_w \mid x)$ 的梯度为负（增加 chosen 概率降低损失）。对 $\log \pi_\theta(y_l \mid x)$ 的梯度为正（增加 rejected 概率增加损失）。训练推高 chosen，压低 rejected。

### 2.3 参考不变性

参考模型（SFT 冻结）必须满足三个不变性：

1. **参数永远不接收梯度** — 对参考模型的 `.backward()` 调用不会改变任何参数
2. **对数概率在轮次间不变** — 同一个输入在不同轮次产生完全相同的对数概率
3. **策略从相同权重初始化** — 策略模型和参考模型初始权重相同，确保对数比率差从零开始

### 2.4 偏好数据格式

每条偏好数据包含三个字段：

```json
{
  "prompt": "2 + 3 等于多少？",
  "chosen": "5",
  "rejected": "2 加 3 大约等于 5，我不太确定"
}
```

chosen 是更理想的回答——简洁、正确、自信。rejected 是不够理想的回答——冗长、犹豫、包含错误。

---

## 3. 从零实现

```python
"""DPO 从零实现——偏好对直接优化。"""
import sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionTokenizer:
    """简单的字节级词元化器。"""
    INST = 256; RESP = 257; VOCAB = 260

    def encode_prompt(self, p):
        return [self.INST] + list(p.encode("utf-8", errors="ignore")) + [self.RESP]

    def encode_completion(self, c):
        return list(c.encode("utf-8", errors="ignore"))


class CausalSelfAttn(nn.Module):
    def __init__(self, D, H, L):
        super().__init__()
        self.H = H; self.dh = D // H
        self.qkv = nn.Linear(D, D * 3, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(L, L, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        return self.out((F.softmax(att, -1) @ v).transpose(1, 2).contiguous().view(B, T, D))


class Block(nn.Module):
    def __init__(self, D, H, L):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.attn = CausalSelfAttn(D, H, L)
        self.ln2 = nn.LayerNorm(D); self.fc1 = nn.Linear(D, D * 4); self.fc2 = nn.Linear(D * 4, D)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)))) + x


class TinyGPT(nn.Module):
    def __init__(self, V, D, H, depth, L):
        super().__init__()
        self.tok = nn.Embedding(V, D); self.pos = nn.Embedding(L, D)
        self.blocks = nn.ModuleList([Block(D, H, L) for _ in range(depth)])
        self.ln = nn.LayerNorm(D); self.head = nn.Linear(D, V, bias=False); self.L = L

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def seq_log_prob(model, tok, inst, comp):
    """计算模型在给定（提示词，补全）上的序列级对数概率。"""
    full = list(inst) + list(comp)
    if len(full) > model.L:
        full = full[-model.L:]
        pl = max(0, len(full) - len(comp))
    else:
        pl = len(inst)
    ids = torch.tensor([full])
    logits = model(ids)
    lp = F.log_softmax(logits, -1)
    targets = torch.tensor(full[pl:])
    pos = torch.arange(pl, len(full) - 1)
    return lp[0, pos, targets].sum()


def dpo_loss(lp_w_pol, lp_l_pol, lp_w_ref, lp_l_ref, beta):
    """DPO 损失——负 sigmoid。返回 (损失, 对数比率差)。"""
    margin = (lp_w_pol - lp_w_ref) - (lp_l_pol - lp_l_ref)
    return -F.logsigmoid(beta * margin), margin


def make_prefs():
    """12 条偏好对数据。"""
    return [
        {"prompt": "法国的首都是什么？", "chosen": "巴黎。", "rejected": "法国是欧洲的一个国家，有很多城市，包括巴黎。"},
        {"prompt": "日本的首都是什么？", "chosen": "东京。", "rejected": "日本是一个岛国，政府在东京。"},
        {"prompt": "计算 2 + 3。", "chosen": "5。", "rejected": "2 加 3 接近 5，我认为。"},
        {"prompt": "计算 7 * 6。", "chosen": "42。", "rejected": "7 乘以 6 大约等于 42。"},
        {"prompt": "计算 12 / 4。", "chosen": "3。", "rejected": "12 除以 4 大约等于 3。"},
        {"prompt": "列举三种颜色。", "chosen": "红色、绿色、蓝色。", "rejected": "颜色包括红色、绿色和蓝色。"},
        {"prompt": "列举三个元音字母。", "chosen": "a, e, i。", "rejected": "元音字母发出开放的嘴形音。"},
        {"prompt": "什么是变量？", "chosen": "绑定到值的名称。", "rejected": "变量是用来存储东西的东西。"},
        {"prompt": "什么是函数？", "chosen": "一个返回输出的可复用代码块。", "rejected": "函数是处理输入的东西。"},
        {"prompt": "Python：打印 42。", "chosen": "print(42)", "rejected": "你可以在 python 中打印数字。"},
        {"prompt": "Python：排序列表。", "chosen": "items.sort()", "rejected": "在 python 中排序列表很容易。"},
        {"prompt": "Python：获取长度。", "chosen": "len(items)", "rejected": "要获取长度，在 items 上调用 len。"},
    ]


def build_models(cfg):
    """构建策略模型和参考模型（相同初始化，参考冻结）。"""
    torch.manual_seed(cfg.seed)
    ref = TinyGPT(cfg.V, cfg.D, cfg.H, cfg.depth, cfg.L)
    torch.manual_seed(cfg.seed)
    pol = TinyGPT(cfg.V, cfg.D, cfg.H, cfg.depth, cfg.L)
    pol.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return ref, pol


def warmup(model, triples, cfg):
    """预训练模型以建立基本语言能力。"""
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_epochs):
        for t in triples:
            ids = torch.tensor([cfg.tok.encode_prompt(t["prompt"]) + cfg.tok.encode_completion(t["chosen"])])
            ids = ids[:, :model.L]
            logits = model(ids)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, cfg.V), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()


def train_dpo(pol, ref, tok, triples, cfg):
    """DPO 训练循环。"""
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.dpo_lr)
    # 冻结参考的对数概率（计算一次，不在轮次间变化）
    ref_lps = []
    for t in triples:
        prompt = tok.encode_prompt(t["prompt"])
        lw_ref = seq_log_prob(ref, tok, prompt, tok.encode_completion(t["chosen"])).detach()
        ll_ref = seq_log_prob(ref, tok, prompt, tok.encode_completion(t["rejected"])).detach()
        ref_lps.append((lw_ref, ll_ref))

    report = {"losses": [], "margins": []}
    for ep in range(cfg.dpo_epochs):
        tl = 0; tm = 0
        for t, (lw_ref, ll_ref) in zip(triples, ref_lps):
            prompt = tok.encode_prompt(t["prompt"])
            lw_pol = seq_log_prob(pol, tok, prompt, tok.encode_completion(t["chosen"]))
            ll_pol = seq_log_prob(pol, tok, prompt, tok.encode_completion(t["rejected"]))
            loss, margin = dpo_loss(lw_pol, ll_pol, lw_ref, ll_ref, cfg.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tm += margin.item()
        report["losses"].append(tl / len(triples))
        report["margins"].append(tm / len(triples))
    return report


def main():
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        V: int = 260; D: int = 64; H: int = 4; depth: int = 2; L: int = 96
        beta: float = 0.2; dpo_lr: float = 1e-3; dpo_epochs: int = 30
        warmup_lr: float = 3e-3; warmup_epochs: int = 8
        seed: int = 0

    cfg = Cfg()
    cfg.tok = InstructionTokenizer()
    torch.manual_seed(0)
    triples = make_prefs()

    print("[预训练] 建立基本语言能力...")
    ref, pol = build_models(cfg)
    warmup(ref, triples, cfg)
    pol.load_state_dict(ref.state_dict())

    print("[DPO 训练] 在偏好对上训练...")
    report = train_dpo(pol, ref, cfg.tok, triples, cfg)
    print(f"初始 margin: {report['margins'][0]:+.4f}")
    print(f"最终 margin: {report['margins'][-1]:+.4f}")
    print(f"初始损失:   {report['losses'][0]:.4f}")
    print(f"最终损失:   {report['losses'][-1]:.4f}")

    assert report["margins"][-1] > report["margins"][0], "DPO 训练失败：margin 未增加"
    print("✓ DPO 训练成功：margin 增加")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. 工业工具

### 4.1 TRL 库（Transformer Reinforcement Learning）

HuggingFace 的 TRL 库提供了工业级的 DPO 实现：

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = DPOTrainer(
    model=ref_model,
    ref_model=ref_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

### 4.2 DPO 的变体

| 变体 | 核心思想 | 适用场景 |
|------|---------|---------|
| DPO | 基础版——负 sigmoid 损失 | 基线，标准偏好对 |
| IPO | 改进的正则化版本 | 偏好对噪声大时 |
| KTO | 仅需要单条偏好标记 | 只有"好/坏"而非成对 |
| ORPO | 将 SFT 和偏好优化统一为单阶段 | 训练资源有限时 |

### 4.3 性能对比

| 实现 | 吞吐量 | 内存 | 说明 |
|------|--------|------|------|
| 自定义（本课） | 1x（基线） | 低 | 学习理解 |
| TRL DPO | 5-10x | 中 | 优化后的训练循环 |
| OpenRLHF | 20-50x | 高 | 分布式训练 |

---

## 5. 工程最佳实践

### 5.1 参考模型的冻结方式

**推荐方式**：在参考模型上调用 `param.requires_grad = False` 并设置为 `eval()` 模式。不要复制一个新模型——那样会浪费内存。

### 5.2 对数概率的计算

对数概率必须在序列级别计算——逐词元交叉熵的求和。在计算 DPO 损失时，只需要计算 chosen 和 rejected 补全部分的对数概率，不包括提示词部分。

### 5.3 中文场景特别建议

- **中文偏好数据质量**：中文偏好对的标注难度远高于英文——中文表达的丰富性使得"好"和"坏"的边界更模糊。建议在标注指南中明确具体标准（如"简洁性"、"准确性"、"完整性"）。
- **中文 DPO 数据集**：目前公开的中文 DPO 数据集较少。推荐使用 `CValues`（价值偏好对）、`Alpaca-Chinese-GPT` 等数据集作为起点。

---

## 6. 常见错误

### 错误 1：参考模型接收梯度

**现象：** DPO 训练后参考模型的参数发生改变。

**原因：** 忘记冻结参考模型的参数（`requires_grad = False`）。

**修复：** 确认参考模型的 `requires_grad` 对所有参数为 False，并在 `eval()` 模式下运行。

### 错误 2：对数概率计算包含提示词部分

**现象：** DPO 损失异常大，训练发散。

**原因：** 在计算 DPO 的对数比率差时，包含了提示词部分——提示词的对数概率在 chosen 和 rejected 之间是相同的，加入会稀释偏好信号。

**修复：** 掩码提示词部分，只计算补全部分的对数概率。

### 错误 3：beta 值设置不当

**现象：** beta 过小导致策略过于自信地偏离参考模型；beta 过大导致策略无法从偏好对中学习。

**修复：** 从 beta=0.1 开始，根据验证集的 margin 变化调整。beta 通常在 0.1 到 0.5 之间。

---

## 7. 面试考点

### Q1：DPO 为什么不需要显式奖励模型？（难度：⭐⭐）

**参考答案：** DPO 的关键数学洞察是：最优策略的对数比率差与奖励差成正比。这个恒等式使得可以直接用偏好的似然作为训练目标，而不需要先拟合奖励模型。KL 约束通过 $\beta$ 参数隐式地嵌入了闭式解中——$\beta$ 越大，策略越接近参考模型（更强的 KL 惩罚）。

### Q2：参考模型的三个不变性分别保证了什么？（难度：⭐⭐⭐）

**参考答案：** (1) **参数不接收梯度**：防止参考模型被意外更新，否则对数比率差的参考锚点会漂移；(2) **对数概率在轮次间不变**：确保 DPO 损失的参考值是固定的，训练收敛；(3) **策略从相同权重初始化**：确保对数比率差从零开始，DPO 的梯度信号是纯粹的偏好信号而非初始化差异的噪声。

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| DPO 损失 | 直接偏好优化的核心损失——缩放对数比率差上的负 sigmoid |
| 参考模型 | 冻结的 SFT 模型，对数概率作为训练的锚点 |
| 策略模型 | 从参考初始化并在 DPO 训练中偏离的可训练模型 |
| Chosen/Rejected | 优选/劣选回答——成对的偏好数据 |
| Beta（$\beta$） | 控制策略偏离参考模型强度的超参数 |

---

## 📚 小结

直接偏好优化是将 RLHF 简化为单一监督损失的革命性方法。你从零实现了 DPO 损失、序列对数概率计算和完整的训练循环，验证了 margin 在训练中持续增长。DPO 是现代大语言模型对齐的基石之一。

下一节将构建完整的评测管道——从困惑度到精确匹配的多维度评估。

---

## ✏️ 练习

1. 【理解】用自己的话解释 DPO 损失中的对数比率差 $\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)$ 的物理含义。为什么要减去参考模型？

2. 【实现】将偏好数据从 12 条扩展到 100 条（用模板生成），观察 DPO 训练的收敛速度是否提高。

3. 【实验】将 beta 从 0.1 变化到 1.0，观察最终 margin 和训练稳定性的影响。

4. 【思考】DPO 与 PPO 在训练开销上有何本质区别？为什么工业界越来越倾向于使用 DPO？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| DPO 从零实现 | `code/main.py` | 完整 DPO 训练循环 |
| 可复用配置 | `outputs/skill-dpo.md` | DPO 训练配置与数据格式指南 |

---

## 📖 参考资料

1. [论文] Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model". NeurIPS 2023. https://arxiv.org/abs/2305.18290
2. [官方文档] HuggingFace TRL DPO. https://huggingface.co/docs/trl/main/en/dpo_trainer
3. [GitHub] OpenRLHF. https://github.com/OpenRLHF/OpenRLHF
4. [论文] Christiano et al. "Deep Reinforcement Learning from Human Preferences". NeurIPS 2017. https://arxiv.org/abs/1706.03741
