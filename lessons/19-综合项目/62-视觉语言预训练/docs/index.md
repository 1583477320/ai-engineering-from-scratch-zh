# 综合项目62——视觉语言预训练（Vision-Language Pretraining）

> 编码器、投影层和解码器已连接。现在训练它们。两个目标驱动学习：对比损失和语言建模损失。组合起来教会模型既找到正确的图像，也为图像写标题。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第58-61节
**预计时间：** 90分钟

---

## 学习目标

- 实现 InfoNCE 对比损失（跨批次图像-标题对）
- 组合对比损失与自回归语言建模损失
- 合成 200 对模拟语料
- 运行 50 步训练循环观察两损失同时下降

---

## 1. 问题

视觉语言模型需要两种技能。排名：给定标题在多张图中找到正确的那张。生成：给定图像写一个标题。只训练一种只能得到一半系统。CLIP 擅长排名但不会生成。GPT-4V 会生成但用单独的检索头排名。多目标预训练一次训练两者。

---

## 2. 核心概念

### 2.1 InfoNCE 对比损失

将 N 个图像嵌入作为行，N 个文本嵌入作为行。L2 归一化两者。计算相似矩阵 `S = I T^T / tau`。对角线是正样本，非对角是负样本。双向交叉熵损失。

### 2.2 温度 `tau`

温度控制 softmax 峰值。太小（0.01）→ 梯度来自最难负样本，训练噪声大。太大 → 梯度消失。CLIP 和本课都学习 `tau` 为可训练参数。

### 2.3 语言建模损失

解码器通过交叉注意力消费图像记忆词元，预测下一个文本词元。标准交叉熵，padding 位置被掩码。

### 2.4 组合损失

```
total = contrastive + lm_weight × lm_loss
```

两个损失都向编码器和投影层回传梯度；只有 LM 损失向解码器回传梯度。

---

## 3. 从零实现

```python
"""视觉语言预训练——InfoNCE + LM 损失。"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def forward(self, img_emb, txt_emb):
        tau = self.log_tau.exp()
        img_n = F.normalize(img_emb, dim=-1)
        txt_n = F.normalize(txt_emb, dim=-1)
        sim = img_n @ txt_n.T / tau
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return (loss_i2t + loss_t2i) / 2


def lm_loss(logits, targets, pad_id=0):
    B, V = logits.shape
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=pad_id)


class MiniViT(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        return self.norm(cls)


class MLP(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class CrossDecBlock(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x, mem):
        B, N, D = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        x = x + self.cross_attn(self.ln2(x), mem, mem)[0]
        return x + self.ffn(self.ln3(x))


class VLModel(nn.Module):
    def __init__(self, vocab=256, dim=64, heads=4, depth=2):
        super().__init__()
        self.enc = MiniViT(dim)
        self.proj = MLP(dim)
        self.txt_emb = nn.Embedding(vocab, dim)
        self.dec = nn.ModuleList([CrossDecBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab)

    def encode_image(self, x):
        return self.proj(self.enc(x))

    def decode(self, text_ids, img_mem, pos_emb):
        x = self.txt_emb(text_ids) + pos_emb[:, :text_ids.shape[1]]
        for blk in self.dec: x = blk(x, img_mem)
        return self.head(x)


def make_pairs(n=200, vocab=256, seq=8, dim=64, seed=0):
    torch.manual_seed(seed)
    pairs = []
    for _ in range(n):
        img = torch.randn(1, 196, dim)
        cap = torch.randint(2, vocab, (seq,))
        pairs.append((img.squeeze(0).unsqueeze(0), cap))
    return pairs


def main():
    model = VLModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pairs = make_pairs()
    batch_size = 16
    pos_emb = nn.Parameter(torch.randn(1, 32, 64) * 0.02)
    model.register_parameter("pos_emb", pos_emb)

    print(f"训练 50 步 (batch_size={batch_size})")
    for step in range(50):
        batch = pairs[(step * batch_size) % len(pairs):(step * batch_size) % len(pairs) + batch_size]
        if len(batch) < 2: continue
        imgs = torch.cat([p[0] for p in batch])
        caps = torch.stack([p[1] for p in batch])

        img_emb = model.encode_image(imgs)
        txt_emb = model.proj(model.txt_emb(caps).mean(1))
        cl = InfoNCELoss()(img_emb, txt_emb)

        mem = model.enc(imgs)
        logits = model.decode(caps[:, :-1], mem, model.pos_emb)
        ll = lm_loss(logits, caps[:, 1:])

        loss = cl + 0.5 * ll
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 10 == 0:
            print(f"  步{step:3d}: 对比={cl.item():.3f} LM={ll.item():.3f} 总={loss.item():.3f}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 框架 | 对比损失 | LM 损失 | 多目标 |
|:----|:--------|:--------|:------|
| CLIP | ✓ | ✗ | ✗ |
| CoCa | ✓ | ✓ | ✓ |
| BLIP-2 | ✓ | ✓ | ✓ |
| SigLIP | sigmoid | ✗ | ✗ |
| 本课 | InfoNCE | ✓ | ✓ |

---

## 5. 工程最佳实践

- `tau` 应该是可学习参数（log-scale 初始化）
- **中文场景建议**：多目标权重 `lm_weight` 对中文标题生成至关重要——中文标题通常比英文短

---

## 6. 常见错误

- **温度 `tau` 未归一化**：使用 `log_tau` + `exp()` 确保正数
- **LM 损失未掩码 padding**：忽略 `pad_id=0` 位置
- **batch 太小**：InfoNCE 至少需要 8+ 样本才有足够负样本

---

## 7. 面试考点

**Q1：InfoNCE 的正负样本如何定义？**（难度：⭐⭐）

**参考答案：** 一个 batch 中 N 对配对是正样本（N 个），非配对的 `N² - N` 个是负样本。双向损失使图像和文本编码器都被正确训练。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| InfoNCE | 对比损失——对相似矩阵做交叉熵 |
| 温度 tau | 控制 softmax 锐度的可学习标量 |
| 硬负样本 | 模型混淆的非配对，可采样增强 |
| 联合嵌入空间 | 图像和文本向量投影后的共享空间 |

---

## 📚 小结

视觉语言预训练通过对比和语言建模的组合训练图像理解与生成能力。下一节构建多模态评估指标。

---

## ✏️ 练习

1. 【实现】替换 InfoNCE 为 SigLIP sigmoid 对比损失
2. 【实验】用 `lm_weight=0` vs `lm_weight=1` 对比对比损失变化

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 预训练循环 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Radford et al. "Learning Transferable Visual Models from Natural Language Supervision". ICML 2021.
2. [论文] Yu et al. "CoCa: Contrastive Captioners are Image-Text Foundation Models". 2022.
