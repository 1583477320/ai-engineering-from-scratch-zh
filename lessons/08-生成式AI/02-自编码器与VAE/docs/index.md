# 自编码器与 VAE

> 自编码器学会压缩-重建；VAE 学会压缩-采样。从"记住输入"到"生成新输入"，只需要一个随机采样步骤。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 3（深度学习核心）、阶段 7 · 05（完整 Transformer）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现自编码器——理解编码器-解码器在无监督学习中的工作方式
- [ ] 实现 VAE——理解 KL 散度正则化如何强制潜在空间的连续性
- [ ] 说明 VAE 的核心权衡——重建质量 vs 采样多样性

---

## 1. 问题

自编码器是生成模型的"Hello World"。编码器将高维输入压缩到低维潜在空间，解码器从潜在空间重建原始输入。训练目标是最小化重建损失。

**但自编码器不能生成新数据。** 潜在空间是离散的点——随机采样一个点，解码器可能输出垃圾。

VAE 的答案：在潜在空间上加一个正则化项（KL 散度），强制潜在分布接近标准正态分布。这样你可以从标准正态分布中采样——解码器就能生成看起来像训练数据的新样本。

---

## 2. 概念

### 2.1 自编码器——"记住输入"

```
输入 x → [编码器] → 潜在向量 z → [解码器] → 重建 x̂

损失 = ||x - x̂||²（均方误差）
```

潜在空间是离散的——随机采样会输出垃圾。

### 2.2 VAE——"生成新数据"

```
输入 x → [编码器] → μ, σ（均值和方差）→ z = μ + σ·ε（重参数化）→ [解码器] → 重建 x̂

损失 = 重建损失 + KL( q(z|x) || p(z) )
                      ↑
                 正则化：强制潜在分布接近 N(0, I)
```

**关键洞察：** KL 正则化将潜在空间从离散点变成连续空间——从标准正态分布中采样 → 解码器生成新样本。

### 2.3 核心权衡

| | 自编码器 | VAE |
|---|---|---|
| 潜在空间 | 离散点 | 连续（可采样） |
| 重建质量 | 高 | 中等（正则化稀释） |
| 生成能力 | ❌ 不能生成 | ✅ 可以生成新样本 |
| 训练稳定性 | 简单 | 稍复杂（KL 散度） |

---

## 3. 从零实现

### Step 1：自编码器

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
```

### Step 2：VAE

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        # 编码器输出 μ 和 σ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # 输出 2 倍维度
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = μ + σ·ε，允许反向传播。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
```

### Step 3：VAE 损失

```python
def vae_loss(x_recon, x, mu, logvar):
    """重建损失 + KL 正则化。"""
    recon_loss = F.mse_loss(x_recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

**KL 散度**强制潜在分布 `q(z|x)` 接近 `N(0, I)`——这是生成能力的来源。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 潜在空间 | 编码器输出的低维表示——生成模型的"中间世界" |
| 重参数化技巧 | VAE 的核心：`z = μ + σ·ε`——允许通过采样进行反向传播 |
| KL 散度 | 正则化项——强制潜在分布接近标准正态分布 |
| ELBO | 证据下界——VAE 的训练目标，是真实对数似然的下界 |

---

## 📚 小结

自编码器学会了压缩-重建但不能生成新数据。VAE 加上 KL 正则化后，潜在空间变成连续的——从标准正态采样即可生成新样本。核心权衡：重建质量 vs 采样多样性。VAE 是 VQ-VAE（离散化）和扩散模型（更好质量）的基础。

---

## ✏️ 练习

1. 在 MNIST 上训练自编码器和 VAE，对比重建质量和潜在空间可视化
2. 改变 latent_dim（2, 8, 32, 64），观察生成质量变化——找到重建质量和生成多样性的最优平衡点

---

## 📖 参考资料

1. [论文] Kingma & Welling. "Auto-Encoding Variational Bayes". ICLR, 2014.
2. [博客] Lilian Weng. "What are Variational Autoencoders (VAEs)?" 2018.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
