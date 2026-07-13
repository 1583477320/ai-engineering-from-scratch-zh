# VAE 实用指南

> 变分自编码器（VAE）是生成式 AI 的"第一课"。本指南帮助你从工程角度理解 VAE 的选型、调参和集成。

---

## 1. 快速选型

| 场景 | 推荐模型 | 原因 |
|---|---|---|
| 数据去噪、特征学习 | 标准自编码器 | 重建质量高，训练简单，不需要生成 |
| 数据压缩后用于下游任务 | 标准自编码器 | 潜在向量是确定性的，下游训练稳定 |
| 从潜在空间采样生成新数据 | VAE | 连续潜在空间，KL 正则化保证可采样 |
| 在潜在空间上做插值/AI 编辑 | VAE | 线性插值自然（自编码器的插值会经过空隙） |

## 2. 超参数指南

### 2.1 潜在维度 (latent_dim)

- **太小（1-4）**：KL 正则化强度相对较大，重建模糊，但采样质量高（多样性好）
- **合适（8-64）**：平衡重建和生成。MNIST 用 16 足够，CIFAR-10 用 64 起步
- **太大（128+）**：KL 正则化的效果弱化，模型倾向于"记住"数据，生成失去多样性

### 2.2 KL 散度权重 (beta)

标准 VAE 的损失是 `recon + KL`。如果你想控制权衡：

```python
# β-VAE：用一个 beta 系数控制 KL 权重
loss = recon_loss + beta * kl_loss
```

- `beta < 1`：重建质量更好（适合数据压缩）
- `beta > 1`：潜在空间更规则，生成多样性更好（适合生成任务）
- `beta = 0`：退化为标准自编码器（没有生成能力）

### 2.3 损失平衡技巧

KL 散度在训练初期可能接近零（后验坍塌——模型学会了不管输入都输出 N(0,I)）。解决方法：

```python
# KL 散度预热：训练初期让 KL 权重逐渐从 0 增加到 1
def kl_anneal(epoch, warmup_epochs=10, beta_max=1.0):
    """轮次线性增加 beta。"""
    if epoch < warmup_epochs:
        return beta_max * (epoch + 1) / warmup_epochs
    return beta_max
```

## 3. 从潜在空间采样

### 3.1 随机采样

```python
z = torch.randn(1, latent_dim)  # 从 N(0, I) 采样
generated = vae.decoder(z)
```

### 3.2 类别条件采样

如果使用条件 VAE（cVAE），采样时需要传入类别标签：

```python
# 提示词：采样类别为 5 的数字
z = torch.randn(1, latent_dim)
c = F.one_hot(torch.tensor([5]), num_classes=10)  # 条件向量
generated = cvae.decoder(torch.cat([z, c], dim=-1))
```

### 3.3 隐空间插值

VAE 的潜在空间是连续的，支持平滑插值：

```python
# 在两张图像之间做插值
z_a = vae.encoder(img_a)[0]  # latent dim: 2
z_b = vae.encoder(img_b)[0]
for alpha in np.linspace(0, 1, 5):
    z_interp = alpha * z_a + (1 - alpha) * z_b
    interp_img = vae.decoder(z_interp)
    # 保存插值结果
```

## 4. VAE 的工业应用

### 4.1 数据放大

在训练数据不足时，VAE 可以合成更多训练样本：

- 训练 VAE 覆盖原始数据分布
- 从 N(0,I) 采样生成新样本
- 筛选高质量样本加入训练集
- 这是半监督学习的一种经典做法

### 4.2 异常检测

重建误差是天然的异常检测分数：

```python
def anomaly_score(model, x):
    """重建误差越高，越可能是异常。"""
    x_recon, _, _ = model(x)
    return F.mse_loss(x_recon, x, reduction="none").mean(dim=-1)

# 异常阈值 = 训练集重建误差的 95% 分位数
```

### 4.3 潜在空间特征提取

VAE 编码器可以作为特征提取器，替代 PCA/t-SNE：

```python
# 将高维数据映射到低维潜在空间
def extract_features(vae, dataset):
    vae.eval()
    features = []
    for x in dataset:
        _, mu, _ = vae(x.unsqueeze(0))
        features.append(mu.detach().numpy())
    return np.vstack(features)
```

### 4.4 文本 VAE（生成文本）

```python
class TextVAE(nn.Module):
    """文本领域使用 VAE——编码器是 LSTM。"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim + embed_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
```

## 5. 局限性

| 局限性 | 说明 | 替代方案 |
|---|---|---|
| 重建模糊 | KL 正则化约束了信息容量 | VQ-VAE（离散潜在空间，保留更多信息） |
| 生成质量低于 GAN | 高斯先验过于简单 | 扩散模型 / GAN 结合 VAE |
| 对先验假设敏感 | 假设后验和先验都是高斯分布 | Normalizing Flow 增强后验的灵活性 |
| 训练不稳定（后验坍塌） | 解码器太强时会忽略潜在变量 | KL 预热 / Free Bits 技巧 |

## 6. 相关资源

- [论文] Kingma & Welling. "Auto-Encoding Variational Bayes". 2013.
- [论文] Higgins et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework". 2017.
- [论文] van den Oord et al. "Neural Discrete Representation Learning" (VQ-VAE). 2017.
- [官方文档] PyTorch VAE 实现示例: https://github.com/pytorch/examples/tree/main/vae

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
