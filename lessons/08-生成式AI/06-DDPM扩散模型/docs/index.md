# DDPM 扩散模型

> 扩散模型将图像逐步加噪到纯高斯噪声，再学会逆向去噪。这个简单的想法在 2020 年后主导了图像、视频和 3D 生成。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 08 · 02（VAE）| **时间：** ~90 分钟

---

## 🎯 学习目标

- [ ] 实现 DDPM 的前向扩散过程——逐步向图像添加高斯噪声
- [ ] 实现去噪网络 U-Net——预测每步的噪声残差
- [ ] 解释采样过程——如何从纯噪声逐步恢复出图像

---

## 1. 问题

GAN 训练不稳定（模式坍塌、训练不收敛）。VAE 生成质量有限（模糊）。有没有一种方法既能像 GAN 一样生成锐利图像，又能像 VAE 一样稳定训练？

扩散模型的思路：**不直接学习数据分布，而是学习去噪过程。** 训练一个网络从噪声中恢复原始图像——然后在推理时从随机噪声开始，反复去噪，直到生成新图像。

---

## 2. 概念

### 2.1 前向过程（加噪）

$$x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1 - \alpha_t} \cdot \epsilon$$

逐步向图像添加高斯噪声。$T$ 步后，图像变成纯高斯噪声。**这个过程不需要学习——它是固定的。**

### 2.2 反向过程（去噪）

训练一个 U-Net 预测每步添加的噪声 $\epsilon$。损失：

$$L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

即预测噪声与真实噪声的均方误差。

### 2.3 采样过程

从随机噪声 $x_T \sim N(0, I)$ 开始，应用 $T$ 步去噪。每一步：预测噪声→从当前图中减去噪声→重复。**T=1000 步的 DDPM 需要 1000 次 U-Net 前向传播——很慢。** 2024-2026 的改进（DDIM、Consistency Model）将其压缩到 4-20 步。

---

## 3. 从零实现

```python
class SimpleUNet(nn.Module):
    """简化 U-Net——预测噪声残差。"""
    def __init__(self, in_channels=1, time_emb_dim=64):
        super().__init__()
        self.time_embed = nn.Linear(1, time_emb_dim)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.mid = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t.float().unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        h = self.down(x + t_emb)
        h = self.mid(h)
        return self.up(h)

def forward_process(x_0, t, betas):
    """前向扩散：向 x_0 添加噪声。"""
    alpha_t = 1.0 - betas[:t].prod()
    noise = torch.randn_like(x_0)
    x_t = math.sqrt(alpha_t) * x_0 + math.sqrt(1 - alpha_t) * noise
    return x_t, noise

def train_step(model, x_0, t, betas):
    """一步训练：预测噪声。"""
    x_t, noise = forward_process(x_0, t, betas)
    pred_noise = model(x_t, torch.tensor([t]))
    return F.mse_loss(pred_noise, noise)
```

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 前向扩散 | 逐步向图像添加高斯噪声到纯噪声 |
| 反向扩散 | 训练去噪网络——从噪声中恢复原始图像 |
| U-Net | 扩散模型的核心网络——编码-解码结构，跳跃连接 |
| 采样步数 T | 去噪的步数——T=1000 是 DDPM 默认，4-20 是加速方法 |

---

## 📚 小结

DDPM 将图像生成简化为"学习去噪"：训练 U-Net 预测每步添加的噪声，然后从随机噪声开始逐步去噪。稳定训练、高质量样本、清晰的数学基础。代价：1000 步去噪需要 1000 次 U-Net 前向——太慢。2024-2026 的 DDIM/Consistency Model 将步数压缩到 4-20，保留了质量。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
