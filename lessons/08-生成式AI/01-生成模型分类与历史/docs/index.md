# 生成模型——分类与历史

> 每一个图像模型、文本模型、视频模型和 3D 模型都属于五类之一。选错类别，你会和数学斗争几周；选对类别，这个领域过去十二年的进步会清晰地堆叠在你脑中。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 2（机器学习基础）、阶段 3（深度学习核心）、阶段 7 · 14（Transformer）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分五大生成模型家族——显式密度（可处理）、显式密度（近似）、隐式密度、基于分数的、基于 token 的自回归
- [ ] 说出 2026 年每个家族的代表模型——Diffusion、Flow Matching、GAN、VAE、GPT
- [ ] 解释为什么 Diffusion 和 Flow Matching 在 2026 年统治图像/视频/3D 生成

---

## 1. 问题

生成模型只做一件事：给定训练样本，输出新的样本——人脸、句子、MIDI、蛋白质结构。但 `p_data`（数据分布）生活在数百万维度的空间中（512×512 RGB 图像 = ~786k 维度），样本只占据其中的薄薄一层流形。暴力求解密度分布是无望的。每个生成模型都是用一个难题换一个稍微不那么难的难题的妥协。

五个家族在过去十二年中存活了下来。知道每个家族做了什么妥协，就能理解为什么它在某些任务上赢了、在某些任务上崩了。

---

## 2. 概念

### 2.1 五大类生成模型

**1. 显式密度，可处理。** 把 `log p(x)` 写成可以实际计算的求和。自回归模型（PixelCNN、WaveNet、GPT）将 `p(x)` 分解为 `∏ p(x_i | x_{<i})`。正常化流（RealNVP、Glow）将 `p(x)` 构建为一个简单基础分布的可逆变换。**优点：** 精确似然，干净的训练损失。**缺点：** 自回归推理是串行的（长序列很慢）；流需要可逆架构（架构限制）。

**2. 显式密度，近似。** 将 `log p(x)` 从下方限定（ELBO）并优化这个界。VAE（Kingma 2013）用编码器-解码器和变分后验。扩散模型（DDPM, Ho 2020）训练一个去噪器，隐式优化加权 ELBO。**2026 年扩散是图像、视频和 3D 的主力。**

**3. 隐式密度。** 跳过密度，直接学习生成器 `G(z)` 产生样本，判别器 `D(x)` 区分真假。GAN（Goodfellow 2014）。推理快（一次前向），训练出名地不稳定。**StyleGAN 1/2/3 仍是固定域照片级真实感（人脸、卧室）的 SOTA。**

**4. 基于分数/连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（分数）。Song & Ermon（2019）展示了分数匹配将扩散推广到 SDE。**Flow Matching（Lipman 2023）是 2024-2026 的热门：无模拟训练、更直的路径、采样比 DDPM 快 4-10 倍。** Stable Diffusion 3、Flux、AudioCraft 2 都用 Flow Matching。

**5. 基于 token 的离散码自回归。** 用 VQ-VAE 或残差量化器将高维数据压缩成离散 token 的短序列，然后用 Transformer 建模 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都是这个——桶 1 加上学到的 tokenizer。

### 2.2 历史里程碑

| 年份 | 模型 | 意义 |
|---|---|---|
| 2013 | VAE（Kingma） | 第一个有可用训练损失的深度生成模型 |
| 2014 | GAN（Goodfellow） | 隐式密度，无似然——令人震惊的锐利样本 |
| 2019 | StyleGAN / StyleGAN2 | 人脸照片级真实感至今难超越 |
| 2020 | DDPM（Ho） | 扩散变得实际可用 |
| 2022 | Stable Diffusion 1 | 潜在扩散 + 文本条件 = 商品化 |
| 2024 | Sora、SD3、Flow Matching | 视频扩散；Flow Matching 胜出 |
| 2026 | Consistency + Rectified Flow | 从扩散骨干一步采样 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| VAE | 变分自编码器——显式密度近似，ELBO 作为训练损失 |
| GAN | 生成对抗网络——隐式密度，G 和 D 对抗训练 |
| Diffusion | 扩散模型——学习去噪过程，隐式优化 ELBO |
| Flow Matching | 流匹配——无模拟训练，更直的路径，采样比 DDPM 快 4-10 倍 |
| VQ-VAE | 向量量化 VAE——将连续嵌入离散化为有限码本中的 token |

---

## 📚 小结

五大生成模型家族：显式可处理密度（自回归）、显式近似密度（VAE/Diffusion）、隐式密度（GAN）、基于分数的（Flow Matching）、基于 token 的离散自回归。2026 年的格局：Diffusion/Flow Matching 主导图像/视频/3D；StyleGAN 在人脸领域仍然统治；自回归 LLM（GPT/LLaMA）在文本领域统治。知道每个家族的妥协——就知道它在哪里赢、在哪里输。

---

## ✏️ 练习

1. 画出五大类生成模型的决策树——给定任务特征，如何选择？图像→？视频→？文本→？3D→？
2. 对比 Diffusion 和 GAN 在人脸生成上的样本质量和训练稳定性——画出各自的 trade-off 曲线

---

## 📖 参考资料

1. [论文] Kingma. "Auto-Encoding Variational Bayes" (VAE). 2013.
2. [论文] Goodfellow et al. "Generative Adversarial Nets" (GAN). 2014.
3. [论文] Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM). 2020.
4. [论文] Lipman et al. "Flow Matching for Generative Modeling" (Flow Matching). 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
