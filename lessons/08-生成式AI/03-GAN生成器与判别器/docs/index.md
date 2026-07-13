# GAN——生成器与判别器

> Goodfellow 2014 年的技巧是完全跳过密度估计。两个网络。一个造假，一个抓假。它们对抗直到假货无法区分真货。这不该有效。它经常失败。但当它有效时，样本仍然是文献中窄域上最锐利的。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 3 · 02（反向传播）、阶段 3 · 08（优化器）、阶段 08 · 02（VAE）| **时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 GAN 的 minimax 训练循环——生成器和判别器的交替更新
- [ ] 解释模式坍塌和梯度消失的机制——GAN 训练的两个经典失败模式
- [ ] 说明 GAN 在 2026 年的实际用途——快速单步采样（SDXL-Turbo）

---

## 1. 问题

VAE 产生模糊样本——因为 MSE 解码损失对多个合理重建取平均值。你想奖励**可信度**而非像素级接近。没有可信度的封闭形式——你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 区分真假图像。训练生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为什么是"真实的"。这个信号随 `G` 的改进而变化——追逐一个移动的目标。如果两个网络都收敛了，`G` 在没有显式写出 `log p(x)` 的情况下学会了数据分布。

---

## 2. 概念

### 2.1 Minimax 博弈

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

**训练 D：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`
**训练 G：** `loss_G = -log D(G(z))`（非饱和形式）

### 2.2 关键失败模式

| 模式 | 机制 | 修复 |
|---|---|---|
| **模式坍塌** | `G` 找到一个 `D` 无法判别的模式并永远生成它 | 噪声注入 + minibatch 判别 + WGAN-GP |
| **梯度消失** | `D` 训练太快，`log D` 饱和 | 降低 D 的学习率 2-5 倍 + 谱归一化 |
| **训练不稳定** | 学习率、batch size 任何变化都可能崩溃 | WGAN-GP + 谱归一化 + 调参 |

### 2.3 2026 年 GAN 的真正用途

GAN 不再是 SOTA 生成器（扩散/Flow Matching 夺走了王冠）。但 GAN 在 2026 年仍然在使用：

- **SDXL-Turbo / SD3-Turbo：** 用对抗训练将 20-50 步扩散管道蒸馏为 1-4 步。**快速单步文本到图像的主导技术**
- **感知损失：** 扩散训练中使用 GAN 判别器作为感知损失
- **人脸：** StyleGAN3 仍然是固定域照片级真实感的基准

---

## 3. 从零实现

### Step 1：非饱和损失

```python
def g_loss(d_fake):
    """G 的损失：最大化 log D(G(z)) → 最小化 -log D(G(z))
    使用非饱和形式避免 D 确信时 G 的梯度消失。"""
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2：训练循环

```python
for step in range(steps):
    # 训练 D：1 步
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)
    
    # 训练 G：1 步（用新的 fake batch，不能用旧的）
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_G(fake_batch)
```

### Step 3：监测模式坍塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式坍塌: 一个模式被饿死了")
```

完整代码见 `code/main.py`。

---

## 4. 生产用途

| 场景 | 选择 |
|---|---|
| 人脸照片级真实感（固定姿态） | StyleGAN3 |
| 快速单步文生图 | 扩散对抗蒸馏（SDXL-Turbo, SD3-Turbo） |
| 图像风格迁移 | Pix2Pix / CycleGAN 或 ControlNet |
| 多模态、开放域生成 | 不要用 GAN——用扩散或 Flow Matching |

GAN 锐利但窄。域一旦开放——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）继续存在，而非作为独立生成器。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 生成器 | 噪声→样本网络 `G: z → x̂` |
| 判别器 | 分类器 `D: x → [0,1]`，区分真假 |
| 非饱和损失 | G 用 `-log D(G(z))` 而非 `log(1-D(G(z)))`——避免梯度消失 |
| 模式坍塌 | G 只生成少数几种样本——尽管数据有多种模式 |
| WGAN | 用 Wasserstein 距离替代 BCE——更平滑的梯度 |
| StyleGAN3 | 对别名自由、平移等变——2026 年人脸基准 |

---

## 📚 小结

GAN 的核心是对抗训练：生成器学会造假，判别器学会抓假，两者博弈直到假货无法区分。两个经典失败：模式坍塌和梯度消失——WGAN-GP 和谱归一化分别解决了它们。2026 年 GAN 不再是 SOTA 生成器——但对抗训练作为蒸馏工具（SDXL-Turbo）仍然主导快速推理。

---

## ✏️ 练习

1. 运行 `code/main.py`（默认参数）。然后设 `D_LR = 5 * G_LR`，观察 G 的 loss 崩溃速度
2. 将 Goodfellow BCE 损失替换为 WGAN 损失——对比训练稳定性和收敛时间

---

## 📖 参考资料

1. [论文] Goodfellow et al. "Generative Adversarial Nets". 2014.
2. [论文] Radford et al. "Unsupervised Representation Learning with Deep Convolutional GAN" (DCGAN). 2015.
3. [论文] Arjovsky et al. "Wasserstein GAN". 2017.
4. [论文] Miyato et al. "Spectral Normalization for Generative Adversarial Networks". 2018.
5. [论文] Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2). 2020.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
