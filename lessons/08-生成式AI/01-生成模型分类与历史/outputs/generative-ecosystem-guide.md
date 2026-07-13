---
name: generative-ecosystem-guide
description: 根据任务特征选择生成模型家族的参考指南。
phase: 8
lesson: 01
---

# 生成模型生态选型指南

> 选择生成模型不是"哪个最好"，而是"哪个最适合你的任务约束"。

---

## 决策流程

```
任务类型是什么？
│
├─ 文本生成（对话、写作、代码、推理）
│  └─ → 自回归大语言模型（GPT-4、Llama 3、Qwen）
│     原因：变长序列、零样本能力、涌现推理
│
├─ 图像生成
│  ├─ 通用（文生图、图生图）→ 扩散 / Flow Matching（SD3、Flux）
│  ├─ 固定域（人脸、室内）→ GAN（StyleGAN3）
│  └─ 需要精确控制 → 扩散 + ControlNet / IP-Adapter
│
├─ 视频生成
│  ├─ 短视频（< 30s）→ 扩散 Transformer（Sora、CogVideoX）
│  └─ 长视频 → 自回归 token + 扩散解码（VideoPoet）
│
├─ 音频 / 语音
│  ├─ 语音合成 → Flow Matching（ChatTTS、Fish-Speech）
│  └─ 音乐生成 → AudioCraft 2（Flow Matching）
│
├─ 3D 生成
│  ├─ 文本到 3D → 扩散 + NeRF（Point-E、Shap-E）
│  └─ 图像到 3D → 扩散 + 多视角生成
│
└─ 蛋白质结构
   └─ → 自回归 / 扩散（AlphaFold 2、RFdiffusion）
```

---

## 五大家族速查表

| 家族 | 训练稳定性 | 样本质量 | 采样速度 | 条件控制 | 典型应用 |
|---|---|---|---|---|---|
| 自回归（GPT） | ★★★★★ | ★★★★ | ★★★★★（单步） | ★★★ | 文本、代码、推理 |
| VAE | ★★★★ | ★★★ | ★★★★★（单步） | ★★ | 表征学习、异常检测 |
| GAN（StyleGAN） | ★★ | ★★★★★ | ★★★★★（单步） | ★★★ | 人脸、固定域图像 |
| 扩散 / Flow Matching | ★★★★ | ★★★★★ | ★★（需多步） | ★★★★★ | 通用图像/视频/3D |
| VQ-VAE + Transformer | ★★★ | ★★★★ | ★★★★ | ★★★★ | 多模态统一（Sora） |

---

## 关键权衡

### 精确似然 vs 训练稳定性

- 需要计算 `log p(x)`（如异常检测、密度估计）→ 选自回归或流模型
- 只需要生成样本 → 扩散模型或 GAN 即可
- 训练预算有限 → 扩散模型最稳定，GAN 风险最高

### 采样速度 vs 样本质量

- 实时应用（聊天、交互）→ 自回归（每步一次前向传播）
- 离线生成（批量图片、视频渲染）→ 扩散模型（10-50 步）
- 极致速度 + 固定域 → GAN（StyleGAN，一次前向传播）

### 通用性 vs 专用性

- 通用任务（任意文本/图像/代码）→ 大语言模型（自回归）
- 专用任务（人脸生成、语音克隆）→ 专用模型（GAN / Flow Matching）
- 多模态统一 → VQ-VAE + Transformer（Sora 路线）

---

## 2026 年趋势

1. **Flow Matching 取代 DDPM**：采样快 4-10 倍，Stable Diffusion 3、Flux 已全面采用
2. **扩散 Transformer（DiT）**：Sora、SD3 用 Transformer 替代 U-Net 作为去噪骨干
3. **Consistency Models**：从扩散模型蒸馏出一步生成模型，推理速度接近 GAN
4. **多模态统一**：VQ-VAE + Transformer 路线让同一个架构处理文本/图像/视频/音频
5. **自回归统治文本**：没有迹象表明扩散模型会在文本生成上取代自回归

---

## 快速推荐

| 你的情况 | 推荐方案 |
|---|---|
| 刚入门生成式 AI | Stable Diffusion + HuggingFace Diffusers |
| 需要文本生成 | GPT-4 / Llama 3 / Qwen |
| 人脸/肖像生成 | StyleGAN3 |
| 视频生成 | Sora / CogVideoX |
| 音频/语音 | AudioCraft 2 / ChatTTS |
| 生产环境部署 | vLLM（文本）/ TensorRT（图像） |
