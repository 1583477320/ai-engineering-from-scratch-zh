# 音频生成

> 文本→语音（TTS）是最成熟的音频生成任务；文本→音乐、文本→音效是扩展。Diffusion 和自回归是两大技术路线。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 06 · 07（TTS）、阶段 06 · 13（音频编解码）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 06 · 07（TTS）— 传统 TTS 管道 | 阶段 08 · 06（DDPM）— 扩散模型在音频中的应用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分 TTS（语音）、音乐生成、音效生成三类任务的技术路线
- [ ] 解释神经编解码器（EnCodec / Codec）如何在音频生成中充当"分词器"
- [ ] 说明 2026 年音频生成的两个前沿——F5-TTS（零样本 TTS）和 Flow Matching
- [ ] 使用 Hugging Face 或 TTS 库生成语音
- [ ] 对比扩散和自回归两种音频生成架构的优劣势

---

## 1. 问题

音频远比图像复杂。图像是 2D 空间数据（H×W），每个像素是标量。音频是 1D 时间序列，每秒有 44,100 个采样点，而且**不是所有数据都重要**——语音中 30% 是沉默，音乐中的噪声频谱很难建模。

早期 TTS 系统（如 Tacotron、FastSpeech）使用梅尔频谱 + 声码器的两阶段管道——先预测频谱，再从频谱合成波形。复杂、臃肿、端到端不可微。

2024-2026 年，**所有音频生成任务（TTS、音乐、音效）收敛到同一个范式**：神经编解码器（Neural Codec）将波形压缩为离散 token，然后一个 Transformer LM 在 token 序列上做自回归生成或扩散生成。这个范式与 LLM 的做法几乎完全一致——音频 token 就是"音频的词元"。

---

## 2. 概念

### 2.1 直观理解

```
传统 TTS 管道：
  文本 → [文本前端] → [声学模型] → 频谱 → [声码器] → 波形
  
现代生成管道：
  文本 → [Transformer] → 音频 token 序列 → [Codec 解码器] → 波形
```

现代方法的关键区别：**用神经编解码器替代了频谱 + 声码器。** Codec 将音频转换为离散 token 的过程，就像分词器（tokenizer）将文本转换为整数序列。Transformer 在 token 空间做预测，而不是在频谱空间。

### 2.2 三类音频生成任务

| 任务 | 输入 | 输出 | 核心挑战 | 代表模型 |
|------|------|------|---------|---------|
| **TTS（语音合成）** | 文本 | 语音 | 自然度、韵律、多音色 | F5-TTS, Kokoro, CosyVoice |
| **音乐生成** | 文本/和弦 | 音乐 | 旋律、和声、节奏、结构 | MusicGen, Stable Audio, Suno |
| **音效生成** | 文本 | 音频 | 多样性、真实感 | AudioCraft, AudioLDM |

### 2.3 神经编解码器——音频的"分词器"

神经编解码器（Neural Codec）将连续的音频波形压缩为离散 token 序列。

```
波形 (16kHz, 16-bit PCM) → [Codec 编码器] → 离散 token 序列 [t_1, t_2, ..., t_N]
                                                    ↓
                                          每个 token ∈ [0, 2^K - 1]
                                                   |
                                                   ↓
离散 token 序列 → [Codec 解码器] → 波形
```

**EnCodec（Meta, 2022）：** 使用 RVQ（残差向量量化，Residual Vector Quantization）将音频压缩为多个层次的 token。第一层 token 编码大结构（如音高），后续层次编码精细细节（如音色）。

| Codec 模型 | 发布 | 比特率 | 质量 | 使用场景 |
|-----------|------|--------|------|---------|
| EnCodec | Meta 2022 | 3-24 kbps | 中 | AudioCraft, MusicGen |
| Audio Codec | Google 2023 | 3-12 kbps | 高 | Gemini |
| DAC | Descript 2023 | 8-32 kbps | 高 | F5-TTS |
| CosyVoice Codec | 阿里 2024 | 10-20 kbps | 高 | CosyVoice TTS |

### 2.4 AudioCraft / MusicGen 的架构

Meta 的 AudioCraft 统一了音频生成的架构：

```
文本 → [文本编码器 T5] → 条件向量 → [Transformer LM 在 EnCodec token 上] → [EnCodec 解码器] → 波形
```

关键：**音频被简化为一个 token 序列后，生成变成 token 预测任务。** 这与 LLM 的文本生成几乎一样。

MusicGen 是 AudioCraft 的音乐生成版。它支持文本到音乐、旋律到音乐（给定一段旋律参考生成类似风格的音乐）。

### 2.5 2026 年的两个前沿

**F5-TTS（2024-2026）：** Flow Matching 用于语音合成。核心创新：不需要音素对齐（phoneme alignment），零样本 TTS。只需 3-5 秒的参考音频即可克隆音色和说话风格。训练直观：将一段音频的梅尔谱图表示为流场中的一个点，用 Flow Matching 从噪声流向目标。

**Flow Matching 用于音频：** 扩散模型的流匹配版本在音频生成中表现出色。优点：(1) 采样步数从 1000 降到 10-50；(2) 质量优于传统扩散模型；(3) 训练更稳定。

### 2.6 自回归 vs 扩散

| 方法 | 优点 | 缺点 | 代表 |
|------|------|------|------|
| **自回归 Transformer** | 高质量、可控、容易条件化 | 速度慢（逐 token） | MusicGen, AudioCraft |
| **扩散 / Flow Matching** | 速度快（并行生成）、训练稳定 | 不如自回归可控 | F5-TTS, Voicebox |
| **混合** | 兼顾速度和可控性 | 复杂 | NaturalSpeech 3 |

---

## 3. 从零实现

AudioCraft/MusicGen 的架构可以用不到 100 行核心代码理解：

### 第 1 步：音频 token 的编解码概念

```python
import torch
import torch.nn as nn
import numpy as np


class NeuralAudioCodecConcept(nn.Module):
    """
    神经编解码器的概念演示。
    真实模型使用 24 层 RVQ、32 码本，这里用简化版说明原理。
    """

    def __init__(self, input_dim=128, codebook_size=1024, num_quantizers=4):
        super().__init__()
        self.num_quantizers = num_quantizers

        # 编码器：梅尔谱图 → 潜向量
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(512, 256, 3, padding=1),
        )

        # 码本（简化版：只用一个码本实际用多个码本的 RVQ）
        self.codebook = nn.Embedding(codebook_size, 256)

        # 解码器：潜向量 → 梅尔谱图
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, input_dim, 3, padding=1),
        )

    def quantize(self, z):
        """
        量化：将连续潜向量映射到最近的码本向量。
        Args:
            z: 潜向量 (B, 256, T)
        Returns:
            z_q: 量化后的潜向量 (B, 256, T)
            indices: 码本条引 (B, T)
        """
        B, D, T = z.shape
        # 将码本展平为 (codebook_size, D)
        codebook = self.codebook.weight  # (codebook_size, D)

        # 转置 z 为 (B, T, D)
        z_perm = z.permute(0, 2, 1).contiguous()  # (B, T, D)

        # 对每个潜向量找到最近的码本条
        # (B, T, D) @ (D, codebook_size) -> (B, T, codebook_size)
        distances = torch.cdist(z_perm, codebook.unsqueeze(0).expand(B, -1, -1))
        indices = distances.argmin(dim=-1)  # (B, T)

        # 查找码本向量
        z_q = codebook[indices].permute(0, 2, 1)  # (B, D, T)
        return z_q, indices

    def encode_to_indices(self, mel):
        """编码为 token 索引。"""
        z = self.encoder(mel)
        _, indices = self.quantize(z)
        return indices

    def decode_from_indices(self, indices):
        """从 token 索引解码为梅尔谱图。"""
        codebook = self.codebook.weight  # (codebook_size, D)
        z_q = codebook[indices].permute(0, 2, 1)  # (B, D, T)
        mel = self.decoder(z_q)
        return mel
```

### 第 2 步：简化版音频生成 Transformer

```python
class SimpleAudioGenerationLM(nn.Module):
    """
    简化版音频生成模型——在音频 token 序列上做 Transformer 生成。
    这是 MusicGen / AudioCraft 的核心。
    """

    def __init__(self, codebook_size=1024, embed_dim=512, num_heads=8,
                 num_layers=6, context_length=512):
        super().__init__()

        # token 嵌入
        self.token_embedding = nn.Embedding(codebook_size, embed_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, context_length, embed_dim) * 0.02
        )

        # 文本条件嵌入（可选）
        self.text_condition_proj = nn.Linear(768, embed_dim)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出头：预测下一个 token
        self.output_proj = nn.Linear(embed_dim, codebook_size)

    def forward(self, audio_tokens, text_embed=None):
        """
        Args:
            audio_tokens: 音频 token 序列 (B, T)
            text_embed: 文本条件嵌入 (B, 768)（可选）
        Returns:
            logits: 下一个 token 的预测 logits (B, T, codebook_size)
        """
        # token 嵌入
        x = self.token_embedding(audio_tokens)  # (B, T, embed_dim)

        # 位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]

        # 文本条件（如果有）
        if text_embed is not None:
            x = x + self.text_condition_proj(text_embed).unsqueeze(1)

        # Transformer 处理
        x = self.transformer(x)

        # 预测下一个 token
        logits = self.output_proj(x)
        return logits
```

### 第 3 步：TTS 推理流程

```python
def tts_sample(model, text_tokens, audio_tokenizer, num_tokens=128, temperature=1.0):
    """
    简化版 TTS 采样流程。
    真实系统还需要: VQ-VAE 解码器 + 声码器或 Codec 解码器。
    """
    model.eval()
    # 从文本嵌入开始
    with torch.no_grad():
        # 文本嵌入（来自 T5 / CLAP 等编码器）
        text_embed = torch.randn(1, 768)  # 简化：实际来自文本编码器

        # 生成音频 token 序列（自回归）
        generated_tokens = []
        # 从 <BOS> token 开始
        current = torch.tensor([[0]], device=next(model.parameters()).device)

        for _ in range(num_tokens):
            logits = model(current, text_embed)
            next_logits = logits[:, -1, :] / temperature
            # 采样下一个 token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)

        # 返回生成的 token 序列
        return torch.cat(generated_tokens, dim=1)
```

---

## 4. 工具

### 4.1 Hugging Face Transformers 中的 TTS

```python
from transformers import AutoProcessor, MusicGenForConditionalGeneration
import torchaudio

# 加载 MusicGen（文本→音乐）
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicGenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# 生成音乐
prompt = "80年代流行摇滚，充满合成器和有力的鼓点"
inputs = processor(text=[prompt], padding=True, return_tensors="pt")
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                              max_new_tokens=256)

# 保存音频
# audio_values: (batch, channels, time)
# torchaudio.save("output.wav", audio_values[0],sample_rate=model.config.audio_encoder.sampling_rate)
```

### 4.2 TTS 库对比

| 模型 | 类型 | 质量 | 速度 | 开源 | 中文支持 | 特殊能力 |
|------|------|------|------|------|---------|---------|
| CosyVoice | 自然 TTS | ⭐⭐⭐⭐ | 快 | ✅ 阿里 | ✅ | 多音色、情感克隆 |
| F5-TTS | 零样本 TTS | ⭐⭐⭐⭐⭐ | 中 | ✅ | ✅ | 3 秒音频克隆 |
| Kokoro | TTS | ⭐⭐⭐⭐ | 极快 | ✅ | ✅ | 轻量级部署 |
| VITS | TTS | ⭐⭐⭐ | 中 | ✅ | ✅ | 经典基线 |
| MetaVoice | TTS | ⭐⭐⭐⭐ | 快 | ✅ | ❌ | 英文优化 |
| Edge TTS | TTS | ⭐⭐⭐⭐ | 快 | ❌ Microsoft | ✅ | 免费 API |

### 4.3 音乐生成对比

| 模型 | 授权 | 时长 | 质量 | 中文提示词 |
|------|------|------|------|-----------|
| MusicGen | 开源（MIT） | 30 秒 | 中 | 有限 |
| Stable Audio | 开源 | 180 秒 | 高 | 有限 |
| Suno | 闭源 | 4 分钟 | 高 | 支持 |
| Udio | 闭源 | 2 分钟 | 高 | 英文 |
| 通义万相音乐 | 闭源 | 60 秒 | 高 | 支持 |

### 4.4 音效生成

```python
# AudioLDM 文本到音效
from diffusers import AudioLDMPipeline

pipe = AudioLDMPipeline.from_pretrained(
    "cvssp/audioldm-s-full-v2",
)
audio = pipe("雨声和雷声").audios[0]
# audio: (time,) numpy array, sample_rate=16000
# (可保存或播放)
```

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **Suno / Udio**：2024-2025 年最流行的 AI 音乐生成平台。Suno v4 可以生成带歌词的完整歌曲，时长可达 4 分钟。底层使用基于 Codec 的自回归 Transformer + 扩散模型的混合架构。
- **ChatGPT 的语音模式**：GPT-4o 支持实时语音对话。它使用了一个类似 F5-TTS 的零样本 TTS 模型，能根据用户输入的语气来调整回复的语气。
- **Meta 的 MusicGen**：基于 EnCodec token 的自回归模型。支持文本到音乐、旋律参考到音乐的生成。

### 5.2 大语言模型时代什么变了？

**音频 token 的统一范式**是 2024-2026 年最大的变化。以前 TTS（语音合成）、音乐生成、音效生成使用完全不同的模型架构。现在它们统一在同一个架构下：Codec（音频→token）+ Transformer（token 预测）+ Codec（token→音频）。

这意味着当你学会了一个领域的音频生成，就能快速迁移到其他领域。它也与文本生成共享 Transformer 基石——唯一区别是输入空间从文本变成了音频 token。

### 5.3 什么没变？

音频 token 的量化方法（RVQ）自 EnCodec 2022 年以来没有根本性变化。变化的只是码本大小、量化器数量和训练数据规模。TTS 对自然度的追求也从未改变——2026 年的 F5-TTS 生成的语音仍然不能完全替代人类录音，但对于大多数应用场景已经足够。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的语音模式说话时，音频流被实时处理为 token 序列，LLM 理解这些 token（转义为语义），生成回复 token，再由 TTS 模型将回复 token 转为语音。音频生成的延迟决定了你的"对话流畅度"。这就是为什么更高效的自回归模型和扩散模型对用户体验至关重要——延迟必须低于 300 毫秒才能达到自然对话的体验。

---

## 6. 工程最佳实践

### 6.1 TTS 场景选型

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 中文 TTS | CosyVoice / F5-TTS | 中文优化，质量高 |
| 英文 TTS | Kokoro / MetaVoice | 速度快，质量好 |
| 实时语音助手 | F5-TTS / CosyVoice | 低延迟，零样本克隆 |
| 有声书 | CosyVoice | 多音色，情感丰富 |
| 多语言 | F5-TTS | 零样本支持多语言 |

### 6.2 提示词策略

- **MusicGen**：描述风格 + 乐器 + 节奏 + 情绪
  - "80年代合成波，重低音，旋律钢琴，BPM 120"
- **AudioLDM**：描述环境和动作
  - "雨声和远处的雷声，自然环境音"
- **TTS**：描述说话风格 + 语速 + 情感
  - "温和的语调，缓慢的语速，带一点快乐的语气"

### 6.3 中文场景特别建议

- CosyVoice 是当前中文 TTS 的最佳选择，支持粤语和普通话
- F5-TTS 的 Flow Matching 架构对中文的声调建模优于传统 TTS
- MusicGen 对中文提示词理解有限，建议用英文写提示词

### 6.4 踩坑经验

- **生成的音频有爆音**：检查音频的增益，输出前应用峰值归一化
- **TTS 发音不准确**：专业名词（如 API、NFT）需要拼写为中文（"啊批爱"而非"API"）
- **音乐生成缺乏结构**：音乐需要前奏、主歌、副歌、结尾的结构信息，写提示词时明确指定

---

## 7. 常见错误

### 错误 1：混淆音频采样率和模型期望的采样率

**现象：** 生成的音频音高异常（像慢放或快进），或者直接输出噪声。

**原因：** 不同模型期望不同的采样率。MusicGen 期望 32kHz，AudioLDM 期望 16kHz，如果输入不匹配的采样率，模型会误解频率范围。

**修复：**

```python
# ❌ 错误：直接保存，不考虑采样率
torchaudio.save("output.wav", audio_values[0], sample_rate=22050)  # 不匹配模型的 32000

# ✓ 正确：使用模型期望的采样率
torchaudio.save("output.wav", audio_values[0],
                sample_rate=model.config.audio_encoder.sampling_rate)  # 32000
```

### 错误 2：TTS 时输入包含特殊符号

**现象：** 生成语音中出现异常的长停顿或奇怪的重音。

**原因：** TTS 模型的文本前端没有做输入清洗。表情符号、URL、特殊字符可能导致分词器输出异常。

**修复：**

```python
import re

def clean_tts_input(text):
    """清洗 TTS 输入文本。"""
    # ❌ 错误：直接使用原始输入
    # text = "这篇文章在 https://example.com 上！😊"

    # ✓ 正确：清洗特殊字符
    text = re.sub(r'https?://\S+', '网址', text)  # URL → "网址"
    text = re.sub(r'[😊🎵🔥]', '', text)           # 去除表情符号
    return text
```

### 错误 3：在没有 GPU 的情况下运行音频生成模型

**现象：** 生成一段 10 秒的音频需要 10 分钟。

**原因：** 音频生成模型（尤其是 MusicGen 和 F5-TTS）的 Transformer 部分需要 GPU 加速。

**修复：**

```python
# ❌ 错误：没有指定设备
model = MusicGenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# ✓ 正确：明确指定 GPU
model = MusicGenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model = model.to("cuda")  # 或 "mps"（Apple Silicon）
```

---

## 8. 面试考点

### Q1：为什么神经编解码器（Neural Codec）在音频生成中如此重要？（难度：⭐⭐）

**参考答案：**
神经编解码器将连续的音频波形压缩为离散的 token 序列，这有三大好处：(1) 音频被转换为与文本相同的 token 空间，使得 Transformer（原本为文本设计）可以直接用于音频生成；(2) 音频生成从连续空间回归问题变为离散空间分类问题，训练更稳定；(3) 压缩大幅降低计算量——44.1kHz 的原始波形每秒 44100 个采样点，Codec token 每秒约 50-100 个 token，降低 440-880 倍。EnCodec 使用残差向量量化（RVQ）实现多层次的音频压缩。

### Q2：F5-TTS 的零样本能力是如何实现的？与传统 TTS 有什么不同？（难度：⭐⭐⭐）

**参考答案：**
传统 TTS（如 Tacotron、VITS）需要大量"文本-语音"对齐训练数据，通常需要 10 小时以上的录音。F5-TTS 采用 Flow Matching 在两个目标上训练：(1) 从噪声到语音梅尔谱图的去噪过程；(2) 以 3-5 秒的参考音频（称为"prompt"）作为条件信号。推理时用户提供一段参考音频，F5-TTS 从中提取说话者的音色特征（Speaker Embedding），然后在生成时"模仿"这个音色。这绕过了传统 TTS 最难的部分——音素对齐——因为在流匹配中，对齐是隐式学习的。

### Q3：自回归 Transformer 和扩散/Flow Matching 在音频生成中各有什么优劣势？（难度：⭐⭐⭐）

**参考答案：**
自回归 Transformer 的优点：(1) 天然适合序列建模——音乐和语音都有长程结构依赖；(2) 容易加入文本条件——交叉注意力直接将文本嵌入注入 Transformer；(3) 可控性强——可以调节 temperature 控制多样性，可以设置 repetition penalty 避免重复。

扩散/Flow Matching 的优点：(1) 生成速度快——自回归需要逐 token 生成，扩散可以并行采样；(2) 训练简单——MSE 损失代替了自回归的交叉熵损失；(3) 对短音频（如音效）特别高效——音效没有复杂的序列依赖，扩散几秒就能生成。

2025-2026 年的趋势是混合架构——用自回归做低层次的 token 序列建模（大结构），用扩散做高层次的 token 细化（细节）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 神经编解码器 (Codec) | "音频的压缩器" | 将连续的音频波形压缩为离散 token 序列的神经网络——类似音频的"分词器" |
| RVQ（残差向量量化） | "多层压缩" | 先用第一级码本做粗量化，后续级别逐级量化残差——实现从低比特到高比特的可扩展音频编码 |
| Flow Matching | "流匹配" | 扩散模型的一个变体——将去噪过程建模为从噪声到数据的连续流，而非离散步的扩散过程 |
| 零样本 TTS | "3 秒克隆声音" | 无需在目标说话者数据上微调，仅凭 3-5 秒参考音频就能克隆其音色和说话风格的 TTS 模型 |
| MusicGen / AudioCraft | "用 Transformer 做音乐" | Meta 开源的音频生成框架——EnCodec token + Transformer 自回归 + 文本条件 |
| 梅尔谱图 (Mel-Spectrogram) | "音频的 2D 图像" | 将音频信号转换为时间-频率的 2D 表示——横轴时间、纵轴频率、颜色为强度 |

---

## 📚 小结

音频生成的统一范式是"Codec + Transformer"：神经编解码器将音频压缩为离散 token 序列，Transformer 在 token 空间做自回归或扩散生成。TTS、音乐、音效共享同一底层架构。F5-TTS 用 Flow Matching 实现零样本语音合成。2026 年的趋势是混合架构——自回归负责结构，扩散负责细节。音频生成的统一 token 范式使之与 LLM 高度融合——同一个模型"说话"、"唱歌"、"配乐"成为可能。

---

## ✏️ 练习

1. **【理解】** 类比"分词器（Tokenizer）将文本转为整数序列"，说明"神经编解码器（Codec）将音频转为 token 序列"的工作过程。你的解释应该让一个理解 LLM 但不了解音频的人能听懂。

2. **【实现】** 修改 `NeuralAudioCodecConcept` 类，加入第二级残差量化（RVQ）。第一级量化主结构，第二级量化第一级的残差，实现两层量化。

3. **【实验】** 使用 CosyVoice 或 F5-TTS，录制 5 秒自己的语音作为参考，然后让模型生成 3 段不同文本的语音。对比生成语音与原始语音在音色上的相似度。

4. **【思考】** 音频 token 可以像文本 token 一样直接输入到大语言模型中进行"音频理解"（如情感分析、说话者识别）。如果统一了音频和文本的 token 空间，你认为会出现什么样的新应用？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| 神经编解码器概念实现 | `code/audio_codec.py` | 简化版音频编解码器（RVQ 演示） |
| 音频生成 Transformer | `code/audio_lm.py` | 在 Codec token 上做 Transformer 生成的实现 |
| TTS/音乐提示词模板 | `outputs/audio-prompt-guide.md` | 面向中文用户的 TTS/音乐生成提示词模板 |

---

## 📖 参考资料

1. [论文] Défossez et al. "EnCodec: High Fidelity Neural Audio Compression". Meta, 2022. https://arxiv.org/abs/2210.13438
2. [论文] Copet et al. "Simple and Controllable Music Generation". NeurIPS, 2023. https://arxiv.org/abs/2306.05284
3. [论文] Chen et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Text-to-Speech with Flow Matching". arXiv, 2024. https://arxiv.org/abs/2410.04085
4. [GitHub] Meta AudioCraft: https://github.com/facebookresearch/audiocraft
5. [GitHub] Hugging Face transformers: https://github.com/huggingface/transformers
6. [GitHub] SWivid/F5-TTS: https://github.com/SWivid/F5-TTS

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
