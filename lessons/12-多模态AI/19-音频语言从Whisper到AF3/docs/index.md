# 音频-语言模型：从 Whisper 到 Audio Flamingo 3

> Whisper（2022 年 12 月）解决了语音识别——68 万小时弱监督多语言语音、简单的编码器-解码器 Transformer。但识别不等于推理。问"这段录音里有什么乐器"、"说话人的情绪是什么"、"第 3 分钟发生了什么"需要音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这个技术栈：保留 Whisper 级别的编码器，加上 Q-Former，在音频-文本指令数据上训练，加入思维链推理。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 06（语音与音频）、阶段 12 · 03（Q-Former）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从波形计算 log-Mel 频谱图——窗口化、FFT、滤波器组、对数变换
- [ ] 解释 Whisper 编码器如何将音频转换为特征序列
- [ ] 描述 AF3 如何结合 Whisper 编码器和 Q-Former 实现音频理解
- [ ] 对比不同音频-语言模型的架构差异

---

## 1. 问题

Whisper 做语音转文本——68 万小时弱监督数据，简单的编码器-解码器架构。但它不能"理解"音频——只能转录。问"这段音乐用了什么乐器"、"说话人是高兴还是难过"、"第 3 分钟发生了什么"需要音频理解能力。

从 Whisper 到 AF3 的演进：保留 Whisper 编码器 + 加上 Q-Former + 在音频-文本指令数据上训练 + 加入思维链推理。

---

## 2. 概念

### 2.1 从波形到特征

```
原始波形 (16kHz, 16-bit)
    ↓ 窗口化 (25ms 窗口, 10ms 步长)
    ↓ FFT (快速傅里叶变换)
    ↓ Mel 滤波器组 (80 个频带)
    ↓ 对数变换
    → log-Mel 频谱图 (T, 80)
    ↓ Whisper 编码器
    → 音频特征 (T', D)
```

### 2.2 Whisper 架构

```
音频波形 → [log-Mel 频谱图] → [Whisper 编码器] → 音频特征
                                                    ↓
文本: "转录这段音频" → [Whisper 解码器] → "Hello world"
```

Whisper 是编码器-解码器架构——编码器处理音频，解码器生成文本。

### 2.3 音频理解模型的演进

| 模型 | 年份 | 突破 |
|------|------|------|
| Whisper | 2022 | 语音转文本——68 万小时弱监督 |
| Qwen-Audio | 2023 | 音频理解——Q-Former + 指令微调 |
| SALMONN | 2023 | 音频问答——多音源理解 |
| Audio Flamingo 3 | 2025 | 音频理解+推理——思维链 |

### 2.4 AF3 的架构

```
音频 → [Whisper 编码器] → 音频特征
                              ↓
                        [Q-Former] → 视觉词元（32 个）
                              ↓
                        [LLM] → 文本回答（带思维链）
```

AF3 保留了 Whisper 编码器（冻结），加上 Q-Former 提取音频特征，然后用 LLM 生成文本回答。

---

## 3. 从零实现

### Step 1：log-Mel 频谱图计算

```python
import torch
import torch.nn.functional as F

def log_mel_spectrogram(waveform, n_mels=80, n_fft=400, hop_length=160, sample_rate=16000):
    """计算 log-Mel 频谱图。"""
    # 窗口化 + FFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                       window=window, return_complex=True)
    magnitude = torch.abs(stft)

    # Mel 滤波器组
    mel_basis = torch.randn(n_mels, n_fft // 2 + 1)  # 简化
    mel_spec = torch.mm(mel_basis, magnitude ** 2)

    # 对数变换
    log_mel = torch.log(mel_spec + 1e-9)
    return log_mel


def compute_whisper_features(waveform, n_mels=80, sample_rate=16000):
    """从波形计算 Whisper 特征。"""
    log_mel = log_mel_spectrogram(waveform, n_mels=n_mels, sample_rate=sample_rate)
    return log_mel.permute(0, 2, 1)  # (B, T, n_mels)
```

### Step 2：音频理解查询

```python
def audio_understanding_query(model, audio_features, query_text):
    """音频理解查询。"""
    # 用 Q-Former 提取音频词元
    audio_tokens = model.qformer(audio_features)
    # 用 LLM 生成回答
    prompt = f"音频内容：{audio_tokens}\n问题：{query_text}\n回答："
    response = model.llm.generate(prompt)
    return response
```

---

## 4. 工具

### 4.1 HuggingFace

```python
from transformers import AutoModel, AutoProcessor

# Qwen-Audio
model = AutoModel.from_pretrained("Qwen/Qwen-Audio")
processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio")
```

### 4.2 Whisper

```python
from transformers import WhisperForConditionalGeneration

whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
```

---

## 6. 工程最佳实践

### 6.1 音频预处理

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 采样率 | 16kHz | Whisper 标准 |
| n_mels | 80 | Mel 滤波器组数量 |
| 窗口大小 | 25ms | 帧分析窗口 |
| 步长 | 10ms | 帧移 |

### 6.2 踩坑经验

- **采样率不匹配**：Whisper 期望 16kHz，其他模型可能期望不同采样率
- **音频太短**：短于 1 秒的音频可能特征提取不充分

---

## 7. 常见错误

### 错误 1：混淆音频转录和音频理解

**现象：** 问"说话人的情绪是什么"但模型只回答转录内容。

**原因：** 只用了 Whisper 编码器——没有理解能力。

**修复：** 使用 Q-Former 或类似桥接网络将 Whisper 特征转换为 LLM 可理解的词元。

---

## 8. 面试考点

### Q1：Whisper 和 Qwen-Audio 的区别是什么？（难度：⭐⭐）

**参考答案：**
Whisper 是语音转文本模型——将音频波形转换为文本序列。Qwen-Audio 是音频理解模型——可以回答关于音频内容的问题（情绪、乐器、事件）。Whisper 只做识别，Qwen-Audio 做理解。AF3 在此基础上加入了思维链推理——不仅能描述音频，还能解释为什么。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Whisper | "语音识别模型" | 68 万小时弱监督训练的编码器-解码器 Transformer——语音转文本 |
| Audio Flamingo 3 | "音频理解" | 保留 Whisper 编码器 + Q-Former + LLM 推理 |
| log-Mel 频谱图 | "音频的 2D 图像" | 将波形转换为时间-频率的对数表示——音频特征提取的基础 |

---

## 📚 小结

从 Whisper（转录）到 AF3（理解+推理），音频-语言模型逐步增强能力。核心：保留 Whisper 编码器 + Q-Former 提取特征 + LLM 生成回答。AF3 加入了思维链推理——能解释"为什么"而不仅仅是"是什么"。

---

## ✏️ 练习

1. **【实现】** 从波形计算 log-Mel 频谱图——验证形状和值范围
2. **【对比】** 对比 Whisper 转录和 Qwen-Audio 理解在同一段音频上的输出差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 音频特征提取 | `code/main.py` | log-Mel 频谱图 + Whisper 特征计算 |

---

## 📖 参考资料

1. [论文] Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper). arXiv, 2022.
2. [论文] Chen et al. "Qwen-Audio: Advancing Universal Audio Understanding". arXiv, 2023.
3. [论文] NVIDIA. "Audio Flamingo 3: Audio Understanding with Chain-of-Thought". arXiv, 2025.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
