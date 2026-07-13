# 音频 Transformer——Whisper

> Whisper 用 680,000 小时的多语言弱监督数据训练了一个编码器-解码器 Transformer。一个架构、多个任务（转录/翻译/时间戳）、99 种语言、笔记本运行——2026 年的参考 ASR。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 06 · 04（ASR）、阶段 07 · 05（完整 Transformer）
**时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 11（混合专家模型）— 对比 Transformer 在音频和文本领域的不同架构选择

---

## 🎯 学习目标

- [ ] 理解 Whisper 如何将音频编码为 log-mel 频谱图再编码为 token
- [ ] 区分 Whisper 编码器（双向 Transformer）和解码器（自回归 Transformer）的作用
- [ ] 说明 Whisper 的"多任务"设计——同一个模型做转录、翻译和时间戳

---

## 1. 问题

ASR 的核心困难是**音频帧与文字之间没有对齐**——"okay" 可能是 200ms 也可能是 1200ms。CTC、RNN-T、Attention 是三种解法。Whisper 选择了 **Attention encoder-decoder**——编码器理解音频，解码器自回归生成文字。

Whisper 用 680,000 小时的弱监督数据训练（大部分来自互联网）——不需要精心标注的平行语料。这种"用噪声数据换规模"的范式是 LLM 时代的核心思路。

---

## 2. 概念

### 2.1 Whisper 架构

```
30 秒 log-mel 频谱图 (80 mel × 3000 帧)
    ↓ 两层 stride-2 卷积（降采样 4x）
    ↓ 24 层 Transformer 编码器
    ↓ 编码器隐藏状态
    ↓ 24 层 Transformer 解码器（因果掩码 + 交叉注意力）
    ↓ 自回归生成：BOS → token1 → token2 → ... → EOS
```

### 2.2 多任务——一个模型做三件事

| 任务 | 启动 token | 用途 |
|---|---|---|
| 转录（语言 X） | `<\|startoftranscript\|><\|lang_X\|><\|transcribe\|>` | 语音识别 |
| 翻译（X→英文） | `<\|startoftranscript\|><\|lang_X\|><\|translate\|>` | 语音翻译 |
| 带时间戳转录 | `<\|startoftranscript\|><\|lang_X\|><\|transcribe\|>` | 字幕生成 |

### 2.3 Whisper 模型家族

| 模型 | 层数 | 参数量 | WER（LibriSpeech test-clean） |
|---|---|---|---|
| tiny | 4 | 39M | 7.6% |
| base | 6 | 74M | 4.9% |
| small | 12 | 244M | 3.5% |
| medium | 24 | 769M | 2.7% |
| **large-v3** | 32 | 1.5B | **1.8%** |
| **large-v3-turbo** | 32 enc + 4 dec | 809M | **1.6%** |

---

## 3. 从零实现

完整代码见 `code/main.py`——纯 NumPy，模拟了 Whisper 的编码器-解码器架构。

```python
# 编码器：log-mel → Transformer → 隐藏状态
enc_output = whisper.encode(mel_spectrogram)  # (帧数, d_model)

# 解码器：词元 + 交叉注意力 → 下一个词元
logits = whisper.decode(token_ids, enc_output)  # (seq_len, vocab_size)
```

---

## 4. 工业工具

### 4.1 OpenAI Whisper Python 包

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 转录音频
result = model.transcribe("audio.mp3")
print(result["text"])

# 翻译（中文→英文）
result = model.transcribe("chinese.mp3", task="translate")
```

### 4.2 faster-whisper（加速版）

```python
from faster_whisper import WhisperModel

# 加载加速版模型
model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 4.3 性能对比

| 实现 | 速度 | 精度 | 内存 |
|---|---|---|---|
| whisper（原版）| 基准 | 100% | 基准 |
| faster-whisper | 4x 加速 | 100% | 50% |
| whisper.cpp | 3x 加速 | 100% | 30% |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

Whisper 是 OpenAI 的语音识别模型，它不直接是"大语言模型"，但它与大语言模型的联系非常紧密：

- GPT-4 的语音模式使用 Whisper 做语音转文本
- Gemini、Qwen-Audio 等多模态模型将 Whisper 风格的音频编码器与大语言模型结合

### 5.2 LLM 时代什么变了？

**从专用模型到多模态。** Whisper 最初是独立的 ASR 模型。现在它是多模态大语言模型的"耳朵"——将语音转换为文本，然后由大语言模型处理。

**弱监督训练范式。** Whisper 用 680K 小时的弱监督数据训练——大部分来自互联网视频的字幕。这种"用噪声数据换规模"的范式被大语言模型广泛采用。

### 5.3 什么没变？

**编码器-解码器架构没变。** Whisper 使用经典的编码器-解码器 Transformer——编码器理解音频，解码器生成文字。

**交叉注意力没变。** 解码器的交叉注意力让每一步生成都可以"回看"编码器的全部音频表示。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中使用语音输入时，模型使用 Whisper 将语音转换为文本，然后大语言模型处理文本。这就是多模态大语言模型的工作原理。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 实时转录 | Whisper-tiny / base | 速度优先 |
| 高质量转录 | Whisper-large-v3 | 精度优先 |
| 长音频 | VAD + Whisper 分块 | 避免幻觉 |
| 生产环境 | faster-whisper + ONNX | 加速 + 量化 |

### 6.2 中文场景特别建议

- Whisper 对中文有专门的 `<|zh|>` 标记——确保指定语言
- 使用 VAD 门控避免静默产生幻觉

### 6.3 踩坑经验

- 长音频必须分块处理——Whisper 只接受 30 秒音频
- 静默片段会导致幻觉——先用 VAD 过滤
- 不同语言的 WER 差异很大——中文通常比英文差

---

## 7. 常见错误

### 错误 1：长音频未分块处理

**现象：** 模型在 30 秒后的音频上输出异常或重复。

**原因：** Whisper 最多处理 30 秒音频。更长的音频需要分块处理。

**修复：**
```python
# ❌ 直接处理长音频
result = model.transcribe("long_audio.mp3")  # 30秒后输出异常

# ✓ 使用 VAD + 分块
segments = get_voice_segments(audio)  # 先检测语音段
for segment in segments:
    result = model.transcribe(segment)  # 每段单独处理
```

### 错误 2：未指定语言导致识别错误

**现象：** 中文语音被识别为英文。

**原因：** Whisper 的自动语言检测可能错误。

**修复：**
```python
# ❌ 依赖自动检测
result = model.transcribe("chinese.mp3")  # 可能误判语言

# ✓ 显式指定语言
result = model.transcribe("chinese.mp3", language="zh")
```

---

## 8. 面试考点

### Q1：Whisper 如何处理音频的变长特性？（难度：⭐⭐）

**参考答案：**
Whisper 使用 30 秒的固定窗口——将音频填充或截断到 30 秒。log-mel 频谱图固定为 80 mel × 3000 帧。超过 30 秒的音频需要分块处理。

### Q2：Whisper 的多任务设计是如何工作的？（难度：⭐⭐⭐）

**参考答案：**
Whisper 使用特殊 token 控制任务类型。解码器的前几个 token 决定任务：
- `<|transcribe|>` → 转录
- `<|translate|>` → 翻译（任何语言→英文）
- `<|lang_X|>` → 指定语言

同一个模型、同一套参数，通过不同的启动 token 切换任务。

### Q3：弱监督训练为什么有效？（难度：⭐⭐⭐）

**参考答案：**
Whisper 用 680K 小时的互联网视频字幕训练——大部分是自动对齐的，有噪声。弱监督有效的原因：
1. 规模优势：680K 小时比人工标注数据集大 100 倍
2. 噪声鲁棒：Transformer 的自注意力机制对噪声有容忍度
3. 泛化能力：多语言、多领域的弱监督数据帮助模型学到更通用的语音表示

### Q4：Whisper 和 CTC 模型有什么区别？（难度：⭐⭐）

**参考答案：**
CTC（连接时序分类）直接从音频帧预测文本——没有解码器，一步预测。Whisper 使用编码器-解码器架构——编码器理解音频，解码器自回归生成文本。

### Q5：VAD 门控如何避免 Whisper 的静默幻觉？（难度：⭐⭐⭐）

**参考答案：**
静默幻觉：Whisper 在静默片段会"凭空"生成文本——因为它试图将"噪声"解释为语音。

VAD 门控：在 Whisper 前加语音活动检测（VAD）——只将语音段送入 Whisper。静默片段直接跳过，避免幻觉。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Log-mel 频谱图 | "音频图像" | 将音频转换为频率-时间表示——作为 Transformer 的输入 |
| 弱监督训练 | "用噪声数据训练" | 用不完美（噪声）的网络数据训练——不需要人工标注 |
| 束搜索解码 | "选最好的句子" | 解码时保持 top-K 个候选序列，选择全局最优 |
| VAD 门控 | "过滤静默" | 语音活动检测——过滤静默避免幻觉 |
| WER | "字错率" | Word Error Rate——衡量 ASR 准确率的标准指标 |
| Mel 频率 | "人耳听觉" | 模拟人耳听觉的频率尺度——低频分辨率高，高频分辨率低 |
| CTC | "连接时序分类" | 直接从音频帧预测文本——没有解码器的 ASR 方法 |
| 交叉注意力 | "解码器看编码器" | 解码器每一步关注编码器的所有位置——生成时可以"回看"音频 |

---

## 📚 小结

Whisper = Log-mel 频谱图 + 24 层 Transformer 编码器 + 24 层解码器。680K 小时弱监督数据训练。一个模型做转录、翻译、时间戳。2026 年的基准：large-v3-turbo 1.6% WER（LibriSpeech）。关键实践：VAD 门控避免静默幻觉，分块处理长音频。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 Whisper 如何将音频当作文本处理。写 200 字以内的说明。

2. **【实现】** 用 Whisper 在 10 句中文语音上测试，记录 WER 和是否有静默幻觉。

3. **【实验】** 对比 Whisper-tiny 和 Whisper-large-v3 的推理延迟和准确率——找出质量-速度的最优解。

4. **【实现】** 实现一个简化的音频编码器（在 code/main.py 中），验证 log-mel 频谱图到词元的转换流程。

5. **【思考】** 阅读 Whisper 论文的摘要，用你自己的话解释弱监督训练为什么有效。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| Whisper 编码器-解码器实现 | `code/main.py` | 模拟 Whisper 的音频编码和文本生成 |
| Whisper 使用指南 | `outputs/whisper-guide.md` | 安装、使用和优化指南 |

---

## 📖 参考资料

1. [论文] Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper). 2022.
2. [代码] openai/whisper. https://github.com/openai/whisper
3. [论文] Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS, 2022.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
