# 频谱图、梅尔尺度与音频特征

> 神经网络不擅长消费原始波形。它们消费频谱图。消费梅尔频谱图更好。2026 年的每一个 ASR、TTS 和音频分类器都因为这一单个预处理选择而存亡。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 01（音频基础） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零构建 STFT——分帧、加窗、FFT 堆叠成时频矩阵——理解为什么 25ms 帧 + 10ms 帧移是 ASR 的标准参数
- [ ] 实现梅尔滤波器组——理解梅尔尺度如何对齐人类的对数音高感知，80 个梅尔 bin 覆盖 0-8kHz
- [ ] 区分 log-mel、dB-mel、MFCC 三种特征——知道 Whisper、Parakeet、ECAPA 各自吃什么

---

## 1. 问题

取一段 10 秒 16kHz 的音频——160,000 个浮点数。它们与标签"狗叫"或"单词 cat"几乎完全不相关。原始波形有信息，但格式不对——同一个音素相隔 100ms 说出的原始采样完全不同。

频谱图修复了这一点。它在人类感知忽略的时间细节（微秒抖动）上做坍缩，在人类感知关注的结构上做保留——哪些频率有能量，时间窗口约 10-25ms。

梅尔频谱图推得更远。人类对音高的感知是对数的：100 Hz vs 200 Hz 听起来"距离一样远"，就像 1000 Hz vs 2000 Hz。梅尔尺度将频率轴扭曲到匹配这种对数感知。梅尔频谱图是从 2010 到 2026 年语音 ML 最重要的单个特征。

---

## 2. 概念

### 2.1 STFT——从波形到频谱图

STFT = 分帧 → 加窗 → FFT → 堆叠幅度谱

| 参数 | 典型值（ASR） | 为什么 |
|---|---|---|
| 帧长 | 25 ms (400采样@16kHz) | 捕获 1-2 个音素周期 |
| 帧移 | 10 ms (160采样@16kHz) | 2.5 倍过采样——保证连续覆盖 |
| 窗函数 | Hann 窗 | 将帧两端平滑衰减到零，消除频谱泄漏 |

### 2.2 对数幅度

原始幅度跨越 5-6 个数量级。`log(|X| + 1e-6)` 压缩动态范围。**每个生产流水线都用对数幅度。**

### 2.3 梅尔尺度——人耳的对数频率

$$m = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$

梅尔尺度在 ~1kHz 以下近似线性，~1kHz 以上近似对数。**80 个梅尔 bin 覆盖 0-8kHz 是标准 ASR 输入。**

梅尔滤波器组：在梅尔尺度上等间隔排列的一组三角滤波器。每个滤波器是相邻 FFT bin 的加权求和。STFT 幅度乘以滤波器组矩阵 = 梅尔频谱图。

### 2.4 对数梅尔频谱图——2026 的标准前端

Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入。**`log(mel_spec + 1e-10)`**。如果你不用音乐——从 80 对数梅尔开始。偏离这个默认值需要你有充分的理由。

### 2.5 MFCC——旧时代的王牌

对对数梅尔频谱图施加 DCT（类型 II），保留前 13 个系数。解相关并进一步压缩。直到 ~2015 年都是主导特征——然后 CNN/Transformer 在原始 log-mel 上追上来了。在说话人识别（x-vector、ECAPA）中仍然使用。

### 2.6 分辨率权衡

| 帧长 / 帧移 | 适用场景 |
|---|---|
| 25ms / 10ms | **ASR 默认** |
| 50ms / 12.5ms | 音乐 |
| 5ms / 2ms | 瞬态检测（鼓点、爆破音） |

更大的 FFT = 更好的频率分辨率但更差的时间分辨率。ASR 选择了 25ms/10ms 的甜点。

---

## 3. 从零实现

### Step 1: 分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

0.5 秒 8kHz 信号 + frame_len=256 + hop=128 → 约 31 帧。

### Step 2: Hann 窗

```python
def hann(N):
    """将帧两端平滑衰减到零——消除 FFT 的频谱泄漏。"""
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

### Step 3: STFT 幅度

```python
def stft_magnitude(signal, frame_len, hop):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [dft_mag([w * f for w, f in zip(win, frame)]) for frame in frames]
```

生产中使用 `torch.stft` 或 `librosa.stft`（FFT 加速、向量化）。

### Step 4: 梅尔滤波器组

```python
def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    """梅尔尺度：f=700Hz→m=1000mel, f=7000Hz→m≈3000mel。"""
    if fmax is None: fmax = sr / 2
    m_lo, m_hi = hz_to_mel(fmin), hz_to_mel(fmax)
    mels = [m_lo + (m_hi - m_lo) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(round(h * n_fft / sr)) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            fb[m][k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[m][k] = (right - k) / max(1, right - center)
    return fb
```

梅尔滤波器组的三角形状——低频滤波器窄（密集）、高频滤波器宽（稀疏）——**精确模拟了人耳在低频区域有更高分辨率的听觉特性**。

### Step 5: 对数梅尔频谱图

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

### Step 6: MFCC

```python
def dct_ii(x, n_coeffs):
    """DCT 类型 II——MFCC 的核心。第 0 个系数编码总能量——通常下游丢弃。"""
    N = len(x)
    return [sum(x[n] * math.cos(math.pi * k * (2*n+1) / (2*N)) for n in range(N))
            for k in range(n_coeffs)]
```

完整代码见 `code/main.py`——纯 stdlib，可立即运行。

---

## 4. 工业工具

| 任务 | 特征 |
|---|---|
| ASR（Whisper, Parakeet, SeamlessM4T） | 80 log-mels, 10ms 帧移, 25ms 帧长 |
| TTS 声学模型（VITS, F5-TTS, Kokoro） | 80 mels, 5-12ms 帧移 |
| 音频分类（AST, PANNs, BEATs） | 128 log-mels, 10ms 帧移 |
| 说话人嵌入（ECAPA, WavLM） | 80 log-mels 或原始波形 SSL |
| 音乐（MusicGen, Stable Audio 2） | EnCodec 离散 token（非梅尔） |
| 关键词检测 | 40 MFCCs（小设备） |

**经验法则：不在做音乐就用 80 log-mel。偏离这个默认值需要充分理由。**

---

## 5. 常见错误

### 错误 1：梅尔数量训练/推理不匹配

**现象：** 模型训练用 80 梅尔，推理时输入 128 梅尔——静默失败，输出垃圾但不报错。

**原因：** 嵌入层/线性层的输入维度不匹配——PyTorch 报维度错误，但如果用的是静态导出模型（ONNX/TensorRT），维度错误可能被静默处理。

**修复：** 在训练和推理两端都打印特征形状并断言。永远不要"凭感觉"改梅尔数量。

### 错误 2：dB-mel 误用为 log-mel

**现象：** 用 `10 * log10(power + eps)` 代替 `log(mel + eps)`——Whisper/Parakeet 表现退化。

**原因：** dB-mel 使用参考归一化——特征分布与 log-mel 不同。Whisper 在 log-mel 上训练——传入 dB-mel = 静默的特征不匹配。

**修复：** 用 `np.log(mel + 1e-10)` 或 `np.log10(mel + 1e-10) * 20`，不是 `librosa.power_to_db()`。仔细检查 Whisper 的 `log_mel_spectrogram` 函数。

### 错误 3：零填充产生的虚假频谱

**现象：** 音频末尾出现不自然的高频伪影。

**原因：** 零填充音频末尾——在最后几帧中产生阶跃不连续——FFT 在这些帧上产生高频能量（混叠效应的时域版本）。

**修复：** 对称填充或复制填充，不使用零填充：

```python
# ❌ 零填充
padded = signal + [0.0] * pad_len

# ✓ 复制填充（镜像尾部）
padded = signal + signal[-pad_len:][::-1]
```

### 错误 4：采样率不匹配在特征提取之前

**现象：** 用 22.05kHz 训练的特征——推理时输入 16kHz 音频——梅尔频谱图的频率分布不同——模型表现退化。

**原因：** 梅尔滤波器组是基于采样率构建的——22.05kHz 和 16kHz 的梅尔频率轴不同。如果训练用 22.05kHz、推理用 16kHz——同一个词的梅尔频谱图在两个频率轴上看起来完全不同。特征提取在音频重采样之后进行——如果重采样漏了或做了两次——梅尔频率轴就会错位。

**修复：** 先重采样到目标频率，再做特征提取——顺序不可交换。训练和推理使用完全相同的采样率 → 重采样工具 → 特征提取参数。

### 错误 5：归一化漂移

**现象：** 训练时按句归一化，推理时按全局归一化——WER 翻倍。

**原因：** 训练时，每句话独立归一化（减均值/除以标准差），让每句话的特征分布一致。推理时如果用了全局统计量——不同句子的特征尺度不同——模型看到的分布偏离了训练分布。

**修复：** 训练和推理使用完全相同的归一化策略。按句归一化是 ASR 最安全的选择——不依赖任何预计算的全局统计量。

---

## 6. 面试考点

### Q1：梅尔滤波器组为什么在低频区域密集、高频稀疏？（难度：⭐⭐）

**参考答案：**
因为人耳的音高感知是对数的。100Hz→200Hz（100Hz差）和 1000Hz→2000Hz（1000Hz差）在人耳听起来"距离一样远"。梅尔滤波器组在梅尔尺度上等距分布——梅尔尺度在低频区域密集（100Hz-1kHz 只覆盖约 1000 mel），高频区域稀疏（1kHz-8kHz 覆盖约 2000 mel）。这种排列方式让特征空间与人类听觉的分辨能力对齐——人类在低频有更高分辨率，滤波器组在低频也更密集。

### Q2：为什么 MFCC 被 log-mel 取代了？（难度：⭐⭐）

**参考答案：**
MFCC = DCT(log-mel)——做了两件事：解相关和压缩维度（80 mel → 13 个系数）。DCT 压缩的前提是信号在频率维度上有强相关性——这对传统高斯混合模型有用（GMM 的协方差矩阵在独立特征上更简单）。但 CNN/Transformer 可以直接处理相关特征——解相关这一步不仅无用，还丢失了有价值的频谱结构信息。MFCC 在说话人识别中仍有用——因为说话人嵌入模型（如 ECAPA-TDNN）需要低维、解相关的特征来高效建模个体差异。

### Q3：给定一个未知来源的音频文件，如何判断它是否符合 Whisper 的输入要求？（难度：⭐⭐⭐）

**参考答案：**
四步检查：**（1）采样率**——Whisper 要求 16kHz。读取 `soundfile.info(path).samplerate`。如果不是 16kHz → 重采样（`torchaudio.transforms.Resample`，带抗混叠）。**（2）声道数**——Whisper 要求单声道。立体声需折叠为单声道（`.mean(dim=0)`）。**（3）数据类型**——Whisper 要求 float32。int16 需要除以 32768.0 转换。**（4）特征维度**——如果你在自己生成特征（而非用 Whisper 的 `log_mel_spectrogram`）——确认是 80 维梅尔而非 128 维。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 帧 (Frame) | "音频的一个切片" | 25ms 的波形片段，交给一次 FFT |
| 帧移 (Hop) | "步幅" | 连续帧之间的采样点数；10ms 是 ASR 默认 |
| 窗函数 | "那个 Hann/Hamming 乘的东西" | 逐点乘以帧——将帧两端平滑衰减到零 |
| STFT | "频谱图的生成器" | 分帧 + 加窗 + FFT；产出时间 × 频率矩阵 |
| 梅尔 | "扭曲的频率" | 对数感知尺度；m = 2595·log10(1 + f/700) |
| 滤波器组 | "那个矩阵" | 三角滤波器将 STFT 投影到梅尔 bin |
| log-mel | "Whisper 的输入" | log(mel_spec + eps)；2026 标准特征 |
| MFCC | "老派特征" | 对数梅尔的 DCT；13 个系数，解相关 |

---

## 📚 小结

频谱图将原始波形转换为时频表示——消除了时间抖动，保留了哪些频率有能量。梅尔频谱图进一步对齐了人类听觉——低频密集、高频稀疏的梅尔滤波器组模拟了人耳的对数分辨率。log-mel 是 2026 语音 ML 的标准前端——Whisper、Parakeet、SeamlessM4T 全部使用 80 个 log-mel 特征、10ms 帧移、25ms 帧长。如果你不在做音乐——**80 log-mel 是你的起点。**

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——合成 chirp 信号（200→4000Hz），打印每帧 argmax 梅尔 bin。绘制（可选）并确认它匹配 chirp 的频率扫频。

2. 【实验】改变 `n_mels` = {40, 80, 128} 和 `frame_len` = {200, 400, 800}，在 chirp 上运行。哪个组合的时间-频率分辨率最好？

3. 【实现】仅用 `math` 实现 `power_to_db`。在 AudioMNIST 上比较 (a) 原始 log-mel、(b) 参考归一化 dB-mel、(c) MFCC-13 + delta + delta-delta 的小型 CNN 分类器准确率。

4. 【思考】为什么 Whisper 选择 80 log-mel 而不是 128？从 128 到 80 丢失了哪些信息？在什么场景下 128 比 80 更好？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 特征提取校验提示词 | `outputs/skill-feature-extractor.md` | 为给定模型选择特征类型、梅尔数、帧长帧移的系统化方案 |

---

## 📖 参考资料

1. [论文] Davis, Mermelstein. "Comparison of parametric representations for monosyllabic word recognition". IEEE, 1980. https://ieeexplore.ieee.org/document/1163420 — MFCC 论文
2. [论文] Stevens, Volkmann, Newman. "A Scale for the Measurement of the Psychological Magnitude Pitch". 1937. — 原始梅尔尺度论文
3. [代码] OpenAI Whisper `log_mel_spectrogram`. https://github.com/openai/whisper/blob/main/whisper/audio.py — 参考实现
4. [文档] librosa — feature extraction. https://librosa.org/doc/main/feature.html — `mfcc`、`melspectrogram`、hop/window 的参考
5. [文档] NVIDIA NeMo — audio preprocessing. https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html — Parakeet + Canary 的生产流水线

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
