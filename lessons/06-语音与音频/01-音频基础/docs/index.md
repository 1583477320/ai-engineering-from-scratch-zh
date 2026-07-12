# 音频基础——波形、采样、傅里叶变换

> 波形是原始信号。频谱图是中间表示。梅尔特征是 ML 友好的形式。每一个现代语音识别和语音合成流水线都走过这道阶梯——第一级台阶是理解采样和傅里叶。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 01 · 06（向量与矩阵）、阶段 01 · 14（概率分布） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释采样率、奈奎斯特频率、混叠三者的关系——16 kHz 采样意味着什么，混叠是怎么发生的
- [ ] 从零实现 DFT——理解 N 个采样点产生 N 个复频率系数（每个 bin 对应频率 k·sr/N Hz）
- [ ] 演示混叠——7 kHz 音调用 10 kHz 采样时会发生什么——理解抗混叠低通滤波器为什么是必须的

---

## 1. 问题

麦克风产生的是气压随时间变化的信号。你的神经网络消费的是张量。两者之间坐着一整套约定——违反任何一个都会产生**静默的 bug**：模型正常训练但词错误率翻倍，或者 TTS 合成出嘶嘶声，或者语音克隆系统记住了麦克风而不是说话者。

语音系统的每一个 bug 都可以追溯到三个问题：

1. **采样率**——数据是以什么采样率录制的？模型期望什么？
2. **混叠**——信号是否发生了混叠？
3. **表示形式**——你在原始采样值上操作还是在频率表示上操作？

前两个问题解决了，阶段 06 的其余部分才可能做下去。做错了——即使是 Whisper-Large-v4 也会输出垃圾。

---

## 2. 概念

### 2.1 波形——一维浮点数组

音频的最简表示：`[-1.0, 1.0]` 范围内的一维浮点数组，按采样点编号索引。转换为秒：`t = n / sr`。10 秒、16 kHz 采样率的片段 = 160,000 个浮点数。

### 2.2 采样率——每秒多少采样点

| 采样率 | 用途 |
|---|---|
| 8 kHz | 传统电话、旧 VoIP。奈奎斯特 4 kHz 会杀死辅音。**ASR 避免使用** |
| 16 kHz | **ASR 标准。** Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2） |
| 44.1 kHz | CD 音频、音乐 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3） |

### 2.3 奈奎斯特-香农采样定理

采样率 `sr` 能无歧义表示的最高频率是 `sr/2`（奈奎斯特频率）。高于 `sr/2` 的能量会**混叠**——折叠回低频区域，污染信号。**每次下采样前必须先低通滤波。**

### 2.4 位深度

16-bit PCM（有符号 int16，范围 ±32,767）是通用交换格式。24-bit 用于音乐，32-bit float 用于内部 DSP。`soundfile` 读入 int16 但暴露为 `[-1, 1]` 的 float32 数组。

### 2.5 傅里叶变换——时域到频域

任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换（DFT）对 N 个采样点计算 N 个复系数——每个频率 bin 一个。**bin k 对应频率 k·sr/N Hz**。幅度是该频率的振幅，相位是时间偏移。

### 2.6 FFT——快速 DFT

FFT：当 N 是 2 的幂时，O(N log N) 的 DFT 算法。所有音频库底层都用 FFT。1024 采样点的 FFT @16kHz → 512 个可用频率 bin，覆盖 0-8 kHz，分辨率 15.6 Hz。

### 2.7 分帧 + 加窗——短时傅里叶变换（STFT）

我们不对整段音频做 FFT——将它切成重叠的**帧**（典型 25ms，帧移 10ms），每帧乘以窗函数（Hann、Hamming）消除边缘不连续性，然后 FFT。这就是**短时傅里叶变换（STFT）**。下一课从这里继续。

---

## 3. 从零实现

### Step 1：合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    """从第一原理合成正弦波。"""
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr) for i in range(n)]

# 440 Hz 正弦波（音乐会 A 音），16 kHz，1 秒 = 16000 个浮点数
```

### Step 2：写入 WAV 文件（16-bit PCM）

```python
import struct, wave

def write_wav(path, samples, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit = 2 bytes
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        )
        w.writeframes(frames)
```

float32 `[-1,1]` → int16 `[-32768,32767]` 的量化误差最大为 1/65536 ≈ 0.0015。对于 16-bit 语音足够。

### Step 3：从零实现 DFT

```python
def dft(x):
    """O(N²)——教学验证用，真实音频用 numpy.fft.rfft。"""
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[j] * math.cos(-2 * math.pi * k * j / N) for j in range(N))
        im = sum(x[j] * math.sin(-2 * math.pi * k * j / N) for j in range(N))
        out.append((re, im))
    return out
```

对 440 Hz 正弦波执行 DFT——峰值 bin = `440 × N / sr`，验证变换的正确性。

### Step 4：检测主导频率

```python
def peak_freq(samples, sr):
    mags = magnitudes(dft(samples))
    half = len(mags) // 2  # 正频率对称
    k = max(range(half), key=lambda i: mags[i])
    return k * sr / len(samples), k
```

### Step 5：演示混叠

```python
# 7 kHz 音调用 10 kHz 采样——7 kHz > 奈奎斯特 5 kHz
tone = sine(7000.0, 10000, 0.0512)
alias_freq, _ = peak_freq(tone, 10000)
# DFT 报告 3000 Hz——实际是 7000 Hz 混叠到 10000-7000=3000 Hz
```

这就是为什么每个 DAC/ADC 都内置了砖墙低通滤波器——不滤掉高于奈奎斯特的频率，混叠会永久污染信号。

### Step 6：朴素下采样 vs 合理下采样

```python
def downsample_naive(samples, factor):
    """直接每第 factor 个采样——不抗混叠，演示混叠现象。"""
    return samples[::factor]
```

24kHz 的 7kHz 音调→朴素下采样到 8kHz→DFT 峰值出现在 1000 Hz（折叠频率）。而抗混叠滤波后的下采样会正确消失 7kHz 信号。

完整代码见 `code/main.py`——仅依赖标准库，可立即运行。

---

## 4. 工业工具——2026 技术栈

| 任务 | 库 | 原因 |
|---|---|---|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile wrapper） | 最快、稳定、返回 float32 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠滤波 |
| STFT / 梅尔特征 | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态 |
| 实时流式 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI，快速，报告 sr/声道/编码 |

**决策规则：采样率匹配优先于其他一切。** Whisper 期望 16kHz 单声道 float32。传入 44.1kHz 立体声会得到看起来像模型 bug 的垃圾输出。

---

## 5. 常见错误

### 错误 1：采样率不匹配静默产生

**现象：** Whisper 输出完全无意义的转录——不是错误，而是胡乱猜测的词。

**原因：** 音频文件是 44.1kHz 立体声。Whisper 期望 16kHz 单声道。模型内部可能静默重采样——但没有正确应用抗混叠滤波——高频成分混叠到低频，ASR 将这些伪影解释为语言内容。

**修复：** 在任何模型推理之前，显式地重采样到目标格式：

```python
import torchaudio
resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
waveform, sr = torchaudio.load("input.wav")
if sr != 16000:
    waveform = resampler(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)  # 立体声→单声道
```

### 错误 2：未加窗的 FFT

**现象：** 频谱图中出现虚假的频谱泄漏——本应是干净的正弦波却出现了多个旁瓣。

**原因：** 直接对 25ms 帧做 FFT——帧的两端有阶跃不连续——这个不连续在频域表现为能量泄漏到所有频率。窗函数（Hann/Hamming）将帧两端平滑衰减到零，消除不连续性。

**修复：** STFT 的每一帧必须乘以窗函数：

```python
import numpy as np
window = np.hanning(frame_len)  # Hann 窗
framed = signal_frames * window  # 逐帧乘窗
```

---

## 6. 面试考点

### Q1：为什么语音 ASR 用 16kHz 而不是 8kHz 或 44.1kHz？（难度：⭐⭐）

**参考答案：**
8kHz 的奈奎斯特频率是 4kHz——人类语音的关键辅音（如 /s/、/f/、/t/）能量在 4-8kHz 范围内，8kHz 会丢失这些辅音，导致 ASR 严重退化。44.1kHz 超出语音实际需要的范围——处理更长的序列、需要更多计算，但不提供额外的可懂度信息。16kHz 的奈奎斯特 8kHz 恰好覆盖了人类语音的所有有意义频率——是 ASR 效率和质量的甜点。Whisper、Parakeet、SeamlessM4T v2 全部使用 16kHz 作为标准输入。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 采样率 | "每秒多少采样点" | ADC 测量信号的频率。16kHz = 每秒 16000 次 |
| 奈奎斯特 | "能表示的最高频率" | sr/2；高于它的频率混叠到低频 |
| 位深度 | "每个采样点的精度" | int16 = 65536 级；float32 = [-1,1] 的 24-bit 精度 |
| DFT | "离散傅里叶变换" | N 个采样点 → N 个复频率系数 |
| FFT | "快速 DFT" | O(N log N) 的 DFT 算法，要求 N = 2 的幂 |
| bin | "频率列" | k · sr / N Hz；分辨率 = sr / N |
| STFT | "频谱图的底层" | 分帧 + 加窗 + FFT |
| 混叠 | "鬼影频率" | 高于奈奎斯特的频率折叠回低频——产生虚假频率成分 |

---

## 📚 小结

音频的最简表示是一维浮点数组（波形）。采样率决定了能捕获的最高频率（奈奎斯特 = sr/2）。DFT 将时域采样点转换为频率系数；FFT 是其 O(N log N) 实现。STFT = 分帧 + 加窗 + FFT → 时频表示——下一课梅尔频谱图从这里开始。

**采样率匹配优先于其他一切。** Whisper 期望 16kHz 单声道 float32——传错格式不报错但输出垃圾。这是阶段 06 所有 bug 的根源。

---

## ✏️ 练习

1. 【实现】合成 1 秒的 220+440+880 Hz 混合正弦波 @16kHz。执行 DFT。确认三个峰值出现在预期的 bin 上。

2. 【实现】录制一段 3 秒的 48kHz 语音。用 `torchaudio.transforms.Resample` 下采样到 16kHz（带抗混叠）。FFT 对比原始和重采样后的信号——混叠发生在什么位置？

3. 【实验】仅用标准库从零实现 STFT——帧长 400、帧移 160、Hann 窗、Step 3 的 DFT。用 `matplotlib` 绘制幅度谱。这就是第 02 课的频谱图。

4. 【思考】为什么 Whisper 在处理一段同时包含语音和背景音乐的音频时，转录质量通常会变差？从采样率和频率分析的角度，音乐的哪个成分最可能干扰语音识别？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 音频加载验证提示词 | `outputs/skill-audio-loader.md` | 根据下游模型需求校验音频文件格式并安全重采样 |

---

## 📖 参考资料

1. [论文] Shannon. "Communication in the Presence of Noise". 1949. https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf — 采样定理的来源
2. [教材] Smith. "The Scientist and Engineer's Guide to Digital Signal Processing". https://www.dspguide.com/ch8.htm — 经典免费 DSP 教材
3. [文档] librosa — audio primer. https://librosa.org/doc/latest/tutorial.html — 带代码的实用入门
4. [博客] Steve Eddins. "FFT Spectrum and Spectral Densities". https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/ — 10 分钟理解频率 bin

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
