# 文本到语音

> 文本到语音（TTS）是 2010 年代 NLP 的"杀手级应用"。从 Siri 到 Claude，从有声书到客服机器人，TTS 是 AI 与人类之间最直接的界面。本课构建从波形到语义的完整理解。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 06 · 01（音频基础） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 TTS 流水线的四个核心阶段——文本预处理、声学模型、声码器、后处理
- [ ] 区分非自回归（FastSpeech2, VITS）与自回归（Tacotron2, VALL-E 2）TTS 的权衡
- [ ] 计算梅尔帧预算——给定文本长度和声码器规格，估算延迟和内存占用

---

## 1. 问题

TTS 的目标：将文本转换为自然的人声。但"自然"是一个高维概念——韵律、重音、停顿、情感、说话人特征都需要正确建模。

2026 年 TTS 的核心挑战：

1. **实时性。** 语音助手需要 <500ms 的首字节延迟（TTFT）。自回归模型天然慢
2. **多样性。** 同一句话用不同语气说。"太好了"是真诚的赞美还是讽刺？
3. **说话人一致性。** 从几秒参考音频中克隆音色，且在多句话中保持一致
4. **中文处理。** 声调、儿化音、"了/着/过"语气词——对英文 TTS 模型是盲区

---

## 2. 概念

### 2.1 TTS 流水线四个阶段

```
文本 → [文本分析] → 音素序列 → [声学模型] → 梅尔频谱图 → [声码器] → 波形
         ↑
   规范化 + G2P
```

| 阶段 | 输入 | 输出 | 代表模型 |
|---|---|---|---|
| 文本分析 | 字符串 | 音素序列 + 韵律标记 | g2p_en / espeak-ng / 文字分析器 |
| 声学模型 | 音素序列 | 梅尔频谱图 | FastSpeech2, VITS, F5-TTS |
| 声码器 | 梅尔频谱图 | 波形 | HiFi-GAN, Vocos, BigVGAN |
| 后处理 | 波形 | 清洁波形 | 降噪、去混响、响度归一化 |

### 2.2 非自回归 vs 自回归

| | 非自回归（FastSpeech2, VITS） | 自回归（Tacotron2, VALL-E 2） |
|---|---|---|
| 生成方式 | 并行生成所有梅尔帧 | 逐帧生成，帧间有条件依赖 |
| 延迟 | 低（并行） | 高（序列化） |
| 流畅度 | 中等（可能有模糊） | 高（上下文连贯） |
| 韵律控制 | 显式（时长/能量/基频预测器） | 隐式（注意力权重） |
| 代表 | FastSpeech2, VITS, F5-TTS | Tacotron2, VALL-E 2, Voicebox |

### 2.3 声码器——从梅尔到波形

- **HiFi-GAN (2020):** 基于 GAN 的声码器。2026 年最常用。3.2M 参数，可在 CPU 上实时生成
- **Vocos (2023):** 基于 iSTFT 的声码器，用 2-3 倍更少的参数达到与 HiFi-GAN 相同的质量
- **BigVGAN (2023):** 大规模声码器，质量最高，但参数量更大
- **F5-TTS 内联声码器：** 端到端架构，不单独需要声码器

### 2.4 2026 TTS 技术栈

| 场景 | 选择 |
|---|---|
| 零样本/少样本 TTS | F5-TTS（无需音素对齐，Flow Matching） |
| 最高音质 + 延迟预算允许 | VALL-E 2 / NaturalSpeech 3 |
| 低延迟生产 | VITS + HiFi-GAN |
| 最小模型 + 离线 | Kokoro v0.19（82M，笔记本可运行） |
| 中文 TTS | CosyVoice / CPT / Fish Speech |
| 声音克隆（少样本） | GPT-SoVITS / RVC |

### 2.5 中文 TTS 特殊挑战

- **声调建模。** 中文有 4 个声调（+轻声），声调变化直接影响语义。FastSpeech2 的声调预测器在中文中比英文更关键
- **语气词。** "啊/呢/吧/吗"是语气/情感的核心信号，不能在音素层面忽略
- **儿化音。** 北方方言的儿化音在字面上是两个字（"花儿"），但发音是一个音节
- **数字/单位读法。** "2024年"→"二零二四年"（逐字读）vs "两千零二十四年"（语义读）——取决于上下文

---

## 3. 从零实现

### Step 1: 文本到音素

```python
# 玩具英语 G2P 映射（真实系统用 espeak-ng 或 g2p_en）
G2P = {"the": ["DH", "AH"], "ing": ["IH", "NG"], ...}

def phonemize(text):
    """最长匹配优先——3字符→2字符→1字符。"""
    text = text.lower()
    phones, i = [], 0
    while i < len(text):
        for length in (3, 2, 1):
            if i + length <= len(text):
                chunk = text[i:i+length]
                if chunk in G2P:
                    phones.extend(G2P[chunk])
                    i += length
                    break
        else:
            i += 1
    return phones
```

### Step 2: 音素时长估计

```python
DURATION_FRAMES = {"AA": 9, "AE": 7, "AH": 6, "AO": 8, ...}  # 每个音素的典型帧数

def estimate_durations(phones, jitter=0.1):
    return [max(1, DURATION_FRAMES.get(p, 5) + int(round(
        DURATION_FRAMES.get(p, 5) * random.uniform(-jitter, jitter)))) for p in phones]
```

### Step 3: 梅尔帧调度

```python
def mel_schedule(phones, durs, hop_ms=12.5):
    """将音素+时长转换为梅尔频谱图的时间调度。"""
    sched, t = [], 0.0
    for p, d in zip(phones, durs):
        sched.append((p, t, t + d * hop_ms))
        t += d * hop_ms
    return sched, t
```

### Step 4: 帧预算计算

```python
total_frames = sum(durs)
audio_samples = total_frames * 300  # 12.5ms hop @ 24kHz = 300 samples
memory_kb = total_frames * 80 * 4 / 1024  # 80 mel bins, float32
```

完整代码见 `code/main.py`——纯标准库，可立即运行。

---

## 4. 工业工具

### 4.1 主流 TTS 框架

```python
# CosyVoice（阿里，中文最佳）
from cosyvoice import CosyVoice
model = CosyVoice("pretrained_models/CosyVoice-300M")
output = model.inference_sft("你好，世界！", speaker="中文女声")

# F5-TTS（零样本 TTS）
from f5_tts import F5TTS
model = F5TTS.from_pretrained("F5-TTS")

# Kokoro（82M，离线可用）
import kokoro
voices = kokoro.load_voices("kokoro-v0_19")
audio = kokoro.generate("Hello, world!", voice=voices["default"])

# Fish Speech（开源声音克隆）
from fish_speech import FishSpeech
model = FishSpeech.from_pretrained("fishaudio/fish-speech-1.5")
```

### 4.2 中文 TTS 特别建议

- **CosyVoice 是 2026 年中文 TTS 的默认起点。** 开箱即用的声音克隆（3-10 秒参考音频），支持情感控制
- **中文需要 G2P（文字到音素）。** 中文不像英文有天然空格——字面转换为拼音是必要步骤。`pypinyin` 或 `g2p` 库提供这个功能
- **数字读法需要上下文。** "2024年"在新闻中读"二零二四年"，在故事中读"两千零二十四年"。简单规则是逐字读（年份/代码），完整读（数量/日期）

---

## 5. 知识连线

- **阶段 06 · 02（频谱图）→** TTS 声学模型的输出正是第 02 课的梅尔频谱图——80 个梅尔 bin
- **阶段 06 · 01（音频基础）→** 声码器输出的波形必须符合第 01 课的采样率规范（TTS 通常 24kHz）
- **阶段 06 · 04（ASR）→** TTS 是 ASR 的逆问题——ASR 从音频到文本，TTS 从文本到音频。两者的 encoder-decoder 架构可以互相借鉴

---

## 6. 常见错误

### 错误 1：未处理中文声调

**现象：** "妈"（妈、麻、马、骂）四个声调读成同一个音——听起来是错误的中文。

**原因：** 直接用英文 TTS 的 G2P 处理中文——英文 G2P 不包含声调信息。"ma"在英文 G2P 中只有一个发音，但中文需要 4 个不同的基频轮廓。

**修复：** 用中文 G2P（`pypinyin`）提取声调，并在声学模型中添加声调预测器。CosyVoice 等中文专用模型已内置处理。

### 错误 2：忽略了韵律断句

**现象：** 生成的语音没有停顿、节奏平坦——像机器人在读流水账。

**原因：** 原始文本没有标点符号或语义分段信息。"猫吃了晚饭在花园里散步"需要在"晚饭"和"散步"之间有停顿——但模型不知道。

**修复：** 在文本分析阶段加入规则或轻量模型检测句子边界和子句边界。确保标点符号被正确转换为停顿时长（句号=12帧，逗号=6帧）。

---

## 7. 面试考点

### Q1：为什么 F5-TTS 不需要音素对齐？（难度：⭐⭐）

**参考答案：**
传统 TTS（如 Tacotron2）需要文本和梅尔频谱图之间的强制对齐（monotonic alignment search）。这在训练时需要额外的计算，并且对齐错误会导致发音失真。F5-TTS 使用 Flow Matching——直接从文本嵌入到梅尔频谱图的连续映射——不需要显式对齐。这个优势使得 F5-TTS 特别适合少样本 TTS（3-10 秒参考音频即可克隆音色），因为对齐是传统 TTS 数据准备中最难的部分。

### Q2：TTS 中声调对中文为什么比英文更重要？（难度：⭐⭐）

**参考答案：**
英文中语调（intonation）主要影响语气和情感——"really?"的升调表示疑问，但不影响词义。中文的声调（tone）直接决定词义——"mā（妈）"、"má（麻）"、"mǎ（马）"、"mà（骂）"是完全不同的四个词。如果 TTS 模型不显式建模声调（通过声调预测器或声调嵌入），生成的语音可能包含无法听懂的歧义——这是中文 TTS 的核心难点，也是英文 TTS 不需要担心的。

### Q3：为什么 Kokoro 的 82M 参数能达到 UTMOS 3.87？（难度：⭐⭐⭐）

**参考答案：**
三个因素：(1) **架构优化**——Kokoro 使用更高效的注意力机制和层归一化策略，减少了冗余参数。(2) **训练数据质量**——高质量的多说话人数据集使得小模型也能学到丰富的语音模式。(3) **评估指标的局限性**——UTMOS 是一个预测指标，不是人类感知的完美代理。Kokoro 在短句和标准普通话上表现很好，但在长句、方言或情感变化上可能不如参数更多的模型。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| G2P | "文字转音素" | Grapheme-to-Phoneme：将字符序列转换为音素序列 |
| 声码器 | "波形生成器" | 从梅尔频谱图恢复波形的神经网络（HiFi-GAN, Vocos） |
| 韵律 | "说话的节奏" | 停顿、重音、语调、语速的变化模式 |
| Flow Matching | "生成模型的新范式" | F5-TTS 的核心：直接从文本嵌入到梅尔频谱图的连续映射，无需对齐 |
| 零样本 TTS | "没有录音也能克隆" | 只需 3-10 秒参考音频，无需在目标说话人数据上微调 |
| UTMOS | "自动音质评分" | 预测主观音质评分（MOS）的模型，0-5 分，越高越好 |

---

## 📚 小结

TTS 是从文本到自然语音的流水线：文本分析（G2P+韵律）→ 声学模型（音素→梅尔）→ 声码器（梅尔→波形）。2026 年非自回归架构（F5-TTS, VITS）在延迟和质量之间取得最佳平衡，而 F5-TTS 的 Flow Matching 去除了音素对齐需求。Kokoro 82M 参数达到 UTMOS 3.87——是当前最小的高质量开源模型。中文 TTS 的核心难点是声调建模和语气词处理。

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——计算一句话的梅尔帧预算。给定 24kHz 声码器，估算生成这段语音需要多少时间。

2. 【实验】用 `torchaudio` 加载一段真实语音，提取梅尔频谱图。反向运行声码器（使用 HiFi-GAN），对比原始和重建波形。

3. 【对比】分别用 CosyVoice 和 Kokoro 生成同一段中文文本。对比：首字节延迟、总时长、模型大小、发音自然度。

4. 【思考】如果你要在嵌入式设备上部署 TTS（内存 < 256MB，无 GPU），你会选择什么架构？为什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| TTS 选择提示词 | `outputs/skill-tts-picker.md` | 按场景选择架构、声码器和质量/延迟权衡 |

---

## 📖 参考资料

1. [论文] Kim et al. "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)". ICML, 2021.
2. [论文] Chen et al. "VALL-E: Neural Codec Language Models are Neural TTS". 2023. https://arxiv.org/abs/2301.02111
3. [论文] Shi et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching". 2024. https://arxiv.org/abs/2410.18723
4. [代码] CosyVoice. https://github.com/FunAudioLLM/CosyVoice — 阿里中文 TTS
5. [代码] Kokoro. https://github.com/hexgrad/kokoro — 82M 开源 TTS

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文 TTS 挑战分析、工程最佳实践、常见错误、面试考点等均为原创内容。
