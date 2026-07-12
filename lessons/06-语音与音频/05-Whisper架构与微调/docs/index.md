# Whisper——架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上训练。一个架构、多个任务、跨越 99 种语言的鲁棒性。2026 年的参考 ASR。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 04（ASR）、阶段 05 · 10（注意力）、阶段 07 · 05（Transformer） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 Whisper 的内部结构——编码器（stride-2 卷积 + Transformer）和解码器（因果自注意力 + 交叉注意力）的完整前向传播
- [ ] 为长音频设计分块策略——理解 30s 窗口 + 5s 步进的重叠设计
- [ ] 实现 LoRA 微调方案——q_proj/v_proj r=16，可训练参数降至原模型的 1/100+

---

## 1. 问题

Whisper 在 2022 年 9 月发布，是第一个商品化的 ASR 模型——粘贴音频、得到文本、99 种语言、鲁棒噪声、笔记本运行。到 2024 年 OpenAI 发布了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 是从播客转录到语音助手到 YouTube 字幕的默认基准。

但 Whisper 不能永远当黑盒用。**领域偏移会杀死它**——技术术语、说话人口音、专有名词、短片段、静默。你需要知道：

1. 它内部实际是什么
2. 如何正确地分块、流式或长音频
3. 什么时候微调以及如何做

---

## 2. 概念

### 2.1 架构——标准 Transformer 编码器-解码器

| 组件 | 说明 |
|---|---|
| **输入** | 30s log-mel 频谱图，80 梅尔，10ms 帧移 → 3000 帧。短于 30s 的零填充，长于 30s 的分块 |
| **编码器** | stride-2 卷积降采样 + N 层 Transformer block。Large-v3: 32 层，1280 维，20 头 |
| **解码器** | N 层 Transformer block，因果自注意力 + 对编码器输出的交叉注意力。与编码器同大小 |
| **输出** | 51,865 token BPE 词表上的分布 |

### 2.2 任务 token——一个模型做所有事

Whisper 通过特殊 token 在同一模型中支持多个任务：

```python
SPECIAL = {
    "SOT":           "<|startoftranscript|>",
    "TRANSCRIBE":    "<|transcribe|>",
    "TRANSLATE":     "<|translate|>",
    "NO_TIMESTAMPS": "<|notimestamps|>",
    "NO_SPEECH":     "<|nospeech|>",
}
LANG = {"en": "<|en|>", "fr": "<|fr|>", "ja": "<|ja|>", "zh": "<|zh|>"}
```

- 转录：`<|startoftranscript|><|en|><|transcribe|><|notimestamps|`
- 翻译到英语：`<|startoftranscript|><|fr|><|translate|><|notimestamps|>`
- 带时间戳转录：`<|startoftranscript|><|ja|><|transcribe|>`（不加 `<|notimestamps|>`）

### 2.3 2026 年的 Whisper 家族

| 版本 | 层 | d_model | 参数量 | 特点 |
|---|---|---|---|---|
| Tiny | 4 enc + 4 dec | 384 | ~39M | 最快，最低质量 |
| Base | 6 enc + 6 dec | 512 | ~74M | |
| Small | 12 enc + 12 dec | 768 | ~244M | |
| Medium | 24 enc + 24 dec | 1024 | ~769M | |
| Large-v3 | 32 enc + 32 dec | 1280 | ~1.5B | 最高质量，最慢 |
| **Turbo** | **32 enc + 4 dec** | 1280 | ~809M | **解码器只有 4 层**，速度 6-8×，2026 默认 |

**Turbo 是关键。** 32 层编码器（保持完整特征提取能力），解码器从 32 层减到 4 层（减少生成延迟）。用大批量蒸馏训练数据补偿解码器简化带来的质量损失。**2026 年生产部署的默认选择。**

### 2.4 长音频处理——分块策略

Whisper 编码器固定处理 30 秒。超过 30s 的音频必须分块：

```
原始音频: [================================]  (10分钟)
分块:
  块1: [0-30s]
  块2: [25-55s]  (5s 重叠——确保边界词不被切断)
  块3: [50-80s]
  ...
  块 N: [末尾30s]
```

重叠（stride 5s）是关键——没有重叠，边界处的词可能被一分为二，两端各自识别一半，合并后产生错误。WhisperX 工具实现了高效分块 + wav2vec 2.0 强制对齐。

### 2.5 微调策略

**LoRA（Low-Rank Adaptation）** 在 q_proj 和 v_proj 上添加低秩分解（r=16）。可训练参数 = 2×d_model×rank 每层。

对于 Medium 级别（24层，1024维）：LoRA 只需训练约 7.8M 参数——原模型有 1.5B 参数。**编码器完全冻结**，只更新 LoRA 适配器和解码器。

何时微调：领域专有名词（医疗/法律）、特定说话人口音、低资源语言、需要统一格式输出时。

---

## 3. 从零实现

### Step 1：解码器提示构建

```python
def build_prompt(language, task="transcribe", timestamps=False):
    toks = [SPECIAL["SOT"], LANG[language]]
    toks.append(SPECIAL["TRANSCRIBE"] if task == "transcribe" else SPECIAL["TRANSLATE"])
    if not timestamps:
        toks.append(SPECIAL["NO_TIMESTAMPS"])
    return toks
```

### Step 2：分块策略

```python
def chunk_schedule(total_seconds, chunk_s=30.0, stride_s=5.0):
    if total_seconds <= chunk_s:
        return [(0.0, total_seconds)]
    out, start, step = [], 0.0, chunk_s - stride_s
    while start < total_seconds:
        end = min(total_seconds, start + chunk_s)
        out.append((round(start, 2), round(end, 2)))
        if end == total_seconds:
            break
        start += step
    return out
```

### Step 3：Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav", language="zh")
print(result["text"])
```

### Step 4：LoRA 微调

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(whisper_model, config)
# 只训练 LoRA 适配器，编码器完全冻结
```

完整代码见 `code/main.py`——参数量计算、分块策略、提示构建。

---

## 4. 工业工具

| 场景 | 选择 |
|---|---|
| 离线英文最高质量 | Whisper-Large-v3-turbo |
| 多语言鲁棒 | SeamlessM4T v2 |
| 流式低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备 | Whisper-Tiny int8 量化版 |
| 领域微调 | LoRA r=16 在 q_proj/v_proj，冻结编码器 |
| 长音频 | WhisperX（分块 + wav2vec 2.0 强制对齐） |
| 低资源语言 | 2-20 小时领域音频 LoRA 微调 |

---

## 5. 知识连线

- **阶段 06 · 02（频谱图）→** Whisper 的输入正是第 02 课的 log-mel 频谱图——80 梅尔，10ms 帧移
- **阶段 06 · 04（ASR）→** Whisper 使用 Attention encoder-decoder 公式——本课的解码器提示和分块策略是第 04 课理论的具体实现
- **阶段 05 · 10（注意力）→** Whisper 的交叉注意力就是阶段 10 的 Q/K/V——解码器的 Q 查编码器的 K 和 V

---

## 6. 常见错误

### 错误 1：未加 VAD 就跑 Whisper

**现象：** 静默片段产生幻觉文本（"Thanks for watching!"）。

**原因：** Whisper 的训练分布几乎全是语音——对静默"困惑"，生成看似合理但虚构的文本。

**修复：** 用 `silero-vad` 或 `webrtcvad` 门控——只有 VAD 检测到语音的片段才送入 Whisper。

### 错误 2：Turbo vs Large-v3 混淆

**现象：** 以为使用了 Large-v3 的完整质量，实际用的是 Turbo——某些边界情况下 Turbo 质量明显下降。

**原因：** Turbo 解码器只有 4 层——它无法进行与 32 层相同复杂度的推理。特别是在低信噪比或强口音场景。

**修复：** 检查实际加载的模型文件名。用 `whisper.load_model("large-v3")` 而非 `"large-v3-turbo"` 时确保你真的需要完整质量。

---

## 7. 面试考点

### Q1：Whisper 的 30 秒窗口限制是如何通过分块策略解决的？（难度：⭐⭐）

**参考答案：**
Whisper 编码器固定处理 30 秒。对于更长音频，使用滑动窗口分块：典型配置是 30s 窗口 + 5s 步进（stride）。每个块独立识别后在重叠区域合并——5s 重叠确保边界词不会被截断。WhisperX 工具在分块基础上加了 wav2vec 2.0 强制对齐，进一步提升合并后的连贯性。不重叠的分块会产生边界断裂——同一个词被分成两半，两边各自识别一半。

### Q2：为什么 LoRA 微调 Whisper 时编码器要完全冻结？（难度：⭐⭐）

**参考答案：**
三个原因：**（1）** 编码器已经在 68 万小时弱监督数据上学到了强大的语音特征——领域适配主要影响的是解码器的语言模型部分（术语、格式），而非语音特征提取。**（2）** 冻结编码器大幅降低显存需求——只需要保存 LoRA 适配器的梯度（约 10M 参数 vs 1.5B 参数）。**（3）** LoRA 的低秩分解本身就是对参数更新的约束——同时更新编码器会破坏这个约束的数学性质。

### Q3：Whisper 的时间戳功能在生产中如何使用？（难度：⭐⭐⭐）

**参考答案：**
Whisper 的时间戳功能有两种精度：**词级时间戳**（通过 WhisperX 工具中的 wav2vec 2.0 强制对齐实现——更准确）和**块级时间戳**（原生 Whisper 只输出块起始时间）。生产中：（1）先用 VAD 分割静默 → 送入 Whisper 识别并获取块级时间戳 → 用强制对齐细化到词级。这个流水线是字幕生成、视频索引、对话系统对齐的标准配置。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 编码器 | "Whisper 的耳朵" | stride-2 卷积 + N 层 Transformer，将 log-mel 帧压缩为隐藏状态 |
| 解码器 | "Whisper 的嘴巴" | 因果自注意力 + 交叉注意力，逐 token 生成文本 |
| LoRA | "高效微调" | 低秩分解——在 q_proj/v_proj 上添加可训练适配器，原模型冻结 |
| 强制对齐 | "时间戳对齐" | 用 wav2vec 2.0 将 token 精确映射到音频位置 |
| 词级时间戳 | "每个词有时间" | 比块级更精确的对齐，用于字幕和视频索引 |
| LoRA r=16 | "秩为 16" | 每层可训练参数 = 2×d_model×rank，约原模型的 1/100 |

---

## 📚 小结

Whisper 是标准 Transformer 编码器-解码器，30 秒窗口，80 log-mel，BPE 词表。2026 年生产默认是 **Turbo 版本**——32 层编码器 + 4 层解码器，质量接近 Large-v3 但速度 6-8 倍。长音频通过滑动窗口分块（30s/5s stride）处理。微调用 **LoRA r=16** 在 q_proj/v_proj 上——可训练参数仅为原模型的 1/100+。始终在 Whisper 前加 VAD 门控——没有它，静默片段会产生幻觉文本。

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——计算 Tiny 到 Turbo 各规格的参数量。对比 LoRA 可训练参数占比。

2. 【实验】用 WhisperLarge-v3-turbo 在 10 条中文新闻音频上测试。记录 WER、是否产生幻觉文本、分块边界是否断裂。

3. 【实现】设计一个 VAD + Whisper 的简化流水线——检测静默、分割语音段、逐段识别、合并结果。

4. 【思考】你的 Whisper 模型在中文医疗术语上频繁出错——"高血压"被转录为"高血牙"。设计一个最简微调方案（用多少数据？多少 epochs？LoRA 配置？）。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| Whisper 微调提示词 | `outputs/skill-whisper-finetune.md` | 按场景选择 Whisper 版本、分块和微调方案 |

---

## 📖 参考资料

1. [论文] Radford et al. / OpenAI. "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision". 2022. https://arxiv.org/abs/2212.04356 — Whisper 论文
2. [代码] whisper 库. https://github.com/openai/whisper — 官方实现
3. [工具] whisperx. https://github.com/m-bain/whisperX — 分块 + 强制对齐
4. [工具] LoRA (PEFT). https://github.com/huggingface/peft — Hugging Face LoRA 实现

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
