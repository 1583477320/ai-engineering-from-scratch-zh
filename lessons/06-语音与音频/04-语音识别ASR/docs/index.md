# 语音识别——CTC、RNN-T、注意力

> 语音识别是每个时间步上的音频分类，被一个懂语言和静音的序列模型粘合在一起。CTC、RNN-T 和注意力是三种方式。选一个，理解为什么。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 05 · 08（CNN/RNN 文本建模）、阶段 05 · 10（注意力机制） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 CTC 贪心解码和束搜索解码——理解 blank 合并规则以及为什么束搜索在噪声 logits 上优于贪心
- [ ] 从零实现 WER（词错误率）——编辑距离在词级上的应用，理解 S/D/I/N 四个分量
- [ ] 区分 CTC、RNN-T、Attention encoder-decoder 三种 ASR 公式——各自的流式能力、内部语言模型、计算开销

---

## 1. 问题

你有一段 10 秒 16kHz 的音频。你想要一个字符串："打开厨房的灯"。结构性困难是：**音频帧不与字符一一对应。** "好的"可能只占 200ms 也可能占 1200ms。静音穿插在语音之间。有些音素比其他更长。**输出 token 的数量在推理时未知。**

三种公式解决这个问题：

1. **CTC（连接时序分类）。** 每帧输出 token 概率（包括一个特殊的 *blank*）。解码时合并重复、去除 blank。非自回归、流式、零前瞻。wav2vec 2.0、MMS 使用。
2. **RNN-T（循环神经网络转录器）。** 联合网络在编码器帧和前序 token 的条件下预测下一个 token。流式。Google 设备端 ASR、NVIDIA Parakeet 使用。
3. **注意力编码器-解码器。** 编码器压缩音频为隐藏状态，解码器交叉注意力生成 token。自回归。Whisper、SeamlessM4T 使用。

2026 年 LibriSpeech test-clean 上的 SOTA WER：1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差距微小；**部署差异巨大。**

---

## 2. 概念

### 2.1 CTC 直觉

编码器输出 T 个帧级分布——每个分布在 V+1 个 token 上（V 个字符 + blank）。对于长度为 U < T 的目标字符串 y，任何折叠后得到 y 的对齐都算。CTC 损失对所有这些对齐求和。推理：逐帧 argmax → 合并重复 → 去除 blank。

**优点：** 非自回归、流式、零前瞻。**缺点：** 条件独立假设——每帧预测独立于其他帧，没有内部语言模型。用外部语言模型 + 束搜索修复。

### 2.2 RNN-T 直觉

加了一个**预测器**网络嵌入 token 历史，和一个**连接器**将预测器状态与编码器帧合并为联合分布。显式建模了 CTC 忽略的条件依赖。流式，因为每一步只依赖过去帧和过去 token。

**优点：** 流式 + 内部语言模型。**缺点：** 训练更复杂、内存更耗（3D 损失格）。

### 2.3 注意力编码器-解码器

6-32 层 Transformer 编码器处理 log-mel 帧。6-32 层 Transformer 解码器交叉注意力生成 token。没有对齐约束——注意力可以看音频的任何位置。非流式（除非限制注意力，如分块 Whisper-Streaming）。

**优点：** 离线 ASR 最高质量。**缺点：** 自回归延迟与输出长度成正比；不工程化就无法流式。

### 2.4 WER——那个你要报告的数字

**WER = (替换 + 删除 + 插入) / 参考词数**。匹配词级 Levenshtein 编辑距离。越低越好。WER > 20% 通常不可用；< 5% 是朗读语音的人类水平。

| 模型 | LibriSpeech test-clean | test-other | 参数量 |
|---|---|---|---|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| SeamlessM4T v2 | 1.7% | 3.5% | 2.3B |

---

## 3. 从零实现

### Step 1：贪心 CTC 解码

```python
def ctc_greedy(frame_probs, blank=0):
    """两步：合并连续重复，去除 blank。"""
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_probs]
    out, prev = [], -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out
```

### Step 2：束搜索 CTC 解码

```python
def ctc_beam(frame_probs, beam_width=8):
    """保持 beam_width 个最优部分序列。"""
    beams = [((), 0.0)]
    for p in frame_probs:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        new_beams = {}
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                if t == BLANK: new_seq = seq
                elif seq and seq[-1] == t: new_seq = seq  # 合并重复
                else: new_seq = seq + (t,)
                # 合并相同序列的概率
                if new_seq in new_beams:
                    new_beams[new_seq] = math.log(math.exp(new_beams[new_seq]) + math.exp(lp + lpt))
                else: new_beams[new_seq] = lp + lpt
        beams = sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width]
    return "".join(VOCAB[i] for i in beams[0][0])
```

### Step 3：WER

```python
def wer(ref, hyp):
    """词错误率 = 编辑距离 / 参考词数。"""
    r, h = ref.split(), hyp.split()
    nr = len(r)
    if nr == 0: return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(nr + 1)]
    for i in range(nr + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[nr][len(h)] / nr
```

### Step 4：Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

### Step 5：流式 ASR

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

| 场景 | 选择 |
|---|---|
| 英文、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘/移动、<500ms 延迟 | Whisper-Tiny 量化版或 Moonshine (2024) |
| 长音频 | Whisper + VAD 分块（WhisperX） |
| 领域特定（医疗/法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

---

## 5. 常见错误

### 错误 1：未加 VAD 就跑 Whisper

**现象：** 静默片段产生幻觉文本（"Thanks for watching!"）。

**原因：** Whisper 的训练分布几乎全是语音——对静默输入"困惑"，产生看起来合理但完全虚构的文本。

**修复：** 在 Whisper 之前用 VAD 过滤静默。`silero-vad` 或 `webrtcvad` 门控。只对 VAD 检测到的语音段调用 Whisper。

### 错误 2：字符级 vs 词级 WER 混用

**现象：** 一个系统报告 5% WER，另一个报告 15%——看起来后者差很多，但实际是不同指标。

**原因：** 字符级 WER 天然低于词级 WER——每个字符错误只占 1/5 到 1/10 个词级错误。**永远报告词级 WER，且必须先归一化**（小写、去标点）。

**修复：** 用 `jiwer` 库统一评估——它内置了归一化和词级 WER 计算。

### 错误 3：Whisper 语言识别漂移

**现象：** 噪声片段被误判为威尔士语或日语——输出完全错误的转录。

**原因：** Whisper 的自动语言识别（auto LID）在噪声输入上不稳定——特别是低信噪比的短片段。

**修复：** 当你知道语言时，强制 `language="en"`（或对应语言代码）。只在真正需要自动检测时才用 auto LID。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| CTC | "blank token 损失" | 对所有帧-字符对齐的边缘化；非自回归 |
| RNN-T | "流式损失" | CTC + 下一个 token 预测器；处理词序依赖 |
| 注意力编码器-解码器 | "Whisper 风格" | 编码器 + 交叉注意力解码器；离线最高质量 |
| WER | "你要报告的数字" | (替换+删除+插入)/参考词数，词级 |
| Blank | "空白帧" | CTC 中的特殊 token，表示"这一帧不输出" |
| LM 融合 | "外部语言模型" | 束搜索中加权 LM 的 log 概率 |
| VAD | "静音门" | 语音活动检测器；过滤非语音段 |

---

## 📚 小结

三种 ASR 公式都在解决同一个问题——音频帧不与字符对齐。CTC 简单快速但没有内部语言模型；RNN-T 流式且有语言模型但训练复杂；注意力编码器-解码器质量最高但自回归延迟高。WER = (S+D+I)/N，是你要报告的唯一数字。2026 年 LibriSpeech test-clean：Parakeet-TDT 1.40%，Whisper-Large-v3-turbo 1.58%。差距微小——**部署差异（流式/离线/延迟/多语言）才是选择的关键。**

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——贪心解码一个手工构造的 CTC 输出，计算 WER。

2. 【实现】正确实现前缀树束搜索（处理 blank 合并规则）。在 10 条合成样本上与贪心对比。

3. 【实验】用 `whisper-large-v3-turbo` 在 LibriSpeech test-clean 上计算前 100 句的 WER。与发表数字对比。

4. 【思考】你的 ASR 系统在安静的录音室环境下 WER=2%，但在嘈杂的餐厅环境中 WER=15%。分析最可能的原因——是模型问题还是信号处理问题？给出具体的排查方案。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| ASR 选择提示词 | `outputs/skill-asr-picker.md` | 按部署目标选择模型、解码、分块和 LM 融合 |

---

## 📖 参考资料

1. [论文] Graves. "Connectionist Temporal Classification". ICML, 2006. https://www.cs.toronto.edu/~graves/icml_2006.pdf — CTC 论文
2. [论文] Graves. "Sequence Transduction with Recurrent Neural Networks". ICML, 2012. https://arxiv.org/abs/1211.3711 — RNN-T 论文
3. [论文] Radford et al. / OpenAI. "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision". 2022. https://arxiv.org/abs/2212.04356 — Whisper
4. [模型] NVIDIA Parakeet-TDT-1.1B. https://huggingface.co/nvidia/parakeet-tdt-1.1b — 2026 Open ASR 排行榜榜首
5. [基准] Hugging Face Open ASR Leaderboard. https://huggingface.co/spaces/hf-audio/open_asr_leaderboard — 实时基准

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
