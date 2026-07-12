# 音频分类——从 MFCC+k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是哪种语言"，都是音频分类。特征是梅尔。架构每个十年变一次。评估一直是 AUC、F1 和每类召回率。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 03 · 06（CNN）、阶段 05 · 08（CNN/RNN 文本建模） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用 MFCC 均值-方差池化 + k-NN 搭建音频分类基线——在合成 4 类数据集上验证
- [ ] 理解为什么 2026 年的默认选择是冻结骨干微调——BEATs 用 1/4 计算量超越 AST
- [ ] 解释类别不平衡对音频分类的影响——以及 Mixup、SpecAugment、平衡采样如何缓解

---

## 1. 问题

给你一段 10 秒的音频。你想知道："这是什么？"城市噪音（警笛、电钻、狗叫）、语音指令（是/否/停止）、语言识别（英/西/阿）、说话人情绪（愤怒/中性）、环境音（室内/室外、嘈杂）——全部是**音频分类**。2026 年基线架构已经成熟：log-mel → CNN 或 Transformer → softmax。

**核心难点不是网络——是数据。** 音频数据集有严重的类别不平衡、强领域偏移（干净 vs 噪声）、标签噪声（谁决定"城市嘈杂"还是"餐厅噪音"？）。80% 的工作是数据策展、增强和评估——不是把 CNN 换成 Transformer。

---

## 2. 概念

### 2.1 架构演化——四个时代

**k-NN on MFCCs（1990s 基线）。** 将每段 MFCC 在时间维上展平，计算与标注库的余弦相似度，返回 top-K 的多数投票。在干净、小数据集上出人意料地强（Speech Commands、ESC-50）。无需 GPU。

**2D CNN on log-mels（2015-2019）。** 将 `(T, n_mels)` 对数梅尔当作图像。应用 ResNet-18 或 VGG 风格。时间轴全局平均池化 → softmax。2026 年大多数 Kaggle 竞赛仍是基线。

**AST（2021-2024）。** 将 log-mel 切成 patch（如 16×16）→ 加位置嵌入 → 送入 ViT。AudioSet 上监督学习 SOTA（mAP 0.485）。

**BEATs/WavLM-base（2024-2026）。** 在数百万小时上自监督预训练。用 1/10-1/10 的监督数据微调。2026 年非语音音频的默认起点。BEATs-iter3 用 1/4 计算量超越 AST 1-2 mAP。

**Whisper-encoder 冻结骨架（2024）。** 取 Whisper 编码器、丢掉解码器、接一个线性分类器。在语言识别和简单事件分类上接近 SOTA——零音频增强。"免费午餐"基线。

### 2.2 类别不平衡——真正的挑战

| 数据集 | 类别数 | 样本数 | 平衡情况 |
|---|---|---|---|
| ESC-50 | 50 | 2000 | 平衡（40/类） |
| UrbanSound8K | 10 | 8732 | 不平衡 10:1 |
| AudioSet | 632 | 200 万 | 长尾 100,000:1 |

**技术：** 训练时平衡采样（不在评估时）、Mixup（线性插值两个样本及其标签）、SpecAugment（随机掩蔽时间和频率带）。

### 2.3 评估指标

- **多类别独占（Speech Commands）：** top-1 准确率、top-5 准确率
- **多类别多标签（AudioSet）：** mAP（平均精度均值）
- **严重不平衡：** 每类召回率 + 宏 F1

**2026 数字你需要知道：**

| 基准 | 基线 | 2026 SOTA |
|---|---|---|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) |

---

## 3. 从零实现

### Step 1：MFCC 特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=256, hop=128):
    """完整 MFCC 特征提取流水线：波形 → STFT → 梅尔 → log → DCT。"""
    mag = stft_mag(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    lm = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in lm]
```

### Step 2：时间维池化——均值+方差

```python
def summarize(mfcc_frames):
    """13 维 MFCC × 时间 → 26 维固定特征（均值+方差）。"""
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    return mean + var
```

简单但强大——均值+方差池化在 ESC-50 上的基线一直到 2017 年都打败了当时的 NN 方法。

### Step 3：k-NN 分类器

```python
def knn(q, bank, labels, k=3):
    """余弦相似度 + 多数投票。"""
    idx = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    return Counter(labels[i] for i in idx).most_common(1)[0][0]
```

### Step 4：升级到 CNN on log-mels（PyTorch）

```python
class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)
    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数。ESC-50 上 ~10 分钟训练（单张 RTX 4090）。80%+ 准确率。

### Step 5：2026 默认——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification
ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=50, ignore_mismatched_sizes=True)
inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 使用 `microsoft/BEATs-base` 通过 `beats` 库；transformers API 形状相同。

完整代码见 `code/main.py`——从零实现的 k-NN MFCC 基线。

---

## 4. 工业工具

| 场景 | 起点 |
|---|---|
| 极小数据集（<1000 条） | k-NN on MFCC 均值（你的基线）+ 音频增强 |
| 中等数据集（1K-100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从零训练或微调 Whisper-encoder |
| 实时、边缘 | 40-MFCC CNN，int8 量化（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE Loss + Mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

**决策规则：从冻结骨架开始——不从零模型。** 微调 BEATs 的分类头在几小时内就能达到 SOTA 的 95%。

---

## 5. 常见错误

### 错误 1：不平衡数据只报准确率

**现象：** 88% 准确率看起来不错——但负面类（少数类）召回率为 0%。模型把所有东西都预测为多数类。

**原因：** 10:1 的不平衡下，全预测多数类就有 90% 准确率。准确率在这个分布下是一个没有信息量的指标。

**修复：** 报告宏 F1（每类 F1 的平均）而非准确率。用平衡采样或类别权重缓解不平衡。

### 错误 2：SpecAugment 忘记应用

**现象：** 训练准确率 98%，验证 82%。过拟合严重。

**原因：** 训练集太小——没用 SpecAugment 就是直接在小数据上过拟合。SpecAugment 在训练时随机掩蔽时间和频率带——等效于把数据集扩大了数倍。

**修复：** 在所有音频分类训练中加入 SpecAugment（时间掩蔽 20-30 帧，频率掩蔽 10-20 帧）。它是防止过拟合的第一道防线。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| AudioSet | "音频的 ImageNet" | Google 的 200 万片段、632 类弱标注 YouTube 数据集 |
| ESC-50 | "小分类基准" | 50 类 × 40 个环境音片段 |
| AST | "音频频谱图 Transformer" | 在 log-mel patch 上的 ViT；2021-2024 SOTA |
| BEATs | "自监督音频" | 微软模型，iter3 在 2026 领先 AudioSet |
| Mixup | "配对增强" | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2` |
| SpecAugment | "掩蔽增强" | 在频谱图上随机遮盖时间带和频率带 |
| mAP | "多标签的主要指标" | 跨类别和阈值的平均精度均值 |

---

## 📚 小结

音频分类的基线从 k-NN on MFCC 均值（出人意料地强）一路演化到 BEATs 自监督微调（97% ESC-50）。**类别不平衡是 80% 问题的根源——不是模型架构。** SpecAugment + Mixup 是防止过拟合的第一道防线。2026 年的默认决策规则：冻结 BEATs 骨干，微调分类头，几小时内达到 SOTA 的 95%。

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——在合成 4 类数据集上训练 k-NN MFCC 基线。报告混淆矩阵。

2. 【实验】将 `summarize` 改为 [均值, 方差, 偏度, 峰度]。在相同合成数据集上对比 4 矩池化 vs 均值+方差。哪个更好？

3. 【实验】用 `torchaudio` 在 ESC-50 fold 1 上训练 2D CNN。报告 5 折交叉验证准确率。加入 SpecAugment（时间掩蔽=20，频率掩蔽=10）并报告差值。

4. 【思考】你的音频分类系统在训练集上准确率 99%，但在新录音上只有 70%。分析最可能的原因——是模型问题还是数据问题？给出具体的排查步骤。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 分类器设计提示词 | `outputs/skill-classifier-designer.md` | 按任务选择架构、增强、类别平衡和评估指标 |

---

## 📖 参考资料

1. [论文] Gong, Chung, Glass. "AST: Audio Spectrogram Transformer". Interspeech, 2021. https://arxiv.org/abs/2104.01778 — 2021-2024 架构
2. [论文] Chen et al. "BEATs: Audio Pre-Training with Acoustic Tokenizers". ICML, 2024. https://arxiv.org/abs/2212.09058 — 2024+ 默认
3. [论文] Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR". Interspeech, 2019. https://arxiv.org/abs/1904.08779 — 主流音频增强
4. [数据集] ESC-50. https://github.com/karolpiczak/ESC-50 — 50 类基准
5. [数据集] AudioSet. https://research.google.com/audioset/ — 632 类 YouTube 分类

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
