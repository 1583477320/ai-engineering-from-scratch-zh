# 说话人识别与验证

> ASR 问"说了什么？"说话人识别问"谁说的？"数学看起来一样——嵌入 + 余弦——但每个生产决策都依赖一个 EER 数字。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 05 · 22（嵌入模型） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现说话人验证流水线——MFCC 统计嵌入 + 余弦相似度 + EER 计算
- [ ] 区分验证（1:1）、识别（1:N）和开集识别——各自的部署场景
- [ ] 理解 ECAPA-TDNN 为什么在 2026 年仍然主导——15M 参数、0.87% EER 的质量-效率平衡

---

## 1. 问题

用户说了一个口令。你需要知道：这个人是他们声称的那个人吗（**验证**，1:1），还是注册库中的第一个人（**识别**，1:N）？还是两者都不是——一个未知说话人（**开集**）？

2018 年之前：GMM-UBM + i-vectors。合理但对信道偏移（手机 vs 笔记本）和情绪脆弱。2018-2022：x-vector（TDNN 骨干 + 角度间隔损失）。2022+：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域被三个模型和一个指标主导。

**EER（等错误率）** 是那个指标——将决策阈值设为错误接受率 = 错误拒绝率时的交叉点。每个论文、每个排行榜、每个采购电话都用它。

---

## 2. 概念

### 2.1 注册-验证流水线

```
注册: 5-30秒语音 → 嵌入器 → 192d/256d 嵌入
验证: 测试语音 → 嵌入器 → 余弦相似度 → 阈值 → 通过/拒绝
```

### 2.2 三大模型

| 模型 | VoxCeleb1-O EER | 参数量 | 特点 |
|---|---|---|---|
| ECAPA-TDNN | 0.87% | 15M | 2026 年生产默认——质量-效率最优平衡 |
| WavLM-SV | 0.42% | 316M | 最高质量——SSL 预训练 + AAM 微调 |
| ReDimNet (2024) | 0.39% | 24M | 新 SOTA——比 ECAPA 好但更复杂 |

### 2.3 EER——等错误率

在决策阈值处 FAR（错误接受率）= FRR（错误拒绝率）时的数值。越低越好。EER=1% 意味着在 FAR=FRR 的平衡点，每 100 次匹配中出 1 次错误。

### 2.4 评分方法

- **余弦相似度**：嵌入向量的点积。简单、快、生产中最常用
- **PLDA**：概率线性判别分析——将嵌入投影到潜在空间，计算似然比。比余弦好 10-20% EER，但需要训练数据
- **分数归一化**：S-norm / AS-norm——针对冒名者人群的均值和标准差归一化

---

## 3. 从零实现

### Step 1：合成说话人 + MFCC 嵌入

```python
def embed_mfcc_stats(signal, sr):
    """从 MFCC 均值+标准差构建说话人嵌入。"""
    frames = featurize(signal, sr)
    n = len(frames[0])
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(n)]
    std = [math.sqrt(sum((f[i]-mean[i])**2 for f in frames)/len(frames)) for i in range(n)]
    return l2_normalize(mean + std)  # 26 维：13 均值 + 13 标准差
```

### Step 2：注册 + 验证 + EER

```python
def eer(same_scores, diff_scores):
    """在 FAR=FRR 的交叉点计算 EER。"""
    thresholds = sorted(set(same_scores + diff_scores))
    best_gap = float("inf")
    best_eer = 1.0
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < best_gap:
            best_gap = abs(fa - fr)
            best_eer = (fa + fr) / 2
    return best_eer
```

完整代码见 `code/main.py`——合成说话人 + 嵌入 + EER 计算。

---

## 4. 工业工具

| 场景 | 选择 |
|---|---|
| 通用说话人验证 | ECAPA-TDNN（15M，EER 0.87%） |
| 最高质量 | WavLM-SV large（316M，EER 0.42%） |
| 设备端、极低延迟 | x-vector（5M） |
| 说话人分离（diarization） | pyannote-audio 3.1 |
| 低资源语言 | ECAPA-TDNN 在 VoxMultilingual 上微调 |

---

## 5. 知识连线

- **阶段 06 · 02（频谱图）→** 说话人嵌入的输入是第 02 课的 MFCC 或 log-mel 频谱图
- **阶段 06 · 03（音频分类）→** 说话人识别本质上是一个分类问题——但类别数可以是数万（VoxCeleb 1211 人）
- **阶段 05 · 22（嵌入模型）→** WavLM-SV 从预训练的 SSL 嵌入微调——与文本中的 Word2Vec→BERT 微调路径平行

---

## 6. 常见错误

### 错误 1：注册音频太短或信道不匹配

**现象：** EER 在训练集上 1%，但跨信道（手机 vs 笔记本）部署后飙升到 15%。

**原因：** 说话人嵌入对声学环境敏感——信道偏移（不同麦克风、混响、距离）会改变嵌入向量的方向。5 秒注册音频可能只在安静环境中有效。

**修复：** 注册多条不同信道的音频（静音室 + 手机 + 笔记本）。对嵌入做信道归一化（CMVN）。在跨信道测试集上校准阈值。

### 错误 2：只报告 EER 不报告 FAR@FRR

**现象：** EER=1% 看起来很好——但安防场景需要 FAR<0.1%。在 FAR=0.1% 处 FRR 可能是 10%——用户体验差。

**原因：** EER 只是 FAR=FRR 的一个平衡点。生产系统通常要求固定 FRR（如 2%）并衡量 FAR——这才是实际部署的性能。

**修复：** 同时报告 EER + FAR@1%FRR + FRR@1%EER。安全等级越高的场景，FAR 预算越紧。

---

## 7. 面试考点

### Q1：EER 为什么是说话人验证的核心指标？它有哪些局限？（难度：⭐⭐）

**参考答案：**
EER 的优势是单数字——一个系统 vs 另一个系统，直接比较 EER 就知道谁更好。它在 FAR=FRR 的平衡点，对称地衡量了两种错误。局限：（1）生产系统通常有不对称的安全约束——安防系统 FAR<0.1% 比 EER 重要得多；（2）EER 不反映分数分布的形状——两个系统可以有相同 EER 但完全不同的 ROC 曲线；（3）EER 对数据集的信道分布敏感——同一个系统在不同测试集上 EER 差异可以很大。

### Q2：为什么 ECAPA-TDNN 在 2026 年仍然优于更大的模型？（难度：⭐⭐⭐）

**参考答案：**
ECAPA-TDNN（15M 参数）在 VoxCeleb1-O 上达到 0.87% EER——而 WavLM-SV（316M 参数）是 0.42% EER。差距只有 0.45% EER，但模型大小差 20 倍。在生产中：（1）ECAPA-TDNN 在单个 CPU 核上处理 100 毫秒的语音只需几毫秒——WavLM-SV 需要 GPU 且慢 10-100 倍；（2）0.87% EER 对绝大多数应用场景已经足够——只有对安全性要求极高的场景才需要 WavLM-SV 的额外 0.45%；（3）ECAPA-TDNN 更容易量化（int8/int4），进一步缩小部署体积。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| EER | "等错误率" | FAR=FRR 的交叉点——越低越好 |
| 验证（1:1） | "这个人是谁吗？" | 检查语音是否属于特定注册用户 |
| 识别（1:N） | "这是谁？" | 在注册库中找最匹配的说话人 |
| AAM-softmax | "带角度间隔的损失" | 在角度空间中强制类间分离——cos(θ + m)，m=0.2 |
| 注册音频 | "教系统认识说话人" | 5-30秒的语音样本，用于提取说话人嵌入 |
| 信道归一化 | "消除录音设备影响" | CMVN / S-norm / AS-norm——让不同麦克风的分数可比 |

---

## 📚 小结

说话人验证 = 嵌入（MFCC/ECAPA/WavLM）+ 余弦相似度 + 阈值决策。EER 是核心指标——EER 越低越好。ECAPA-TDNN 在 2026 年仍然是生产默认——15M 参数、0.87% EER，在质量和效率之间达到了最佳平衡。**同时报告 EER 和 FAR@1%FRR**——生产系统通常有不对称的安全约束。

跨信道、跨情绪的鲁棒性仍然是主要挑战——多条件注册和分数归一化是缓解手段。零样本说话人验证（无需注册）是 2024-2026 的新方向——ECAPA + 无监督聚类或 LLM 辅助。

---

## ✏️ 练习

1. 【实现】运行 `code/main.py`——在合成数据上计算 EER。对比不同噪声水平下的 EER 变化。

2. 【实现】为 5 个合成说话人构建嵌入池（每人 5 段），实现 1:N 说话人识别（最相似嵌入匹配）。报告识别准确率。

3. 【实验】用 `speechbrain` 在 VoxCeleb1 上训练 ECAPA-TDNN。报告 5 折交叉验证的 EER。对比不同注册时长（5s vs 15s vs 30s）的影响。

4. 【思考】你的说话人验证系统在安静环境 EER=0.5%，在嘈杂餐厅环境 EER=12%。分析原因——是模型问题还是预处理问题？设计两个修复方案。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 说话人设计提示词 | `outputs/skill-speaker-designer.md` | 按场景选择模型、注册策略和评分方法 |

---

## 📖 参考资料

1. [论文] Desplanques et al. "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN". Interspeech, 2020. https://arxiv.org/abs/2005.07143
2. [论文] Chen et al. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing". 2022. https://arxiv.org/abs/2110.13900
3. [数据集] VoxCeleb 1/2. https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
