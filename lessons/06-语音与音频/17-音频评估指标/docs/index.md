# 音频评估指标

> 语音质量的评估不是选择一个数字，而是选择一组互补的视角。PESQ 衡量感知质量，STOI 衡量可懂度，FAD 衡量分布距离，UTMOS 预测人类评分。2026 年每个生产系统都需要多指标组合。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 06 · 02（频谱图）、阶段 06 · 04（ASR）、阶段 06 · 07（TTS）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 计算 PESQ、STOI、FAD、UTMOS——理解每个指标衡量什么、值域、优劣方向
- [ ] 根据任务类型选择评估指标组合——语音质量 vs ASR vs 音乐生成
- [ ] 解释为什么单一指标无法全面反映音频系统性能

---

## 1. 问题

评估一个 TTS 系统——"输出自然吗？"。但"自然"是多维的：语音清晰度、说话人音色一致性、情感表达、背景噪声、音乐结构。单一指标无法捕捉全部维度。2026 年每个音频系统都需要指标组合。

---

## 2. 概念

### 2.1 六大指标

| 指标 | 衡量 | 范围 | 越高越好 | 适用 |
|---|---|---|---|---|
| PESQ | 感知语音质量 | 1.0-4.5 | ✓ | 语音合成/通话 |
| STOI | 短时可懂度 | 0.0-1.0 | ✓ | 语音合成/降噪 |
| FAD | 生成分布距离 | 0-∞ | ✗ | 音乐生成/音效 |
| UTMOS | 主观音质预测 | 0-5 | ✓ | 语音合成/音乐 |
| WER | 转录错误率 | 0%-100% | ✗ | ASR |
| CER | 字符错误率 | 0%-100% | ✗ | ASR |

### 2.2 关键数字（2026 SOTA）

**语音合成：**
- PESQ > 3.5 = "好质量"，> 4.0 = "接近原始"
- UTMOS 3.8+ = 开源系统可用（Kokoro 3.87，F5-TTS 3.95）

**语音识别：**
- WER < 5% = 可接受，< 2% = 人类水平
- LibriSpeech test-clean SOTA：1.4%（Parakeet）

**音乐生成：**
- FAD < 3.0 = 生成分布接近真实
- UTMOS 3.8+ = 人类偏好高（Suno v5 ELO 1293）

---

## 3. 从零实现

```python
import math

def pesq_score(original, degraded):
    """PESQ 模拟——真实版用 ITU-T P.862 标准实现（pesq 库）。"""
    mse = sum((a - b) ** 2 for a, b in zip(original, degraded)) / len(original)
    return max(1.0, min(4.5, 3.5 - 10 * math.log10(max(mse, 1e-10))))

def stoi_score(original, degraded, sr=16000):
    """STOI 模拟——真实版用 pystoi 库。"""
    corrs = []
    chunk_size = sr // 50
    for i in range(0, min(len(original), len(degraded)), chunk_size):
        orig = original[i:i+chunk_size]
        deg = degraded[i:i+chunk_size]
        if len(orig) < 2 or len(deg) < 2: continue
        mo, md = sum(orig)/len(orig), sum(deg)/len(deg)
        cov = sum((o-mo)*(d-md) for o, d in zip(orig, deg))
        vo, vd = sum((o-mo)**2 for o in orig), sum((d-md)**2 for d in deg)
        if vo > 0 and vd > 0: corrs.append(cov / math.sqrt(vo * vd))
    return sum(corrs) / len(corrs) if corrs else 0.0
```

完整代码见 `code/main.py`。

---

## 4. 工具选择

| 工具 | 用途 | 适用 |
|---|---|---|
| pesq | PESQ 计算 | 语音质量评估 |
| pystoi | STOI 计算 | 可懂度评估 |
| FAD（panns） | Fréchet 音频距离 | 音乐/音效生成 |
| UTMOS | 主观音质预测 | 语音/音乐生成 |
| jiwer | WER/CER 计算 | ASR 评估 |
| MUSHRA | 主观音频质量测试 | 高精度人工评估 |

---

## 5. 知识连线

- **阶段 06 · 07（TTS）→** TTS 系统用 PESQ + UTMOS 评估——PESQ > 3.5 = "好质量"
- **阶段 06 · 09（音乐生成）→** FAD 是音乐生成的核心指标——衡量生成分布与真实分布的距离
- **阶段 06 · 04（ASR）→** WER 是 ASR 的标准指标——SOTA 1.4%（Parakeet）

---

## 6. 常见错误

### 错误 1：仅用 WER 评估 ASR

**现象：** WER 3% 但用户抱怨"听不懂"。

**原因：** WER 只衡量转录准确性，不衡量听感质量。高 WER 的系统可能通过 STOI 仍然可懂——低 WER 的系统可能因为重复和幻觉而用户体验差。

**修复：** 同时报告 WER + CER（字符级）+ 插入/删除/替换分解 + 人类可懂度评分。

### 错误 2：FAD 低但音质差

**现象：** FAD = 2.0（很好）但用户说"这段音乐听起来像是在水下"。

**原因：** FAD 衡量的是分布距离——生成分布与真实分布在统计上接近。但单个样本的音质（高频丢失、时序失真）无法通过分布指标捕捉。需要 UTMOS 或人类评估补充。

---

## 7. 面试考点

### Q1：PESQ 和 STOI 的核心区别是什么？（难度：⭐⭐）

**参考答案：** PESQ 衡量**感知语音质量**——人类听觉对失真的感知。STOI 衡量**可懂度**——语音中信息是否足够让人类理解内容。PESQ > 3.5 的系统听起来"好"但不一定"清晰"；STOI > 0.9 的系统听起来"清晰"但不一定"自然"。生产 TTS 需要同时满足两个指标。

### Q2：为什么 FAD 值低但用户说音质差？（难度：⭐⭐⭐）

**参考答案：** FAD 衡量的是分布距离——全局统计相似性。两个分布可以有相同的均值和方差（FAD 低），但单个样本可能有严重的局部失真（高频丢失、节奏错乱、爆破音模糊）。FAD 是"宏观健康度"指标，但无法捕获"微观细节"。补充 UTMOS（预测主观音质）或 MUSHRA（人类评估）来捕捉单样本质量。

---

## 📚 小结

PESQ（感知质量）、STOI（可懂度）、FAD（分布距离）、UTMOS（主观音质）——每个衡量不同的维度。生产系统必须同时报告多个指标。PESQ > 3.5 + UTMOS 3.8+ = TTS 可用。FAD < 3.0 = 音乐生成分布接近真实。单一指标无法全面反映音频系统性能。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 指标选择提示词 | `outputs/skill-metric-picker.md` | 按任务选择评估指标组合 |

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| PESQ | 感知语音质量评估——预测人类听觉感知的失真程度。ITU-T P.862 标准 |
| STOI | 短时客观可懂度——衡量语音中信息是否足以让人类理解内容 |
| FAD | Fréchet 音频距离——生成音频与真实音频分布的距离，越低越好 |
| UTMOS | 音频质量预测模型——预测人类 MOS 评分，0-5 分 |
| MUSHRA | 主观音频质量测试——人类评估者对比多个系统，最高精度但成本最高 |

---

## ✏️ 练习

1. 【实验】用 `pesq` 库评估不同比特率 MP3 压缩后的语音质量。找出可接受的最低比特率。
2. 【对比】用 UTMOS 对比 F5-TTS、Kokoro、VITS 的输出——哪个在中文上得分最高？

---

## 📖 参考资料

1. [标准] ITU-T P.862 — PESQ 标准
2. [库] pesq. https://github.com/ludlown/qualitymetrics — PESQ Python 实现
3. [库] pystoi. https://github.com/MuqiT/PySTOI — STOI Python 实现
4. [库] FAD. https://github.com/google-research/google-research/tree/master/gan_evaluations — FAD 计算
5. [库] jiwer. https://github.com/jitsi/jiwer — WER/CER 计算

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
