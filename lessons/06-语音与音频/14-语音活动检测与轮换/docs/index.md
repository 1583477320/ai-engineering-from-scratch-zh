# 语音活动检测与轮换——Silero、Cobra 与 Flush Trick

> 每个语音助手的生死取决于两个判断：用户现在是否在说话，用户是否说完了。VAD 回答第一个问题，轮换检测（VAD + 静默挂起 + 语义端点模型）回答第二个。任何一个判断错误，你的助手要么打断用户，要么永远闭嘴。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 06 · 11（实时音频处理）、阶段 06 · 12（语音助手流水线）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 VAD 三级流水线——能量门控 → Silero VAD → 语义端点检测
- [ ] 配置端点检测参数——阈值、最小语音时长、静默挂起、预滚动缓冲
- [ ] 解释 Flush trick 如何将 STT 端到端延迟从 500ms 降到 125ms

---

## 1. 问题

语音助手在每个 20ms 帧上要做三个判断：

1. **这帧是否有语音？** —— VAD，二分类，逐帧
2. **用户是否开始了一段新话？** —— onset 检测
3. **用户是否说完了？** —— end-pointing（轮换端点检测）

朴素答案（能量阈值）在任何噪声中都会失败——交通、键盘、人群嘈杂。2026 年的答案：Silero VAD（开源深度学习）+ 语义端点检测模型 + VAD 校准的静默挂起。

---

## 2. 概念

### 2.1 VAD 三级级联

**第一层：能量门控。** 最便宜。RMS 能量阈值 -40 dBFS。过滤明显静默，但对任何高于阈值的噪声都会触发。

**第二层：Silero VAD**（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。单 CPU 线程处理 30ms 帧 < 1ms。TPR 87.7% @ 5% FPR。开源默认。

**第三层：语义轮换检测器。** LiveKit 的轮换检测模型（2024-2026）或自建小型分类器。区分"句子中暂停"和"说完了"。使用语言上下文（语调+最近词），不只是静默。

### 2.2 关键参数

- **阈值。** Silero 输出概率；> 0.5（默认）或 > 0.3（敏感）判定为语音。越低 = 越少首词截断，越多假阳性
- **最小语音时长。** 拒绝短于 250ms 的语音——通常是咳嗽或椅子噪音
- **静默挂起（端点检测）。** VAD 回到非语音后等待 500-800ms 再宣布话轮结束。太短 → 打断用户；太长 → 感觉迟钝
- **预滚动缓冲。** 在 VAD 触发前保留 300-500ms 音频。防止"嗨"被截掉

### 2.3 Flush Trick（Kyutai 2025）

流式 STT 有前瞻延迟（Kyutai STT-1B 500ms，STT-2.6B 2.5s）。通常需要等这么久。Flush trick：当 VAD 触发端点时——**发送 flush 信号给 STT**，强制立即输出。STT 以 ~4 倍实时速度处理 500ms 缓冲 → 实际只需 ~125ms。

**端到端：** 125ms VAD + flush STT = 对话级延迟。

---

## 3. 从零实现

```python
def energy_gate(signal, threshold_db=-40):
    """第一层：RMS 能量阈值。"""
    rms = math.sqrt(sum(s*s for s in signal) / max(1, len(signal)))
    db = 20 * math.log10(max(rms, 1e-10))
    return db > threshold_db

def simulate_vad(signal, threshold=0.5, chunk_size=160):
    """第二层：模拟 Silero VAD——每块输出语音概率。"""
    results = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        energy = sum(s*s for s in chunk) / max(1, len(chunk))
        prob = min(1.0, energy * 10)
        results.append(prob > threshold)
    return results

def turn_detection(vad_outputs, hangover_chunks=25):
    """第三层：端点检测——VAD 返回非语音后等待 hangover。"""
    turns, current, silent = [], False, 0
    for speech in vad_outputs:
        if speech and not current:
            current, silent = True, 0
            turns.append("开始说话")
        elif not speech and current:
            silent += 1
            if silent >= hangover_chunks:
                turns.append("结束说话")
                current = False
        else:
            silent = 0
    return turns
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

| 工具 | 用途 | 适用 |
|---|---|---|
| Silero VAD 4.0 | VAD，< 1ms/帧 CPU | 2026 开源默认 |
| webrtcvad | VAD 备选 | Google 实现 |
| Pyannote VAD | VAD | 高精度 |
| LiveKit turn-detection | 语义端点检测 | 流式语音助手 |

---

## 5. 知识连线

- **阶段 06 · 11（实时音频）→** VAD 是第 11 课"实时语音助手"流水线中的关键门控组件
- **阶段 06 · 12（语音助手）→** 打断处理依赖本课的端点检测——静默挂起参数直接影响打断响应时间

---

## 6. 常见错误

### 错误 1：能量门控作为唯一 VAD

**现象：** 在有背景噪声的环境中（咖啡厅、街道），VAD 持续误触发——助手以为没人说话时突然打断。

**原因：** 能量门控无法区分语音能量和噪声能量。键盘敲击、空调声、背景人声都会超过 -40 dBFS 阈值。

**修复：** 必须叠加 Silero VAD——深度学习模型能区分语音和非语音能量。能量门控仅作为第一层快速过滤。

### 错误 2：静默挂起设置过短

**现象：** 用户一句话还没说完，助手就开始响应。

**原因：** 静默挂起 < 300ms——用户在思考时的短暂停顿被误判为话轮结束。

**修复：** 最小静默挂起设为 500ms。对英语和中文——自然思考停顿通常在 300-800ms，超过这个范围才是真正的结束。

---

## 7. 面试考点

### Q1：Silero VAD 和 WebRTC VAD 的主要区别是什么？（难度：⭐⭐）

**参考答案：** WebRTC VAD 是基于规则的能量检测——在 30ms 帧上计算 RMS 能量并与阈值比较。在安静环境中效果好，但无法区分语音和非语音噪声。Silero VAD 是深度学习模型（100 万参数），在 6000+ 种语言上训练——能区分语音和噪声，TPR 87.7%（WebRTC VAD 仅 50%）。Silero 延迟 < 1ms，适合实时应用；WebRTC VAD 30ms，延迟较高。

---

## 📚 小结

VAD 是语音助手的"耳朵"——三级流水线（能量门 Silero VAD 语义端点）确保用户说开始时助手开始听，说完时助手开始答。Silero VAD 4.0 以 < 1ms 延迟成为 2026 年开源默认。Flush trick 将 STT 延迟从 500ms 降到 125ms——对对话流畅性至关重要。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| VAD 配置提示词 | `outputs/skill-vad-configurator.md` | 按应用和环境配置 VAD 参数 |

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| VAD | 语音活动检测——判断当前帧是否有语音（二分类） |
| 端点检测 | 判断用户是否说完——静默挂起后宣布话轮结束 |
| 预滚动缓冲 | VAD 触发前保留的音频——防止首词截断 |
| Flush trick | STT 缓冲强制立即输出——将延迟从 500ms 降到 125ms |

---

## ✏️ 练习

1. 【实现】用 `silero-vad` 对 10 秒录音进行逐帧检测，输出每帧的语音概率。绘制概率曲线。
2. 【实验】调整静默挂起参数（300ms/500ms/800ms），在 20 句自然对话中测试端点检测准确率。

---

## 📖 参考资料

1. [论文] Silero Team. "Silero VAD". 2020-2024. https://github.com/snakers4/silero-vad
2. [论文] Kyutai. "Flush Trick for Streaming STT". 2025.
3. [论文] Pyannote Audio. "Silero VAD". https://github.com/snakers4/silero-vad

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
