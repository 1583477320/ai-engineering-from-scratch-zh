# 实时音频处理

> 批处理流水线处理一个文件。实时流水线在下一个 20 毫秒到来之前处理下一个 20 毫秒。每一个对话 AI、广播演播室、电话机器人都存活和死亡于这个延迟预算。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 06 · 02（频谱图）、阶段 06 · 04（ASR）、阶段 06 · 07（TTS）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出对话式 AI 的完整延迟预算：收音→VAD→ASR→LLM→TTS→播放，总计 < 500ms
- [ ] 理解环形缓冲区、VAD 门控、打断检测在实时流水线中的作用
- [ ] 说明 2026 年 Moshi（200ms）和 GPT-4o-realtime（320ms）的技术差异

---

## 1. 问题

你想要一个感觉"活着的"语音助手。人类对话轮换延迟 ~230ms（静默到回应）。超过 500ms 感觉像机器人；超过 1500ms 感觉坏了。2026 年的完整**听→理解→响应→说话**循环预算：

| 阶段 | 预算 |
|---|---|
| 麦克风→缓冲区 | 20 ms |
| VAD（语音活动检测） | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首 token） | 100 ms |
| TTS（首个 chunk） | 100 ms |
| 渲染→扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai, 2024）做到了 200ms 全双工。GPT-4o-realtime（2024）约 320ms。2022 年的流水线级联需要 2500ms。10 倍改进来自三个技术：(1) 全链路流式化；(2) 异步流水线 + 部分结果；(3) 可中断生成。

---

## 2. 概念

### 2.1 帧/块/窗口

实时音频以固定大小块流动。常见选择：20ms（16kHz 下 320 个采样）。所有下游处理必须跟上这个节奏。

### 2.2 关键组件

- **环形缓冲区：** 固定大小循环缓冲区。生产者线程写新帧，消费者线程读取。防止热路径分配。大小 ≈ 最大延迟 × 采样率
- **VAD（语音活动检测）：** 在无人说话时门控下游工作。Silero VAD 4.0 在 CPU 上 < 1ms/30ms 帧
- **流式 ASR：** 音频到达时即输出部分转录。Parakeet-CTC-0.6B 流式模式 2-5% WER @ 320ms 延迟
- **打断检测：** 用户在助手说话时开口 → 必须在 100ms 内 (a) 检测到打断 (b) 停止 TTS (c) 取消 LLM 剩余输出
- **WebRTC Opus 传输：** 20ms 帧，48kHz，自适应码率 8-128 kbps。浏览器和移动设备标准
- **抖动缓冲区：** 网络包乱序/延迟到达时重排序和平滑。60-80ms 典型值

---

## 3. 从零实现

### 帧预算计算

```python
def ring_buffer_size(max_latency_ms, sample_rate=16000):
    """环形缓冲区大小 = 最大延迟 × 采样率"""
    return int(max_latency_ms / 1000 * sample_rate)

# 200ms 延迟 @16kHz → 3200 采样点
```

### 延迟分解

| 阶段 | 2026 预算 | 2022 参考 | 改进来源 |
|---|---|---|---|
| 麦克风→缓冲区 | 20ms | 50ms | 原生 AudioUnit/ALSA |
| VAD | 10ms | 50ms | Silero VAD（< 1ms/帧） |
| ASR | 150ms | 800ms | 流式 Parakeet/Whisper-streaming |
| LLM 首 token | 100ms | 500ms | 流式推理 + 模型优化 |
| TTS 首 chunk | 100ms | 800ms | 流式声码器 + 预热 |
| 渲染→扬声器 | 20ms | 30ms | PortAudio + 低延迟 DAC |

### 关键代码模式

```python
import asyncio

async def real_time_loop():
    """三步流水线：捕获 → 处理 → 播放"""
    async with MicrophoneStream(sr=16000, chunk_ms=20) as stream:
        async for frame in stream:
            if vad.detect(frame):           # VAD 门控
                transcript = await asr.transcribe(frame)
                if transcript:
                    llm_response = await llm.generate(transcript)
                    await tts.stream(llm_response)  # 流式 TTS
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

| 工具 | 用途 | 推荐场景 |
|---|---|---|
| Silero VAD 4.0 | 语音活动检测，CPU < 1ms/帧 | 所有实时场景 |
| webrtcvad | Google VAD 实现，备选 | 已有 WebRTC 基础设施 |
| sounddevice | 跨平台音频捕获/播放 | Python 开发 |
| PortAudio | C 音频库 | 生产环境、延迟敏感 |
| LiveKit / Daily.co | WebRTC 实时音视频基础设施 | 流式语音助手部署 |
| WebRTC AEC3 | 回声消除 | 有扬声器输出时必须 |

---

## 5. 知识连线

- **阶段 06 · 01（音频基础）→** 采样率 16kHz 和帧结构是实时处理的基础——所有组件必须匹配
- **阶段 06 · 04（ASR）→** 流式 ASR 的输入正是第 04 课的 log-mel 频谱图
- **阶段 06 · 07（TTS）→** 流式 TTS 是第 07 课模型的流式变体——首 chunk 延迟是关键瓶颈
- **阶段 06 · 14（VAD）→** VAD 是实时流水线的门控组件——下一课详解

---

## 6. 常见错误

### 错误 1：Python GIL 饥饿音频线程

**现象：** 音频出现卡顿/丢帧——听起来像网络抖动。

**原因：** Python 的 GIL + 重模型（LLM/TTS）占用了 CPU，音频线程被饿死。音频处理需要严格的实时保证，不能被其他线程阻塞。

**修复：** 使用 C 回调音频库（`sounddevice`、PortAudio），Python 不在热路径上。LLM/TTS 推理放单独进程。使用 `asyncio` 或 `multiprocessing` 避免 GIL 竞争。

### 错误 2：TTS 首次调用延迟

**现象：** 第一次对话响应特别慢（延迟翻倍），后续正常。

**原因：** TTS 模型首次请求时有 100-200ms 热启动——模型加载 + CUDA 缓存预热。这个延迟在冷启动后消失。

**修复：** 系统启动时用哑数据运行一次 TTS（预热），确保推理引擎完全初始化。Kokoro 预热约 300ms，F5-TTS 约 200ms。

### 错误 3：采样率转换延迟累积

**现象：** 多个组件各自重采样，累加延迟超过预算。

**原因：** ASR 需要 16kHz、TTS 输出 24kHz、播放需要 48kHz——每次重采样累积 5-20ms。

**修复：** 选定一个内部采样率（如 24kHz），所有组件在该采样率上工作。输入端一次性重采样到目标采样率，输出端一次性重采样到播放采样率。

---

## 7. 面试考点

### Q1：实时语音助手的完整延迟预算是多少？（难度：⭐⭐）

**参考答案：** 2026 年笔记本 CPU 上完整"听→理解→响应→说话"循环目标 < 800ms。分解：麦克风→缓冲 20ms + VAD 10ms + 流式 ASR 150ms + LLM 首 token 100ms + TTS 首 chunk 100ms + 渲染 20ms = ~400ms。还有余量。Moshi 全双工做到 200ms——因为它把 ASR+LLM+TTS 合并到单个模型，消除了组件间切换。GPT-4o-realtime 约 320ms。2022 年的级联流水线需要 2500ms——10 倍改进来自全链路流式化和异步流水线。

### Q2：打断检测为什么必须在 100ms 内完成？（难度：⭐⭐）

**参考答案：** 打断检测需在 100ms 内完成三个动作：VAD 检测到用户开口 + 停止 TTS 播放 + 取消 LLM 剩余输出。如果超过 100ms——用户会听到助手在自己说完之后还在继续说话——这是"助手聋了"的体验。Moshi 全双工架构（200ms 端到端）天然解决了这个问题，因为它是单模型而非级联流水线——没有组件间切换。

### Q3：实时流式 ASR 的"部分转录"和"最终转录"有什么区别？为什么都需要？（难度：⭐⭐⭐）

**参考答案：** 部分转录（partial）是 ASR 在音频到达时实时输出的"不完整"结果——它可能被后续输入修正。最终转录（final）是模型确认一个停顿后的"确定"结果。两者都需要：部分转录让 LLM 可以提前开始推理（降低延迟），最终转录用于精确的后处理和归档。部分转录的典型 WER 比最终高 2-5%，但延迟低 100-200ms——这正是 400ms 延迟预算中的关键权衡。

---

## 📚 小结

实时音频处理的核心是延迟预算——完整"听→理解→响应→说话"循环目标 < 800ms（笔记本 CPU）。2026 年的关键技术：环形缓冲区（20ms）、Silero VAD（< 1ms）、流式 ASR（150ms）、流式 TTS（100ms）。Moshi 已做到 200ms 全双工。10 倍改进来自全链路流式化、异步流水线、可中断生成。三个陷阱：Python GIL 饥饿音频线程、TTS 首次热启动、采样率转换延迟累积。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 实时延迟优化提示词 | `outputs/skill-latency-optimizer.md` | 按流水线配置诊断和优化延迟瓶颈 |

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| TTFT | "首 token 延迟" | LLM 生成第一个 token 所需时间——语音助手体验的核心指标 |
| VAD | "语音检测" | 语音活动检测——判断当前是否有语音，门控下游处理以节省资源 |
| Barge-in | "打断" | 用户在助手说话时开口 → 系统必须在 100ms 内停止 TTS、取消 LLM |
| Ring Buffer | "环形缓冲区" | 固定大小循环缓冲区——防止实时路径中的内存分配 |
| WebRTC Opus | "实时音频格式" | 20ms 帧、48kHz、8-128kbps 自适应码率的实时音频传输标准 |

---

## ✏️ 练习

1. 【理解】画出完整的"听→理解→响应→说话"流水线，标注每个阶段的延迟预算。用中文注释每个组件的作用。
2. 【实验】用 `sounddevice` 录制 10 秒音频，每 20ms 打一个时间戳，验证帧率是否稳定。用 `silero-vad` 添加 VAD，打印语音起止时间。
3. 【实现】搭建一个简化版实时流水线：sounddevice 录音 → silero-vad → 打印转录。测量从语音结束到打印结果的延迟。
4. 【思考】Moshi 为什么能用 200ms 实现全双工，而级联流水线需要 2500ms？从架构差异上分析。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 实时延迟优化提示词 | `outputs/skill-latency-optimizer.md` | 按流水线配置诊断和优化延迟瓶颈 |

---

## 📖 参考资料

1. [论文] Défossez et al. "Moshi: a speech-text foundation model for real-time dialogue". Kyutai, 2024.
2. [项目] Pipecat. https://github.com/foundation-models/foundation-model-frameworks — 开源实时语音 AI 框架
3. [项目] LiveKit. https://livekit.io — 实时音视频基础设施
4. [论文] WebRTC Audio Processing. https://webrtc.org/ — 实时音频传输标准

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、工程最佳实践、常见错误、面试考点等均为原创内容。
