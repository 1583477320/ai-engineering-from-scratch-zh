# 语音助手流水线——阶段 06 的毕业设计

> 第 01-11 课的一切内容，缝合在一起。构建一个能听、能想、能说的语音助手。在 2026 年这已经是一个已解决的工程问题，而非研究问题——但集成细节决定了能否交付。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 06 · 04, 05, 06, 07, 11；阶段 11 · 09（函数调用）；阶段 14 · 01（智能体循环）
**时间：** ~120 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 搭建完整的七组件语音助手流水线——麦克风→VAD→STT→LLM+工具→TTS→扬声器
- [ ] 识别三个常见的部署陷阱——首词截断、中途打断混乱、静默幻觉
- [ ] 对比 2026 年的三种生产级流水线——LiveKit+Deepgram、Pipecat+Whisper、Moshi 全双工

---

## 1. 问题

构建一个端到端助手：

1. 捕获麦克风输入（16kHz 单声道）
2. 检测用户说话的开始/结束
3. 流式转录
4. 将转录传递给可调用工具的 LLM（计时器、天气、日历）
5. 将 LLM 文本流式传入 TTS
6. 播放音频回给用户
7. 用户打断时停止

**延迟目标：** 笔记本 CPU 上用户说完话后 800ms 内输出首字节 TTS 音频。**质量目标：** 无丢词、无静默幻觉字幕、无声音克隆泄露、无提示注入成功。

---

## 2. 概念

### 2.1 七组件流水线

| 组件 | 实现 |
|---|---|
| 音频捕获 | 麦克风→16kHz 单声道→20ms 块。Python 用 `sounddevice`；生产用原生 AudioUnit/ALSA/WASAPI |
| VAD | Silero VAD 阈值 0.5，最小语音 250ms，静默挂起 500ms |
| 流式 STT | Whisper-streaming / Parakeet-TDT / Deepgram Nova-3 |
| LLM + 工具调用 | GPT-4o / Claude 3.5 / Gemini 2.5 Flash。JSON Schema 定义工具。流式 token |
| 流式 TTS | Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。LLM 输出 20 token 后即启动 TTS |
| 播放 | 扬声器输出；opus 编码用于低带宽网络 |
| 打断处理 | VAD 在 TTS 播放期间触发 → 停止播放 → 取消 LLM → 重启 STT |

### 2.2 三个会踩到的失败模式

1. **首词截断。** VAD 开始晚了一拍。用户的"嗨"被截掉。将起始阈值设为 0.3 而非 0.5
2. **中途打断混乱。** LLM 在用户打断后继续生成；助手和用户同时说话。将 VAD → 取消 LLM 直连
3. **静默幻觉。** Whisper 在静默预热帧上输出"谢谢观看"。始终 VAD 门控

### 2.3 2026 生产级技术栈

| 栈 | 延迟 | 许可 | 备注 |
|---|---|---|---|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500ms | 商业 API | 2026 行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800ms | 大部分开源 | DIY 友好 |
| Moshi（全双工） | 200-300ms | CC-BY 4.0 | 单模型；架构不同（阶段 15） |

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| 全双工 | 人类双方可以同时说话——不是轮流制 |
| Barge-in | 用户在助手说话时开口 → 必须在 100ms 内停止 TTS、取消 LLM |
| 首词截断 | VAD 开始太晚导致用户第一个词被截掉 |
| 静默幻觉 | 对静默音频运行 Whisper 会产生虚构文本 |

---

## ✏️ 练习

1. 【实现】用 `sounddevice` 录制 30 秒音频，加上 VAD（`silero-vad`），输出语音起止时间戳。
2. 【实验】用 Whisper-streaming + 简单 LLM 实现一个问答助手。测量端到端延迟。
3. 【思考】如果你要在微信小程序上部署语音助手，哪些组件需要替换？延迟预算会如何变化？

---

## 📖 参考资料

1. [项目] Pipecat. https://github.com/foundation-models/foundation-model-frameworks — 开源实时语音 AI 框架
2. [项目] LiveKit. https://livekit.io — 实时音视频基础设施
3. [论文] Défossez et al. "Moshi". Kyutai, 2024. — 全双工语音模型

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
