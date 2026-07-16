# 语音智能体：Pipecat 和 LiveKit

> 语音智能体是 2026 年的一等生产类别。Pipecat 提供基于帧的 Python 管道（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产延迟目标在 450-600 毫秒端到端。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 Pipecat 的帧管道：DOWNSTREAM（源→汇）和 UPSTREAM（控制）
- [ ] 理解 VAD（语音活动检测）在实时语音对话中的作用
- [ ] 对比 Pipecat 和 LiveKit 的架构差异
- [ ] 设计一个低延迟的语音智能体管道

---

## 1. 问题

2026 年的语音智能体需要实时处理——你说话，模型听，然后回答，延迟 < 500 毫秒。这需要精心设计的管道：VAD 检测你在说话→STT 转录→LLM 推理→TTS 合成→传输返回。

---

## 2. 概念

### 2.1 Pipecat 帧管道

```
DOWNSTREAM（数据流）: 音频帧 → [VAD] → [STT] → [LLM] → [TTS] → 音频帧
UPSTREAM（控制流）: 开始说话 → 停止说话 → 推理触发 → 语音合成 → 播放
```

### 2.2 延迟预算

| 阶段 | 延迟 | 优化手段 |
|------|------|---------|
| VAD | ~50ms | 预测模型 |
| STT | ~100ms | 流式 Whisper |
| LLM | ~200ms | 小模型/推测解码 |
| TTS | ~100ms | 流式 TTS |
| 传输 | ~50ms | WebRTC |
| **总计** | **~450ms** | 250ms 是高端目标 |

### 2.3 Pipecat vs LiveKit

| 方面 | Pipecat | LiveKit |
|------|---------|--------|
| 语言 | Python | 多语言 |
| 管道 | 基于帧 | WebRTC 集成 |
| 部署 | 自托管 | 云端/自托管 |
| 特点 | 简单灵活 | 实时通信基础设施 |

---

## 3. 从零实现

### Step 1：简化版语音管道

```python
class VoicePipeline:
    """简化版语音管道。"""
    def __init__(self, vad_fn, stt_fn, llm_fn, tts_fn):
        self.vad = vad_fn
        self.stt = stt_fn
        self.llm = llm_fn
        self.tts = tts_fn
        self.buffer = []

    def process_frame(self, audio_frame):
        """处理音频帧。"""
        self.buffer.append(audio_frame)

        # VAD 检测
        if self.vad.detect_speech(self.buffer):
            # STT 转录
            text = self.stt.transcribe(self.buffer)
            self.buffer = []

            # LLM 推理
            response = self.llm.generate(text)

            # TTS 合成
            audio_response = self.tts.synthesize(response)
            return audio_response

        return None
```

---

## 4. 工具

### 4.1 Pipecat

```python
# Pipecat 框架
from pipecat.pipeline import Pipeline
from pipecat.processors import VADProcessor, STTProcessor, LLMProcessor, TTSProcessor

pipeline = Pipeline([
    VADProcessor(),
    STTProcessor(model="whisper"),
    LLMProcessor(model="gpt-4o-mini"),
    TTSProcessor(model="tts-1"),
])
```

### 4.2 LiveKit

```python
# LiveKit Agents
from livekit.agents import AgentSession

session = AgentSession(voice="alloy")
await session.connect()
```

---

## 5. 工程最佳实践

### 5.1 低延迟设计

- **流式处理**：每个组件逐帧处理——不等待完整输入
- **并行化**：STT 和 LLM 可以重叠执行
- **模型选择**：推理用小模型（如 gpt-4o-mini），质量用大模型

### 5.2 踩坑经验

- **VAD 误触发**：背景噪声导致误触发——调整阈值
- **STT 延迟**：长停顿后重新检测——设置合理缓冲

---

## 6. 常见错误

### 错误 1：不处理 VAD 误触发

**现象：** 背景噪声导致智能体在用户没说话时就开始回复。

**修复：** 设置最小语音时长——只有连续语音 >500ms 才触发 STT。

---

## 7. 面试考点

### Q1：语音智能体的延迟预算如何分配？（难度：⭐⭐）

**参考答案：**
总延迟 <500ms：VAD ~50ms + STT ~100ms + LLM ~200ms + TTS ~100ms + 传输 ~50ms。关键优化：VAD 用预测模型，STT 用流式 Whisper，LLM 用小模型+推测解码，TTS 用流式合成。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Pipecat | "语音管道框架" | 基于帧的 Python 语音处理管道——VAD→STT→LLM→TTS |
| VAD | "语音检测" | 语音活动检测——判断用户是否在说话 |
| 流式 TTS | "边生成边播放" | 语音合成实时输出——不等待完整生成 |

---

## 📚 小结

语音智能体 = VAD + STT + LLM + TTS。Pipecat 提供帧管道，LiveKit 提供实时通信。生产延迟目标 450-600ms。VAD 是实时对话的关键——误触发会导致体验问题。

---

## ✏️ 练习

1. **【分析】** 计算一个语音智能体的完整延迟预算——每个阶段多少毫秒
2. **【设计】** 设计一个中文语音客服系统——考虑 VAD、STT、TTS 的中文支持

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 语音管道 | `code/main.py` | VAD + STT + LLM + TTS 帧管道 |

---

## 📖 参考资料

1. [GitHub] Pipecat: https://github.com/pipe-cat/pipecat
2. [GitHub] LiveKit: https://github.com/livekit/livekit
3. [文档] LiveKit Agents
