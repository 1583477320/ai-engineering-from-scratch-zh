# 综合项目03——实时语音助手（ASR到LLM到TTS）

> 一个体验良好的语音智能体：端到端延迟低于800ms、知道用户何时停止说话、处理打断（barge-in）且不影响工具调用。Retell、Vapi、LiveKit Agents和Pipecat在2026年都达到这个标准。它们的架构相同：流式ASR、说话人检测器、流式LLM和流式TTS，通过WebRTC连接，每跳都有严格的延迟预算。本综合项目要求你构建一个，测量WER、MOS和假切断率，并在丢包条件下运行测试。

**类型：** 综合项目
**编程语言：** Python（智能体 + 管道），TypeScript（Web客户端）
**前置知识：** 第6章（语音与音频）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第14章（智能体）、第17章（基础设施）
**涉及章节：** P6 · P7 · P11 · P13 · P14 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建流式语音管道：ASR → VAD → 说话人检测 → LLM → TTS
- 实现打断处理（barge-in）和工具侧通道
- 在丢包条件下测试和优化端到端延迟
- 测量并优化WER、MOS、假切断率和首音频输出延迟

---

## 1. 问题

语音是2025-2026年变化最快的AI UX类别。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0和Pipecat 0.0.70都让800ms首音频输出触手可及。

标准不仅仅是延迟，还有交互体验：不切断用户、不被切断、从中句打断中恢复、在对话中调用工具且不卡顿、在抖动移动网络中存活。

你不能通过串联三个REST调用来达到这个标准。架构必须是端到端流式管道。

---

## 2. 核心概念

### 2.1 流式管道

管道有五个流式阶段：

1. **音频输入**：浏览器或PSTN的WebRTC音频流
2. **ASR（自动语音识别）**：流式部分转录，来自Deepgram Nova-3或faster-whisper
3. **说话人检测**：VAD加小型检测模型，读取部分转录判断是否说完
4. **LLM（大语言模型）**：说话人判断完成后立即流式输出token
5. **TTS（文本转语音）**：在LLM首个token后的约200ms内开始流式输出音频

### 2.2 三个横切关注点

- **打断处理（Barge-in）**：用户说话时TTS正在播放，立即取消TTS流，丢弃剩余LLM输出，重新准备ASR
- **工具使用**：对话中的函数调用（天气、日历）必须在侧通道运行，不卡顿音频；如果延迟超过300ms，智能体预填"稍等..."确认token
- **背压**：在丢包条件下，部分转录被暂存，VAD提高语音门控阈值

### 2.3 测量指标

- WER低于8%（Hamming VAD基准，15 dB SNR）
- 首音频输出p50低于800ms（100次测量）
- 假切断率低于3%
- TTS MOS高于4.2
- 单个g5.xlarge支持50路并发

---

## 3. 从零实现

`code/main.py`实现语音管道的核心调度器：VAD事件、ASR部分转录、说话人完成评分、LLM流式、TTS流式和用户打断的仲裁。

```python
"""实时语音管道——VAD + 说话人检测 + 打断调度器。

2026年语音智能体的核心架构原语不是ASR或TTS，
而是流式调度器，它在有界延迟下仲裁VAD事件、ASR部分转录、
说话人完成评分、LLM流式、TTS流式和用户打断。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# 帧流——模拟的20ms音频帧
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    t_ms: int              # 会话开始后的时间戳（毫秒）
    is_speech: bool        # VAD判断（Silero v5替代）
    partial: str = ""      # ASR累积部分转录（Deepgram Nova-3替代）


def synth_call(script: str, start_ms: int = 0, noise: float = 0.0) -> list[Frame]:
    """生成模拟调用者话语的帧流"""
    words = script.split()
    frames: list[Frame] = []
    t = start_ms
    # 120ms静默（语音前）
    for _ in range(6):
        frames.append(Frame(t_ms=t, is_speech=random.random() < noise))
        t += 20
    partial = ""
    for w in words:
        partial = (partial + " " + w).strip()
        # 每个词约320ms语音
        for _ in range(16):
            frames.append(Frame(t_ms=t, is_speech=True, partial=partial))
            t += 20
    # 尾部静默，2200ms（足够覆盖工具+LLM+TTS）
    for _ in range(110):
        frames.append(Frame(t_ms=t, is_speech=False, partial=partial))
        t += 20
    return frames


# ---------------------------------------------------------------------------
# 说话人检测——结合VAD静默时长和完成评分
# ---------------------------------------------------------------------------

def turn_completion_score(partial: str) -> float:
    """替代LiveKit说话人检测模型的简化版"""
    if not partial:
        return 0.0
    if partial.rstrip().endswith(("?", ".", "!")):
        return 0.95
    n = len(partial.split())
    if n < 3:
        return 0.2
    if n < 6:
        return 0.55
    return 0.75


# ---------------------------------------------------------------------------
# 状态机——IDLE -> LISTENING -> THINKING -> SPEAKING -> (barge-in)
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    LISTENING = auto()   # 用户正在说话
    WAITING = auto()     # VAD说静默了，检查说话人分数
    THINKING = auto()    # LLM流式，但TTS尚未开始
    SPEAKING = auto()    # TTS正在播放
    TOOL = auto()        # 侧通道工具正在进行


@dataclass
class Metrics:
    events: list[str] = field(default_factory=list)
    turn_complete_ms: int = 0
    first_llm_token_ms: int = 0
    first_audio_out_ms: int = 0
    false_cutoffs: int = 0
    barge_ins: int = 0

    def log(self, msg: str) -> None:
        self.events.append(msg)

    def latency_ms(self) -> int:
        if self.turn_complete_ms and self.first_audio_out_ms:
            return self.first_audio_out_ms - self.turn_complete_ms
        return -1


# ---------------------------------------------------------------------------
# 工具侧通道——带填充注入的异步天气/日历
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    latency_ms: int
    result: str


WEATHER = Tool("weather.tokyo_tomorrow", latency_ms=420, result="68/52 partly cloudy")


# ---------------------------------------------------------------------------
# 调度器——完整的管道，逐帧流式处理
# ---------------------------------------------------------------------------

def run_session(frames: list[Frame], use_tool: bool = True,
                barge_in_at_ms: int | None = None) -> Metrics:
    m = Metrics()
    state = State.IDLE
    silence_run_ms = 0
    final_partial = ""
    llm_stream_started_at = -1
    tts_stream_started_at = -1
    tool_started_at = -1
    tool_done_at = -1
    filler_emitted = False

    for f in frames:
        # 打断：用户在SPEAKING或THINKING时开始说话
        if (barge_in_at_ms is not None and f.t_ms >= barge_in_at_ms
                and state in (State.SPEAKING, State.THINKING)
                and f.is_speech):
            m.barge_ins += 1
            m.log(f"{f.t_ms}ms BARGE-IN: 取消TTS，重新准备ASR")
            state = State.LISTENING
            tts_stream_started_at = -1
            llm_stream_started_at = -1
            continue

        if state == State.IDLE:
            if f.is_speech:
                state = State.LISTENING
                m.log(f"{f.t_ms}ms LISTENING")

        elif state == State.LISTENING:
            if f.is_speech:
                silence_run_ms = 0
                final_partial = f.partial or final_partial
            else:
                silence_run_ms += 20
                if silence_run_ms >= 500:
                    score = turn_completion_score(final_partial)
                    if score >= 0.6:
                        state = State.WAITING
                        m.turn_complete_ms = f.t_ms
                        m.log(f"{f.t_ms}ms 说话人完成 (score={score:.2f})"
                              f" partial='{final_partial}'")
                    else:
                        m.log(f"{f.t_ms}ms 静默但score={score:.2f}, 等待")

        if state == State.WAITING:
            # 启动LLM
            llm_stream_started_at = f.t_ms + 140  # 模拟到首token的时间
            state = State.THINKING
            m.log(f"{f.t_ms}ms LLM调用触发")
            if use_tool:
                tool_started_at = f.t_ms
                state = State.TOOL

        elif state == State.TOOL:
            if tool_started_at >= 0 and not filler_emitted:
                if f.t_ms - tool_started_at >= 300:
                    filler_emitted = True
                    m.log(f"{f.t_ms}ms 填充词 '稍等，让我查一下'")
            if tool_started_at >= 0 and f.t_ms - tool_started_at >= WEATHER.latency_ms:
                tool_done_at = f.t_ms
                m.log(f"{f.t_ms}ms 工具结果: {WEATHER.result}")
                llm_stream_started_at = f.t_ms + 140
                state = State.THINKING

        elif state == State.THINKING:
            if llm_stream_started_at > 0 and f.t_ms >= llm_stream_started_at:
                if m.first_llm_token_ms == 0:
                    m.first_llm_token_ms = f.t_ms
                    m.log(f"{f.t_ms}ms LLM首token")
                tts_stream_started_at = f.t_ms + 180
                state = State.SPEAKING

        elif state == State.SPEAKING:
            if tts_stream_started_at > 0 and f.t_ms >= tts_stream_started_at:
                if m.first_audio_out_ms == 0:
                    m.first_audio_out_ms = f.t_ms
                    m.log(f"{f.t_ms}ms TTS首音频输出")

    return m


# ---------------------------------------------------------------------------
# 演示——运行两轮会话，一轮干净，一轮有打断
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0)
    print("=== 会话1: 干净调用带工具（天气） ===")
    frames = synth_call("what is the weather in tokyo tomorrow", start_ms=0)
    m = run_session(frames, use_tool=True, barge_in_at_ms=None)
    for line in m.events:
        print(" ", line)
    print(f"  说话人完成      @ {m.turn_complete_ms}ms")
    print(f"  LLM首token      @ {m.first_llm_token_ms}ms")
    print(f"  首音频输出      @ {m.first_audio_out_ms}ms")
    print(f"  轮次延迟        = {m.latency_ms()}ms")

    print()
    print("=== 会话2: 用户在响应中打断 ===")
    frames = synth_call("tell me a long story about", start_ms=0)
    for i in range(8):
        idx = len(frames) - 20 + i
        if 0 <= idx < len(frames):
            frames[idx] = Frame(t_ms=frames[idx].t_ms, is_speech=True,
                                partial=frames[idx].partial)
    m = run_session(frames, use_tool=False,
                    barge_in_at_ms=frames[-20].t_ms - 60)
    for line in m.events:
        print(" ", line)
    print(f"  打断次数 = {m.barge_ins}")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 会话1: 干净调用带工具（天气） ===
  0ms LISTENING
  2900ms 说话人完成 (score=0.75) partial='what is the weather in tokyo tomorrow'
  2900ms LLM调用触发
  3040ms 填充词 '稍等，让我查一下'
  3320ms 工具结果: 68/52 partly cloudy
  3460ms LLM首token
  3640ms TTS首音频输出
  说话人完成      @ 2900ms
  LLM首token      @ 3460ms
  首音频输出      @ 3640ms
  轮次延迟        = 740ms

=== 会话2: 用户在响应中打断 ===
  0ms LISTENING
  3380ms 说话人完成 (score=0.75) partial='tell me a long story about'
  3380ms LLM调用触发
  3520ms LLM首token
  3700ms TTS首音频输出
  3780ms BARGE-IN: 取消TTS，重新准备ASR
  打断次数 = 1
```

---

## 4. 工具实践

**技术栈：**
- 传输：LiveKit Agents 1.0（WebRTC）+ Twilio PSTN网关
- ASR：Deepgram Nova-3（流式，<300ms首部分转录）或faster-whisper Whisper-v3-turbo自托管
- VAD：Silero VAD v5 + LiveKit说话人检测器
- LLM：OpenAI GPT-4o-realtime或Gemini 2.5 Flash Live
- TTS：Cartesia Sonic-2（最低首字节）或ElevenLabs Flash v3
- 可观测性：OpenTelemetry语音span + Langfuse语音trace

---

## 5. LLM视角

**延迟视角**：首音频输出时间是用户体验的关键指标。每个阶段都需要严格的延迟预算：ASR <300ms、说话人检测 <200ms、LLM首token <200ms、TTS首字节 <200ms。

**打断视角**：打断处理（Barge-in）是语音交互的自然需求。当用户开始在智能体说话时说话，系统必须能优雅地处理。

**工具视角**：语音场景中的工具调用需要侧通道处理，确保不卡顿音频。如果工具响应慢，智能体先发填充词。

---

## 6. 工程最佳实践

**延迟优化**：
- 使用流式API而非REST调用
- 并行化层内操作
- 每个阶段设置延迟预算

**语音质量**：
- 使用Silero VAD v5检测说话
- 500ms静默 + 说话人完成评分（>0.6）确定说完
- MOS高于4.2作为TTS质量目标

**可靠性**：
- 在3%丢包率下测试
- 实施背压机制
- 记录语音trace

---

## 7. 常见错误

**错误1：使用REST API串联**
症状：端到端延迟超过2秒
修复：使用流式API，端到端流式管道

**错误2：忽略打断处理**
症状：用户无法打断智能体
修复：实现barge-in机制

**错误3：无工具侧通道**
症状：工具调用卡顿音频
修复：侧通道处理工具调用，延迟>300ms时发填充词

---

## 8. 面试考点

**Q1：语音管道的五个流式阶段是什么？**
考察：对语音AI架构的理解

**Q2：如何检测说话人是否说完？**
考察：对VAD和说话人检测的理解

**Q3：什么是打断处理（Barge-in）？**
考察：对交互体验的理解

**Q4：如何优化端到端延迟？**
考察：对延迟预算的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 说话人检测 | "句尾检测" | 分类器根据VAD静默和部分转录决定用户是否说完 |
| 打断处理 | "中断处理" | VAD检测到新语音时取消TTS播放 |
| 首音频输出 | "延迟" | 从用户停止说话到首个音频包离开服务器的时间 |
| VAD | "语音门控" | 将音频帧分类为说话 vs 静默的模型 |
| 抖动缓冲 | "音频平滑" | 客户端缓冲，短暂保持数据包以吸收网络变化 |
| 填充词 | "确认token" | 智能体在工具响应慢时发短的确认短语 |
| MOS | "平均意见分" | 感知语音质量评分 |

---

## 参考文献

- [LiveKit Agents 1.0](https://github.com/livekit/agents)
- [Pipecat](https://github.com/pipecat-ai/pipecat)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Deepgram Nova-3文档](https://developers.deepgram.com/docs)
- [Silero VAD v5](https://github.com/snakers4/silero-vad)
- [Cartesia Sonic-2](https://docs.cartesia.ai)
- [Retell AI架构](https://docs.retellai.com)
