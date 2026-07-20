"""实时语音管道——VAD + 说话人检测 + 打断调度器。

2026年语音智能体的核心架构原语不是ASR或TTS，
而是流式调度器，它在有界延迟下仲裁VAD事件、ASR部分转录、
说话人完成评分、LLM流式、TTS流式和用户打断。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
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
    # 尾部静默，2200ms
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

        if state == State.WAITING:
            llm_stream_started_at = f.t_ms + 140
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
# 演示
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
