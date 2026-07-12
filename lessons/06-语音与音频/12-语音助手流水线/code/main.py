# 语音助手流水线：七组件架构演示
# 对应课程：阶段 06 · 12

from typing import List, Tuple, Optional


class AudioCapture:
    """组件1：音频捕获——麦克风→16kHz单声道→20ms块"""
    def __init__(self, sr=16000, chunk_ms=20):
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.samples_per_chunk = int(sr * chunk_ms / 1000)
        print(f"  [音频捕获] sr={sr}Hz, chunk={chunk_ms}ms, {self.samples_per_chunk} 采样点/块")


class VAD:
    """组件2：语音活动检测——判断当前是否有语音"""
    def __init__(self, threshold=0.5, min_speech_ms=250, silence_hangover_ms=500):
        self.threshold = threshold
        self.min_speech = min_speech_ms
        self.hangover = silence_hangover_ms
        print(f"  [VAD] 阈值={threshold}, 最小语音={min_speech_ms}ms, 静默挂起={silence_hangover_ms}ms")

    def detect(self, energy: float) -> bool:
        return energy > self.threshold


class StreamingSTT:
    """组件3：流式语音转文字"""
    def __init__(self, model="whisper-streaming", latency_ms=150):
        self.model = model
        self.latency_ms = latency_ms
        print(f"  [STT] 模型={model}, 延迟={latency_ms}ms")

    def transcribe(self, audio_chunk: List[float]) -> str:
        return "[转录结果]"


class LLMWithTools:
    """组件4：LLM + 工具调用"""
    def __init__(self, model="gpt-4o", tools=None):
        self.model = model
        self.tools = tools or []
        print(f"  [LLM] 模型={model}, 工具数={len(self.tools)}")

    def generate(self, transcript: str) -> Tuple[str, Optional[str]]:
        return "收到你的请求。", None  # (文本, 工具调用)


class StreamingTTS:
    """组件5：流式文本转语音"""
    def __init__(self, model="kokoro-82m", latency_ms=100):
        self.model = model
        self.latency_ms = latency_ms
        print(f"  [TTS] 模型={model}, 首chunk延迟={latency_ms}ms")

    def synthesize(self, text: str) -> List[float]:
        return [0.0] * 1000  # 模拟音频


class AudioPlayback:
    """组件6：音频播放"""
    def play(self, audio: List[float]):
        pass  # 实际使用 sounddevice/PortAudio


class InterruptionHandler:
    """组件7：打断处理"""
    def __init__(self, vad, tts, llm):
        self.vad = vad
        self.tts = tts
        self.llm = llm
        print("  [打断处理] VAD→停止TTS→取消LLM→重启STT")

    def check_barge_in(self, mic_energy: float) -> bool:
        return self.vad.detect(mic_energy)


def main():
    print("=== 语音助手流水线：七组件架构 ===\n")

    print("初始化组件:")
    capture = AudioCapture()
    vad = VAD()
    stt = StreamingSTT()
    llm = LLMWithTools(tools=["timer", "weather", "calendar"])
    tts = StreamingTTS()
    playback = AudioPlayback()
    interruption = InterruptionHandler(vad, tts, llm)

    print(f"\n=== 延迟预算 ===")
    budgets = [("麦克风→缓冲区", 20), ("VAD", 10), ("ASR", 150),
               ("LLM首token", 100), ("TTS首chunk", 100), ("渲染→扬声器", 20)]
    total = 0
    for name, ms in budgets:
        bar = "█" * (ms // 10)
        print(f"  {name:<18} {ms:>4}ms  {bar}")
        total += ms
    print(f"  {'总计':<18} {total:>4}ms {'✅ < 800ms' if total < 800 else '❌ 超时'}")

    print(f"\n=== 三大常见陷阱 ===")
    print("  1. 首词截断: VAD 开始太晚 → 用户'嗨'被截掉。阈值设为 0.3 而非 0.5")
    print("  2. 中途打断混乱: LLM 继续生成 → 助手和用户同时说话。VAD 直连取消 LLM")
    print("  3. 静默幻觉: Whisper 在静默帧输出'谢谢观看'。始终 VAD 门控")

    print(f"\n=== 2026 生产级技术栈 ===")
    stacks = [
        ("LiveKit+Deepgram+GPT-4o+Cartesia", "350-500ms", "商业 API"),
        ("Pipecat+Whisper+GPT-4o+Kokoro", "500-800ms", "大部分开源"),
        ("Moshi（全双工）", "200-300ms", "CC-BY 4.0"),
    ]
    for name, lat, lic in stacks:
        print(f"  {name}: {lat} ({lic})")


if __name__ == "__main__":
    main()
