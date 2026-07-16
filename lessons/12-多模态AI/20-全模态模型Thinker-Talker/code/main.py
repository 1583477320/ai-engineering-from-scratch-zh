# Thinker-Talker 延迟预算 + VAD 模拟


class ThinkerTalkerPipeline:
    """简化版 Thinker-Talker 管道。"""
    def __init__(self, asr_latency=80, thinker_latency=120, talker_latency=80):
        self.asr_latency = asr_latency
        self.thinker_latency = thinker_latency
        self.talker_latency = talker_latency
        self.buffer_latency = 50

    def total_latency(self):
        return self.asr_latency + self.thinker_latency + self.talker_latency + self.buffer_latency

    def analyze_budget(self):
        total = self.total_latency()
        budget = 250
        return {
            "ASR": f"{self.asr_latency}ms",
            "Thinker": f"{self.thinker_latency}ms",
            "Talker": f"{self.talker_latency}ms",
            "缓冲": f"{self.buffer_latency}ms",
            "总延迟": f"{total}ms",
            "预算": f"{budget}ms",
            "状态": "✓ 可行" if total <= budget else "✗ 超预算",
        }


def vad_detect(audio_level, threshold=0.01, silence_frames=10):
    """简化 VAD——检测语音活动。"""
    return audio_level > threshold


if __name__ == "__main__":
    print("Thinker-Talker 延迟预算分析\n")
    pipeline = ThinkerTalkerPipeline(asr_latency=80, thinker_latency=120, talker_latency=80)
    budget = pipeline.analyze_budget()
    for k, v in budget.items():
        print(f"  {k}: {v}")

    print("\nVAD 模拟:")
    levels = [0.001, 0.05, 0.1, 0.3, 0.005]
    for i, level in enumerate(levels):
        speaking = vad_detect(level)
        print(f"  帧 {i+1}: 音量={level:.3f} -> {'说话中' if speaking else '静默'}")
