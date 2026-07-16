# 语音管道模拟


class VoicePipeline:
    def __init__(self, vad_fn, stt_fn, llm_fn, tts_fn):
        self.vad = vad_fn
        self.stt = stt_fn
        self.llm = llm_fn
        self.tts = tts_fn
        self.buffer = []

    def process_frame(self, audio_frame):
        self.buffer.append(audio_frame)
        if len(self.buffer) > 20:
            text = self.stt.transcribe(self.buffer)
            self.buffer = []
            response = self.llm.generate(text)
            return self.tts.synthesize(response)
        return None


if __name__ == "__main__":
    print("语音管道演示\n")
    pipe = VoicePipeline(
        vad_fn=lambda buf: len(buf) > 20,
        stt_fn=lambda buf: "用户说了你好",
        llm_fn=lambda text: f"回答: {text}",
        tts_fn=lambda resp: f"[TTS] {resp}",
    )
    for i in range(25):
        result = pipe.process_frame(f"音频帧_{i}")
        if result:
            print(f"  输出: {result}")
            break
