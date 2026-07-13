# Whisper 使用指南

> OpenAI Whisper 的安装、使用和优化指南。

## 快速使用

```python
import whisper

# 加载模型
model = whisper.load_model("base")  # tiny/base/small/medium/large-v3

# 转录音频
result = model.transcribe("audio.mp3")
print(result["text"])

# 翻译（中文→英文）
result = model.transcribe("chinese_audio.mp3", task="translate")

# 带时间戳
result = model.transcribe("audio.mp3", word_timestamps=True)
```

## 模型选择

| 模型 | 速度 | 内存 | WER | 推荐场景 |
|------|------|------|-----|----------|
| tiny | ~10x | 1GB | 7.6% | 实时推理 |
| base | ~7x | 1GB | 4.9% | 轻量应用 |
| small | ~4x | 2GB | 3.5% | 通用场景 |
| large-v3-turbo | ~6x | 2GB | 1.6% | 高质量 |

## 常见优化

- **VAD 门控**：使用 silero-vad 检测语音段，避免静默产生幻觉
- **分块处理**：长音频分块（30s）并行处理
- **模型量化**：使用 FP16 或 int8 量化减少显存
