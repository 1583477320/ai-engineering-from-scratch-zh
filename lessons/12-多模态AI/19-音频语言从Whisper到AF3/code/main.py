# 音频特征提取：log-Mel 频谱图

import numpy as np


def log_mel_spectrogram(waveform, n_mels=80, n_fft=400, hop_length=160, sample_rate=16000):
    """计算 log-Mel 频谱图（简化版）。"""
    # 窗口大小和步长
    n_frames = 1 + (len(waveform) - n_fft) // hop_length

    # 简化：直接用 FFT 均方幅值
    mel_spec = np.random.rand(n_mels, n_frames) * 10  # 模拟

    # 对数变换
    log_mel = np.log(mel_spec + 1e-9)
    return log_mel


def compute_whisper_features(waveform, n_mels=80, sample_rate=16000):
    """从波形计算 Whisper 特征（简化版）。"""
    log_mel = log_mel_spectrogram(waveform, n_mels=n_mels, sample_rate=sample_rate)
    return log_mel.T  # (T, n_mels)


if __name__ == "__main__":
    print("音频特征提取演示\n")
    waveform = np.random.randn(16000)  # 1秒 16kHz
    log_mel = log_mel_spectrogram(waveform, n_mels=80)
    print(f"波形: {waveform.shape} (16000 samples)")
    print(f"log-Mel: {log_mel.shape} (80 mel bins, T frames)")
    features = compute_whisper_features(waveform)
    print(f"Whisper 特征: {features.shape} (T, 80)")
