# 音频基础——合成、DFT、峰值检测、混叠演示
# 依赖：无（纯标准库：math, wave, struct, tempfile）
# 对应课程：阶段 06 · 01（音频基础）

import math
import os
import struct
import tempfile
import wave
from typing import List, Tuple


def sine(freq_hz: float, sr: int, seconds: float, amp: float = 0.5) -> List[float]:
    """合成正弦波。"""
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr) for i in range(n)]


def mix(*signals: List[float]) -> List[float]:
    """将多个信号等权混合。"""
    length = min(len(s) for s in signals)
    return [sum(s[i] for s in signals) / len(signals) for i in range(length)]


def write_wav(path: str, samples: List[float], sr: int) -> None:
    """写 16-bit PCM WAV 文件。"""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        )
        w.writeframes(frames)


def read_wav(path: str) -> Tuple[List[float], int]:
    """读 WAV 文件——返回 (采样点列表, 采样率)。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    ints = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    return [x / 32768.0 for x in ints], sr


def dft(x: List[float]) -> List[Tuple[float, float]]:
    """离散傅里叶变换——O(N^2)，仅用于教学验证。
    生产中使用 numpy.fft.rfft 或 torch.fft.rfft。
    """
    n = len(x)
    out = []
    for k in range(n):
        re = sum(x[j] * math.cos(-2.0 * math.pi * k * j / n) for j in range(n))
        im = sum(x[j] * math.sin(-2.0 * math.pi * k * j / n) for j in range(n))
        out.append((re, im))
    return out


def magnitudes(spectrum: List[Tuple[float, float]]) -> List[float]:
    return [math.sqrt(re * re + im * im) for re, im in spectrum]


def peak_freq(samples: List[float], sr: int) -> Tuple[float, int]:
    """找主导频率——DFT 幅度谱的峰值索引→频率（Hz）。"""
    mags = magnitudes(dft(samples))
    half = len(mags) // 2
    mags = mags[:half]  # 只看正频率
    k = max(range(len(mags)), key=lambda i: mags[i])
    return k * sr / len(samples), k


def downsample_naive(samples: List[float], factor: int) -> List[float]:
    """朴素下采样——直接抽样（不抗混叠），演示混叠现象。"""
    return samples[::factor]


def main():
    sr = 8000
    duration = 0.064

    # Step 1: 合成正弦波
    print("=== Step 1: 合成 440 Hz 正弦波，8 kHz，64 ms ===")
    a = sine(440.0, sr, duration)
    print(f"  采样点数: {len(a)}")
    print(f"  前5个采样值: {[round(x, 4) for x in a[:5]]}")

    # Step 2: WAV 文件读写往返
    print("\n=== Step 2: WAV 文件往返 ===")
    tmpdir = tempfile.mkdtemp(prefix="audio_01_")
    path = os.path.join(tmpdir, "a440.wav")
    write_wav(path, a, sr)
    loaded, loaded_sr = read_wav(path)
    diff = max(abs(a[i] - loaded[i]) for i in range(len(a)))
    print(f"  文件大小: {os.path.getsize(path)} bytes, 采样率={loaded_sr}")
    print(f"  16-bit 量化误差: {diff:.5f}")

    # Step 3: DFT 峰值检测
    print("\n=== Step 3: DFT 峰值检测 ===")
    freq, k = peak_freq(a, sr)
    resolution = sr / len(a)
    print(f"  峰值 bin={k}, 频率={freq:.1f} Hz")
    print(f"  期望频率 ~440.0 Hz, bin 分辨率 {resolution:.2f} Hz")

    # Step 4: 混合信号的 DFT
    print("\n=== Step 4: 混合信号 (220+440+880 Hz) ===")
    mixed = mix(sine(220, sr, duration), sine(440, sr, duration), sine(880, sr, duration))
    mags = magnitudes(dft(mixed))[:len(mixed) // 2]
    top3 = sorted(range(len(mags)), key=lambda i: -mags[i])[:3]
    peaks_hz = sorted(round(k * sr / len(mixed), 1) for k in top3)
    print(f"  前3个峰值: {peaks_hz} Hz")

    # Step 5: 混叠演示
    print("\n=== Step 5: 混叠演示——7 kHz 音调用 10 kHz 采样 ===")
    alias_sr = 10000
    tone = sine(7000.0, alias_sr, 0.0512)
    alias_freq, _ = peak_freq(tone, alias_sr)
    nyquist = alias_sr / 2
    folded = alias_sr - 7000.0
    print(f"  真实频率: 7000.0 Hz (超过奈奎斯特频率 {nyquist} Hz)")
    print(f"  DFT 报告: {alias_freq:.1f} Hz")
    print(f"  预期混叠: {folded:.1f} Hz (= sr - f_true)")

    # Step 6: 合理下采样 vs 朴素下采样
    print("\n=== Step 6: 合理下采样 vs 朴素下采样 ===")
    orig_sr = 24000
    sig = sine(7000.0, orig_sr, 0.032)
    decimated = downsample_naive(sig, 3)
    new_sr = orig_sr // 3
    peak_new, _ = peak_freq(decimated, new_sr)
    print(f"  24 kHz → 8 kHz (无抗混叠): 峰值={peak_new:.1f} Hz")
    print(f"  预期混叠: 1000 Hz (= sr_new - f_true)")
    print(f"  教训: 下采样前必须先低通滤波（抗混叠）！")


if __name__ == "__main__":
    main()
