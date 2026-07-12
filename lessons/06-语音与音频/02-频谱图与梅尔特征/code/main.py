# 频谱图、梅尔滤波器组、MFCC——纯标准库实现
# 依赖：无（纯 stdlib：math）
# 对应课程：阶段 06 · 02

import math


def sine(freq_hz, sr, seconds, amp=0.5, phase=0.0):
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr + phase) for i in range(n)]


def chirp(f0, f1, sr, seconds, amp=0.5):
    """频率扫描信号——从 f0 线性扫到 f1。"""
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * (f0 + (f1 - f0) * i / n) * i / sr)
            for i in range(n)]


def hann(N):
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * n / (N - 1))) for n in range(N)]


def dft_mag(x):
    """DFT 幅度——O(N²)，仅教学用。"""
    n = len(x)
    half = n // 2 + 1
    return [math.sqrt(sum(
        x[j] * math.cos(-2.0 * math.pi * k * j / n) for j in range(n)) ** 2 +
        sum(
        x[j] * math.sin(-2.0 * math.pi * k * j / n) for j in range(n)) ** 2)
        for k in range(half)]


def frame_signal(signal, frame_len, hop):
    if len(signal) < frame_len: return []
    return [signal[i * hop : i * hop + frame_len]
            for i in range(1 + (len(signal) - frame_len) // hop)]


def stft_magnitude(signal, frame_len, hop):
    w = hann(frame_len)
    frames = frame_signal(signal, frame_len, hop)
    return [dft_mag([w[j] * f[j] for j in range(frame_len)]) for f in frames]


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
    """三角梅尔滤波器组——梅尔尺度在 1kHz 以下近似线性，以上近似对数。"""
    if fmax is None:
        fmax = sr / 2
    m_lo, m_hi = hz_to_mel(fmin), hz_to_mel(fmax)
    mels = [m_lo + (m_hi - m_lo) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    half = n_fft // 2 + 1
    bins = [min(half - 1, int(round(h * n_fft / sr))) for h in hzs]
    fb = [[0.0] * half for _ in range(n_mels)]
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            fb[m][k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[m][k] = (right - k) / max(1, right - center)
    return fb


def apply_filterbank(stft_mag, fb):
    """STFT 幅度 × 梅尔滤波器组 = 梅尔频谱图。"""
    return [[sum(s * w for s, w in zip(spec, filt)) for filt in fb]
            for spec in stft_mag]


def log_transform(mel_spec, eps=1e-10):
    """对数梅尔频谱——Whisper/Parakeet 的标准输入。"""
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]


def dct_ii(x, n_coeffs):
    """离散余弦变换（类型 II）——MFCC 的核心。"""
    N = len(x)
    return [sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
                for n in range(N))
            for k in range(n_coeffs)]


def main():
    sr, frame_len, hop, n_mels, n_fft = 8000, 256, 128, 40, 256

    # Step 1: 分帧
    tone = sine(2000.0, sr, 0.5)
    frames = frame_signal(tone, frame_len, hop)
    print(f"=== Step 1: 分帧 ===")
    print(f"  采样点: {len(tone)}, 帧数: {len(frames)}, 帧长: {frame_len}, 帧移: {hop}")

    # Step 2: Hann 窗
    w = hann(frame_len)
    print(f"\n=== Step 2: Hann 窗 ===")
    print(f"  h(0)={w[0]:.4f}  h(mid)={w[frame_len // 2]:.4f}  h(last)={w[-1]:.4f}")

    # Step 3: STFT 幅度
    mag = stft_magnitude(tone, frame_len, hop)
    mid = mag[len(mag) // 2]
    k_peak = max(range(len(mid)), key=lambda i: mid[i])
    print(f"\n=== Step 3: STFT 幅度 ===")
    print(f"  帧数: {len(mag)}, 每帧 bin 数: {len(mid)}")
    print(f"  峰值 bin={k_peak}, 频率={k_peak * sr / n_fft:.1f} Hz (期望 2000 Hz)")

    # Step 4: 梅尔滤波器组
    fb = mel_filterbank(n_mels, n_fft, sr)
    widths = [sum(1 for x in f if x > 0) for f in fb]
    print(f"\n=== Step 4: 梅尔滤波器组 ({n_mels} 个梅尔) ===")
    print(f"  滤波器组形状: {n_mels} × {len(fb[0])}")
    print(f"  低频滤波器窄（密集），高频滤波器宽（稀疏）— 模拟人耳听觉")
    print(f"  宽度（前6）: {widths[:6]}  （后6）: {widths[-6:]}")

    # Step 5: chirp + 梅尔频谱
    c = chirp(200.0, 4000.0, sr, 0.4)
    cmag = stft_magnitude(c, frame_len, hop)
    mel_spec = apply_filterbank(cmag, fb)
    lm = log_transform(mel_spec)
    print(f"\n=== Step 5: chirp 200→4000 Hz, argmax 梅尔 ===")
    step = max(1, len(lm) // 8)
    for i in range(0, len(lm), step):
        am = max(range(n_mels), key=lambda m: lm[i][m])
        print(f"  t={i:3d}  argmax_mel={am:2d}")

    # Step 6: MFCC
    mfcc = dct_ii(lm[len(lm) // 2], 13)
    print(f"\n=== Step 6: MFCC-13 (中帧) ===")
    print(f"  {[round(c, 3) for c in mfcc]}")
    print("  注意: 第 0 个系数编码整体能量，通常在下游丢弃")


if __name__ == "__main__":
    main()
