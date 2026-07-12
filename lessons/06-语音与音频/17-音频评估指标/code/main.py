# 音频评估指标——PESQ、STOI、FAD、UTMOS
# 对应课程：阶段 06 · 17
"""音频质量评估：从原始波形到主观预测分数"""
import math
import random


def pesq_score(original, degraded, sr=16000):
    """PESQ（感知语音质量评估）的简化模拟。
    真实 PESQ 需要 ITU-T P.862 标准实现（pesq 库）。"""
    if len(original) != len(degraded):
        return 0.0
    mse = sum((a - b) ** 2 for a, b in zip(original, degraded)) / len(original)
    pesq_val = max(1.0, min(4.5, 3.5 - 10 * math.log10(max(mse, 1e-10))))
    return round(pesq_val, 2)


def stoi_score(original, degraded, sr=16000):
    """STOI（短时客观可懂度）的简化模拟。"""
    if len(original) != len(degraded):
        return 0.0
    corrs = []
    chunk_size = sr // 50  # 20ms 帧
    for i in range(0, min(len(original), len(degraded)), chunk_size):
        orig_chunk = original[i:i+chunk_size]
        deg_chunk = degraded[i:i+chunk_size]
        if len(orig_chunk) < 2 or len(deg_chunk) < 2:
            continue
        mean_o = sum(orig_chunk) / len(orig_chunk)
        mean_d = sum(deg_chunk) / len(deg_chunk)
        cov = sum((o - mean_o) * (d - mean_d)
                  for o, d in zip(orig_chunk, deg_chunk))
        var_o = sum((o - mean_o) ** 2 for o in orig_chunk)
        var_d = sum((d - mean_d) ** 2 for d in deg_chunk)
        if var_o > 0 and var_d > 0:
            corrs.append(cov / math.sqrt(var_o * var_d))
    return sum(corrs) / len(corrs) if corrs else 0.0


def fad_score(real_features, fake_features, dim=20):
    """FAD（Fréchet 音频距离）的简化模拟。
    真实 FAD 用 PANNs/VGGish 嵌入。"""
    real_mean = sum(f) / len(real_features) if real_features else 0
    fake_mean = sum(f) / len(fake_features) if fake_features else 0
    real_var = sum((f - real_mean)**2 for f in real_features) / max(1, len(real_features))
    fake_var = sum((f - fake_mean)**2 for f in fake_features) / max(1, len(fake_features))
    diff = real_mean - fake_mean
    return math.sqrt(max(0, diff**2 + real_var + fake_var - 2 * math.sqrt(real_var * fake_var)))


def main():
    random.seed(42)
    sr = 8000
    duration = 0.5

    # 生成参考信号
    original = [math.sin(2 * math.pi * 440 * i / sr) for i in range(int(sr * duration))]

    # 不同质量的退化信号
    quality_levels = {
        "原始": original,
        "轻微退化": [x + random.gauss(0, 0.01) for x in original],
        "中等退化": [x + random.gauss(0, 0.05) for x in original],
        "严重退化": [x + random.gauss(0, 0.2) for x in original],
    }

    print("=== PESQ/STOI 评估（语音质量）===")
    print("  PESQ: 1.0-4.5 分，越高越好（1.0=差，4.5=接近原始）")
    print("  STOI: 0.0-1.0 分，越高越好（1.0=完全可懂）\n")
    for name, signal in quality_levels.items():
        pesq = pesq_score(original, signal)
        stoi = stoi_score(original, signal)
        print(f"  {name:<12}: PESQ={pesq:.2f}  STOI={stoi:.3f}")

    print(f"\n=== FAD（Fréchet 音频距离）===")
    print("  FAD: 越低越好，衡量生成音频与真实音频的分布距离")
    real_feats = [sum(original[i:i+100]) / 100 for i in range(0, len(original), 100)]
    for name, signal in quality_levels.items():
        fake_feats = [sum(signal[i:i+100]) / 100 for i in range(0, len(signal), 100)]
        fad = fad_score(real_feats, fake_feats)
        print(f"  {name:<12}: FAD={fad:.4f}")

    print(f"\n=== UTMOS（主观音质预测）===")
    print("  UTMOS: 预测人类 MOS 评分（0-5），越高越好")
    print("  ground truth: 4.08  |  F5-TTS: 3.95  |  Kokoro: 3.87")
    print("  VITS: 3.62          |  XTTS v2: 3.81")

    print(f"\n=== 评估指标速查 ===")
    print("  | 指标     | 用途              | 值范围    | 越高越好？ |")
    for name, use, rng, higher in [
        ("PESQ", "语音质量", "1.0-4.5", "✓"),
        ("STOI", "可懂度", "0.0-1.0", "✓"),
        ("FAD", "生成音频分布距离", "0-∞", "✗"),
        ("UTMOS", "主观音质预测", "0-5", "✓"),
        ("WER", "转录错误率", "0%-100%", "✗"),
        ("CER", "字符错误率", "0%-100%", "✗"),
    ]:
        print(f"  | {name:<8} | {use:<16} | {rng:<10} | {higher:<10} |")


if __name__ == "__main__":
    main()
