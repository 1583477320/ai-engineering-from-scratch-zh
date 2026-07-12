# 神经音频编解码器：RVQ 概念演示
# 对应课程：阶段 06 · 13

import math
from typing import List


def quantize(value: float, n_codes: int) -> int:
    """简单量化：将连续值映射到最近的离散码。"""
    return max(0, min(n_codes - 1, int((value + 1) * n_codes / 2)))


def dequantize(code: int, n_codes: int) -> float:
    """反量化：将离散码映射回连续值。"""
    return code * 2 / n_codes - 1


def rvq_encode(signal: List[float], n_codebooks: int = 4, n_codes: int = 8) -> List[List[int]]:
    """RVQ 编码：级联量化残差。"""
    codebooks = []
    residual = list(signal)

    for cb in range(n_codebooks):
        encoded = []
        for i, val in enumerate(residual):
            code = quantize(val, n_codes)
            encoded.append(code)
            # 计算残差
            residual[i] = val - dequantize(code, n_codes)
        codebooks.append(encoded)

    return codebooks


def rvq_decode(codebooks: List[List[int]], n_codes: int = 8) -> List[float]:
    """RVQ 解码：累加所有 codebook 的重建。"""
    n_frames = len(codebooks[0]) if codebooks else 0
    signal = [0.0] * n_frames
    for cb_codes in codebooks:
        for i, code in enumerate(cb_codes):
            signal[i] += dequantize(code, n_codes)
    return signal


def codec_stats(original: List[float], decoded: List[float]) -> dict:
    """计算编解码统计。"""
    mse = sum((a - b) ** 2 for a, b in zip(original, decoded)) / len(original)
    snr = 10 * math.log10(sum(a**2 for a in original) / (mse + 1e-10))
    return {"MSE": f"{mse:.6f}", "SNR(dB)": f"{snr:.1f}"}


def main():
    # 生成测试信号
    import random
    random.seed(42)
    sr = 24000
    duration = 0.05  # 50ms
    n_samples = int(sr * duration)
    signal = [math.sin(2 * math.pi * 440 * i / sr) + 0.3 * random.gauss(0, 1) for i in range(n_samples)]

    print("=== RVQ 编解码演示 ===\n")

    for n_cb in [1, 2, 4, 8]:
        codebooks = rvq_encode(signal, n_codebooks=n_cb, n_codes=8)
        decoded = rvq_decode(codebooks, n_codes=8)
        stats = codec_stats(signal, decoded)
        total_tokens = len(signal) * n_cb  # 每帧 × 每codebook一个token
        kbps = total_tokens * 3 / (duration * 1000)  # 3 bit per token

        print(f"  {n_cb} codebooks:")
        print(f"    tokens/秒: {int(1/duration * n_cb)}")
        print(f"    比特率: {kbps:.1f} kbps")
        print(f"    SNR: {stats['SNR(dB)']} dB")
        print()

    print("=== 帧率对 LM 的影响 ===")
    codecs = [
        ("EnCodec-24k", 75, "音乐、通用音频"),
        ("DAC-44.1k", 86, "高保真音乐"),
        ("SNAC 粗帧", 12, "AR-LM 高效"),
        ("Mimi", 12.5, "流式语音"),
    ]
    print(f"  {'编解码器':<16} {'帧率':>6} {'1秒=N帧':>10} {'适用场景'}")
    for name, fps, use in codecs:
        print(f"  {name:<16} {fps:>5}Hz {fps:>9} {use}")

    print(f"\n=== 语义-声学分离 ===")
    print("  Mimi: codebook 0 从 WavLM 蒸馏 → 语义内容")
    print("       codebook 1-7 → 声学残差（音色、说话人、噪声）")
    print("  这种分离让 LLM 只需建模语言结构，声学由解码器处理")


if __name__ == "__main__":
    main()
