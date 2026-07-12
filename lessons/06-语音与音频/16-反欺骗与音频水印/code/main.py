# 音频反欺骗与水印演示
# 对应课程：阶段 06 · 16
"""反欺骗检测 + 音频水印"""

import hashlib
import math
import random


def detect_spoofing(audio_features, threshold=0.7):
    """模拟反欺骗检测——特征向量相似度判断。"""
    # 真实语音特征更稳定；合成语音特征分布偏移
    stability = sum(1 for i in range(1, len(audio_features))
                    if abs(audio_features[i] - audio_features[i-1]) < 0.1)
    score = stability / max(1, len(audio_features) - 1)
    return score > threshold


def watermark_embed(signal, payload_bits, strength=0.003):
    """不可闻音频水印——按位 DC 偏移。"""
    n_bits = len(payload_bits)
    return [signal[i] + ((1 if payload_bits[i % n_bits] else -1) * strength)
            for i in range(len(signal))]


def watermark_detect(original, watermarked, n_bits=32):
    """检测水印——差异信号的分段平均。"""
    diff = [a - b for a, b in zip(watermarked, original)]
    bits = []
    for b in range(n_bits):
        chunk = diff[b::n_bits]
        avg = sum(chunk) / max(1, len(chunk))
        bits.append(1 if avg > 0 else 0)
    return bits


def bit_accuracy(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def main():
    print("=== 反欺骗检测演示 ===")
    random.seed(42)
    real_features = [1.0 + random.gauss(0, 0.02) for _ in range(50)]
    fake_features = [random.uniform(0, 1) for _ in range(50)]

    print(f"  真实语音稳定性: {detect_spoofing(real_features):.3f}")
    print(f"  合成语音稳定性: {detect_spoofing(fake_features):.3f}")
    print(f"  判断: 真实={detect_spoofing(real_features)}, 合成={detect_spoofing(fake_features)}")

    print("\n=== 音频水印演示 ===")
    sr = 8000
    signal = [math.sin(2 * math.pi * 440 * i / sr) + random.gauss(0, 0.01)
              for i in range(sr)]
    payload = [int(b) for b in bin(0xDEADBEEF)[2:].zfill(32)]

    wm = watermark_embed(signal, payload, strength=0.003)
    detected = watermark_detect(signal, wm, 32)
    acc = bit_accuracy(payload, detected)

    print(f"  Payload:   {''.join(map(str, payload[:16]))}...")
    print(f"  Detected:  {''.join(map(str, detected[:16]))}...")
    print(f"  比特准确率: {acc * 100:.1f}%")
    print(f"  注意: 生产系统（SilentCipher/PerTh）在 MP3 重编码后仍保持 >99% 准确率")

    print("\n=== 法律合规要求 ===")
    print("  欧盟 AI 法案 (2026.08生效):")
    print("    - 所有 AI 生成语音必须包含不可闻水印")
    print("    - 每次克隆必须关联可验证的同意记录")
    print("    - 未经同意不得克隆他人声音")
    print("  加州 AB 2905 (2025生效):")
    print("    - 类似要求，面向商业用途")
    print("    - 要求元数据披露 AI 生成")


if __name__ == "__main__":
    main()
