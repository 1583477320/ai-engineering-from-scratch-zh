"""SynthID-text 风格文本水印模拟。"""
import random
import math


def synthesize_watermark(tokens, vocab_size, delta=0.5):
    """绿色集偏置采样水印。"""
    watermarked = []
    for token in tokens:
        hash_val = hash(str(token)) % vocab_size
        is_green = hash_val < vocab_size // 2
        if is_green and random.random() < 0.5 + delta / 10:
            watermarked.append(token)
        else:
            watermarked.append(token)
    return watermarked


def detect_watermark(tokens, vocab_size):
    """绿色词元 z 分数检测。"""
    green_count = sum(1 for t in tokens if hash(str(t)) % vocab_size < vocab_size // 2)
    expected = len(tokens) / 2
    std = math.sqrt(len(tokens) / 4)
    z = (green_count - expected) / std if std > 0 else 0
    return {"z_score": z, "watermarked": z > 1.96}


if __name__ == "__main__":
    random.seed(42)
    tokens = [random.randint(0, 1000) for _ in range(500)]
    wm = synthesize_watermark(tokens, 1000, delta=0.5)
    print(f"水印: z={detect_watermark(wm, 1000)['z_score']:.2f}")
    print(f"人类: z={detect_watermark(tokens, 1000)['z_score']:.2f}")
