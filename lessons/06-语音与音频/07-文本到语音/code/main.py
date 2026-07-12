# TTS 内部演示：音素查找 + 时长估计 + 梅尔帧调度
# 纯标准库。构建玩具英语字素到音素映射，估计时长，打印帧调度。
# 对应课程：阶段 06 · 07

import math, random

G2P = {
    " ": ["_"], "a": ["AH"], "b": ["B"], "c": ["K"], "d": ["D"],
    "e": ["EH"], "f": ["F"], "g": ["G"], "h": ["HH"], "i": ["IH"],
    "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"],
    "o": ["AO"], "p": ["P"], "q": ["K"], "r": ["R"], "s": ["S"],
    "t": ["T"], "u": ["UH"], "v": ["V"], "w": ["W"], "x": ["K", "S"],
    "y": ["Y"], "z": ["Z"],
    "the": ["DH", "AH"], "ing": ["IH", "NG"], "er": ["ER"],
    "sh": ["SH"], "ch": ["CH"], "th": ["TH"],
    ".": ["_PAUSE_"], ",": ["_SHORT_"], "?": ["_PAUSE_"], "!": ["_PAUSE_"],
}

DURATION_FRAMES = {
    "AA": 9, "AE": 7, "AH": 6, "AO": 8, "AW": 9, "AY": 8, "B": 4,
    "CH": 6, "D": 4, "DH": 5, "EH": 6, "ER": 7, "EY": 8, "F": 6,
    "G": 5, "HH": 4, "IH": 5, "IY": 7, "JH": 6, "K": 5, "L": 5,
    "M": 5, "N": 5, "NG": 6, "OW": 8, "OY": 9, "P": 5, "R": 5,
    "S": 6, "SH": 7, "T": 4, "TH": 5, "UH": 6, "UW": 8, "V": 5,
    "W": 5, "Y": 5, "Z": 6, "ZH": 7,
    "_": 3, "_SHORT_": 6, "_PAUSE_": 12,
}


def phonemize(text):
    """字素到音素转换——最长匹配优先（3→2→1字符）。"""
    text = text.lower()
    phones, i = [], 0
    while i < len(text):
        matched = False
        for length in (3, 2, 1):
            if i + length <= len(text):
                chunk = text[i : i + length]
                if chunk in G2P:
                    phones.extend(G2P[chunk])
                    i += length; matched = True; break
        if not matched:
            i += 1
    return phones


def duration(phones, jitter=0.1, seed=0):
    """估计每个音素的持续时长（帧数）——加随机抖动模拟自然变异。"""
    random.seed(seed)
    return [max(1, DURATION_FRAMES.get(p, 5) +
            int(round(DURATION_FRAMES.get(p, 5) * random.uniform(-jitter, jitter))))
            for p in phones]


def mel_schedule(phones, durs, hop_ms=12.5):
    """生成梅尔帧调度——每个音素的起止时间。"""
    sched, t = [], 0.0
    for p, d in zip(phones, durs):
        sched.append((p, t, t + d * hop_ms))
        t += d * hop_ms
    return sched, t


def main():
    text = "Please remind me to water the plants at 6 pm."

    print("=== Step 1: 字素到音素 ===")
    phones = phonemize(text)
    print(f"  text: {text!r}")
    print(f"  phones ({len(phones)}): {' '.join(phones[:20])}")

    print("\n=== Step 2: 估计每个音素时长 ===")
    durs = duration(phones, jitter=0.1, seed=42)
    print(f"  durations: {durs[:20]}")

    print("\n=== Step 3: 梅尔帧调度 (12.5ms hop) ===")
    sched, total_ms = mel_schedule(phones, durs)
    print(f"  总时长: {total_ms:.1f} ms ({total_ms / 1000:.2f} s)")
    for p, s, e in sched[:10]:
        print(f"    {p:<10} {s:6.1f} – {e:6.1f} ms")

    print("\n=== Step 4: 送入声码器的帧预算 ===")
    total_frames = sum(durs)
    audio_samples = total_frames * 300  # 12.5ms @ 24kHz
    print(f"  mel 帧数: {total_frames}")
    print(f"  音频采样 @24kHz: {audio_samples}")
    print(f"  内存预算: {total_frames * 80 * 4 / 1024:.1f} KB (mel, float32)")

    print("\n=== Step 5: 2026 TTS 质量榜单 ===")
    print("  | 模型           | UTMOS | CER% | 参数量 |")
    for name, u, c, s in [("ground truth", 4.08, 1.2, "—"),
                           ("F5-TTS", 3.95, 2.1, "335M"),
                           ("Kokoro v0.19", 3.87, 1.8, "82M"),
                           ("XTTS v2", 3.81, 3.5, "470M"),
                           ("VITS", 3.62, 3.1, "25M")]:
        print(f"  | {name:<14} | {u:.2f}  | {c:.1f}   | {s:<4} |")


if __name__ == "__main__":
    main()
