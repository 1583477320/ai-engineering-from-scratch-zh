"""音乐生成演示：基于符号级的和弦/鼓点生成（玩具）
真实音乐生成用神经编解码 LM（MusicGen/ACE-Step）或潜在扩散（Stable Audio）
这里用"随时间变化的 token"概念做符号级演示
"""
import random

MAJOR_KEYS = {
    "C": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
    "G": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
    "D": ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
    "A": ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
}

COMMON_PROGRESSIONS = {
    "pop": [1,5,6,4], "ballad": [1,6,4,5], "jazz": [2,5,1,6],
    "rock": [1,4,5,1], "lofi": [6,4,1,5],
}

DRUM_PATTERNS = {
    "pop": "X.o.X.o.X.o.X.o.", "rock": "X..oX..oX..oX..o",
    "lofi": "X...o...X...o.o.", "jazz": "X.oox.oxX.oox.ox", "trap": "Xooox.oxXooox.ox",
}

def chord_progression(key, genre, bars=8):
    scale = MAJOR_KEYS[key]
    pat = COMMON_PROGRESSIONS.get(genre, COMMON_PROGRESSIONS["pop"])
    return [scale[i - 1] for i in (pat * (bars // len(pat) + 1))[:bars]]

def drum_pattern(genre, bars=8):
    base = DRUM_PATTERNS.get(genre, DRUM_PATTERNS["pop"])
    return (base * bars)[:bars * 16]

def fake_generate(prompt, rng=None):
    rng = rng or random.Random(0)
    prompt_lower = prompt.lower()
    key = next((k for k in MAJOR_KEYS if f" {k.lower()}" in f" {prompt_lower} "), "C")
    genre = next((g for g in COMMON_PROGRESSIONS if g in prompt_lower), "pop")
    bpm = 120
    for token in prompt_lower.split():
        if token.endswith("bpm"):
            try: bpm = int(token[:-3])
            except: pass
    return {"key": key, "genre": genre, "bpm": bpm, "bars": 8,
            "chords": chord_progression(key, genre, 8), "drums": drum_pattern(genre, 8)}

def main():
    prompts = [
        "upbeat pop in G major at 128 bpm",
        "slow lofi groove in C",
        "rock anthem in D at 140 bpm",
        "jazz swing in A",
    ]
    print("=== Step 1: 文本 → 符号音乐（玩具）===")
    for p in prompts:
        piece = fake_generate(p)
        print(f"  prompt: {p!r}")
        print(f"  调={piece['key']}  风格={piece['genre']}  速度={piece['bpm']}bpm")
        chord_str = " | ".join(piece['chords'])
        print(f"  和弦: {chord_str}")
        print(f"  鼓点: {piece['drums']}")
        print()

    print("=== Step 2: 2026 音乐生成模型速查 ===")
    print("  | 模型             | 参数量(M) | 时长 | 人声 | 许可证 |")
    for name, p, l, v, lic in [("MusicGen-large",3300,"30s","否","MIT"),
                                 ("Stable Audio Open",1200,"47s","否","非商用"),
                                 ("ACE-Step XL",4000,"2min+","是","Apache-2.0"),
                                 ("YuE",7000,"2min+","是","Apache-2.0"),
                                 ("Suno v5",0,"4min","是","商业"),
                                 ("Udio v4",0,"4min","是+stems","商业")]:
        print(f"  | {name:<18} | {p:>10} | {l:>5} | {v:<6} | {lic:<12} |")

if __name__ == "__main__":
    main()
