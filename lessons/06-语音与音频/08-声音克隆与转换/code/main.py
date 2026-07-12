"""声音克隆演示：模拟 (内容, 说话人) 分解与交换
构建玩具"内容"向量（哈希）和"说话人"向量（每说话人音调轮廓）
演示克隆后的音频保持内容，说话人嵌入余弦指向目标。
"""
import hashlib, math, random

def content_vector(text, dim=64):
    """确定性'内容'表示——玩具 PPG 代替品。"""
    h = hashlib.sha256(text.encode()).digest()
    expanded = (h * ((dim + len(h) - 1) // len(h)))[:dim]
    return [b / 255.0 - 0.5 for b in expanded]

def speaker_vector(seed, dim=64):
    """确定性'说话人嵌入'——玩具 ECAPA-TDNN 代替品。"""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1e-12
    return [x / norm for x in v]

def fake_tts(content, speaker, mix=0.5):
    return [(1 - mix) * c + mix * s for c, s in zip(content, speaker)]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def extract_speaker(wave, ref_speakers):
    sims = [(name, cosine(wave, vec)) for name, vec in ref_speakers.items()]
    return max(sims, key=lambda x: x[1])

def extract_content(wave, ref_contents):
    sims = [(text, cosine(wave, vec)) for text, vec in ref_contents.items()]
    return max(sims, key=lambda x: x[1])

def watermark(wave, payload_bits, strength=0.003):
    """玩具不可闻水印——按位 DC 偏移。"""
    n = len(payload_bits)
    return [wave[i] + ((1 if payload_bits[i % n] else -1) * strength) for i in range(len(wave))]

def detect_watermark(orig, wm, n_bits=32):
    diff = [a - b for a, b in zip(wm, orig)]
    return [1 if sum(diff[b::n_bits]) / max(1, len(diff[b::n_bits])) > 0 else 0 for b in range(n_bits)]

def bit_accuracy(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)

def main():
    DIM = 64
    alice = speaker_vector("alice", DIM)
    bob = speaker_vector("bob", DIM)
    carol = speaker_vector("carol", DIM)
    speakers = {"alice": alice, "bob": bob, "carol": carol}

    text_greet = "hello this is a test"
    text_remind = "please remember to water plants"
    contents = {text_greet: content_vector(text_greet, DIM),
                text_remind: content_vector(text_remind, DIM)}

    print("=== Step 1: alice 说 'hello' ===")
    wav = fake_tts(contents[text_greet], alice)
    name, sc = extract_speaker(wav, speakers)
    txt, tc = extract_content(wav, contents)
    print(f"  说话人: {name} (cos={sc:.3f}), 内容: {txt[:30]!r}")

    print("\n=== Step 2: 零样本克隆——alice 的声音说 bob 的话 ===")
    wav_clone = fake_tts(contents[text_remind], alice)
    name, sc = extract_speaker(wav_clone, speakers)
    txt, tc = extract_content(wav_clone, contents)
    print(f"  说话人: {name} (cos={sc:.3f})  ← 应保持 alice")
    print(f"  内容: {txt[:30]!r}")

    print("\n=== Step 3: 声音转换——将 bob 的话转为 alice ===")
    wav_bob = fake_tts(contents[text_remind], bob)
    matched_text, _ = extract_content(wav_bob, contents)
    wav_conv = fake_tts(contents[matched_text], alice)
    name, sc = extract_speaker(wav_conv, speakers)
    print(f"  转换后说话人: {name} (cos={sc:.3f}), 内容: {matched_text!r}")

    print("\n=== Step 4: 水印演示 ===")
    payload = [int(b) for b in bin(0xDEADBEEF)[2:].zfill(32)]
    wm = watermark(wav_clone, payload)
    detected = detect_watermark(wav_clone, wm, 32)
    acc = bit_accuracy(payload, detected)
    print(f"  payload:  {''.join(map(str, payload[:16]))}...")
    print(f"  detected: {''.join(map(str, detected[:16]))}...")
    print(f"  比特准确率: {acc*100:.1f}% (SilentCipher 在 MP3 重编码后 ~99%)")

    print("\n=== Step 5: 2026 克隆排行榜 ===")
    print("  | 模型          | SECS | CER% | 参数量 |")
    for name, s, c, p in [("VoiceBox",0.78,2.1,"330M"),("VALL-E 2",0.77,2.4,"370M"),
                            ("F5-TTS",0.72,2.1,"335M"),("OpenVoice v2",0.70,2.8,"220M"),
                            ("XTTS v2",0.65,3.5,"470M")]:
        print(f"  | {name:<14} | {s:.2f}  | {c:.1f}   | {p:<4} |")

if __name__ == "__main__":
    main()
