import math

SPECIAL = {
    "SOT":           "<|startoftranscript|>",
    "EOT":           "<|endoftranscript|>",
    "TRANSCRIBE":    "<|transcribe|>",
    "TRANSLATE":     "<|translate|>",
    "NO_TIMESTAMPS": "<|notimestamps|>",
    "NO_SPEECH":     "<|nospeech|>",
}
LANG = {"en": "<|en|>", "fr": "<|fr|>", "ja": "<|ja|>", "zh": "<|zh|>"}


def build_prompt(language, task="transcribe", timestamps=False):
    toks = [SPECIAL["SOT"], LANG[language]]
    toks.append(SPECIAL["TRANSCRIBE"] if task == "transcribe" else SPECIAL["TRANSLATE"])
    if not timestamps:
        toks.append(SPECIAL["NO_TIMESTAMPS"])
    return toks


def chunk_schedule(total_seconds, chunk_s=30.0, stride_s=5.0):
    if total_seconds <= chunk_s:
        return [(0.0, total_seconds)]
    out, start, step = [], 0.0, chunk_s - stride_s
    while start < total_seconds:
        end = min(total_seconds, start + chunk_s)
        out.append((round(start, 2), round(end, 2)))
        if end == total_seconds: break
        start += step
    return out


def encoder_frames(seconds, sr=16000, hop=160):
    return 1 + (int(seconds * sr) - 400) // hop


def transformer_params(n_layers, d_model, d_ff, n_heads, vocab):
    per_block = 4*d_model*d_model + 2*d_model*d_ff + 4*d_model
    enc = n_layers * per_block
    dec = n_layers * (per_block + 4*d_model*d_model + 4*d_model)
    embed = vocab * d_model + 3000 * d_model
    return enc, dec, embed


def lora_params(n_layers, d_model, rank=16, modules=("q_proj", "v_proj")):
    per_module = 2 * d_model * rank
    return n_layers * 2 * len(modules) * per_module


def main():
    print("=== Whisper 解码器提示 ===")
    prompts = [
        ("en", "transcribe", False),
        ("fr", "translate", False),
        ("ja", "transcribe", True),
        ("zh", "transcribe", False),
    ]
    for lang, task, ts in prompts:
        p = build_prompt(lang, task=task, timestamps=ts)
        print(f"  {lang} {task} ts={ts}: {' '.join(p)}")

    print("\n=== 编码器帧预算 ===")
    for s in [1.0, 10.0, 30.0]:
        print(f"  {s:4.1f}s @16kHz -> {encoder_frames(s)} 帧")
    print("  Whisper 将所有输入零填充到 30s -> 3000 帧 -> 1500 编码器 token")

    print("\n=== 分块策略 (10min) ===")
    sched = chunk_schedule(600.0)
    print(f"  块数 (30s窗口, 5s步进): {len(sched)}")
    for start, end in sched[:5]:
        print(f"    {start:6.1f}s -> {end:6.1f}s")

    print("\n=== 各规格参数量 ===")
    configs = [
        ("Tiny", 4, 384, 1536, 6),
        ("Base", 6, 512, 2048, 8),
        ("Small", 12, 768, 3072, 12),
        ("Medium", 24, 1024, 4096, 16),
        ("Large-v3", 32, 1280, 5120, 20),
        ("Turbo", 4, 1280, 5120, 20),
    ]
    for name, layers, d, d_ff, heads in configs:
        enc, dec, embed = transformer_params(layers, d, d_ff, heads, 51865)
        if name == "Turbo":
            enc = transformer_params(32, d, d_ff, heads, 51865)[0]
        print(f"  {name:<10} enc={enc/1e6:5.1f}M  dec={dec/1e6:5.1f}M  total={enc+dec+embed:.1f}M")

    print("\n=== LoRA r=16 可训练参数 ===")
    for name, layers, d, *_ in configs[3:6]:
        lp = lora_params(layers, d, rank=16)
        print(f"  {name:<10} LoRA: {lp/1e6:.3f}M (原模型 ~{(lora_params(layers, d, 16)/transformer_params(layers, d, d_ff, configs[0][4], 51865)[2]):.1%}?)")

    print("\n=== 2026 推理配方 ===")
    recipes = [
        ("离线英文最佳WER", "large-v3-turbo + Silero VAD"),
        ("长音频+词级时间戳", "whisperx + wav2vec 2.0 强制对齐"),
        ("流式 2s延迟", "whisper-streaming 或 Parakeet-TDT"),
        ("移动端/边缘设备", "whisper-tiny int8 或 moonshine"),
        ("低资源语言", "LoRA 微调 2-20h 领域音频"),
    ]
    for s, r in recipes:
        print(f"  {s:<20} -> {r}")


if __name__ == "__main__":
    main()
