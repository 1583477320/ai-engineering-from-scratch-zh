# 语音识别：CTC 贪心/束搜索解码 + WER
# 依赖：无（纯标准库）
# 对应课程：阶段 06 · 04

import math
import random

BLANK = 0
VOCAB = "_abcdefghijklmnopqrstuvwxyz "  # index 0 = blank


def ctc_greedy(frame_probs):
    """贪心 CTC 解码——每帧取 argmax，合并重复，去除 blank。"""
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_probs]
    out, prev = [], -1
    for p in preds:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return "".join(VOCAB[i] for i in out)


def ctc_beam(frame_probs, beam_width=8):
    """束搜索 CTC 解码——保持 beam_width 个最优部分序列。"""
    beams = [((), 0.0)]
    for p in frame_probs:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        new_beams = {}
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                if t == BLANK:
                    new_seq = seq
                elif seq and seq[-1] == t:
                    new_seq = seq  # 合并重复
                else:
                    new_seq = seq + (t,)
                if new_seq in new_beams:
                    new_beams[new_seq] = math.log(
                        math.exp(new_beams[new_seq]) + math.exp(lp + lpt))
                else:
                    new_beams[new_seq] = lp + lpt
        beams = sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width]
    best = beams[0][0]
    return "".join(VOCAB[i] for i in best)


def wer(ref, hyp):
    """词错误率 (Word Error Rate)——基于编辑距离。"""
    r, h = ref.split(), hyp.split()
    nr = len(r)
    if nr == 0:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[nr][len(h)] / nr


def one_hot_like(char, noise=0.02, vocab_size=len(VOCAB)):
    base = [noise] * vocab_size
    idx = VOCAB.index(char)
    base[idx] = 1.0 - noise * (vocab_size - 1)
    return base


def build_frame_probs(target, duration_per_char=3, blank_runs=1):
    """为教学示例构建逐帧概率分布。"""
    random.seed(0)
    frames = []
    for c in target:
        for _ in range(duration_per_char):
            frames.append(one_hot_like(c))
        for _ in range(blank_runs):
            frames.append(one_hot_like("_"))
    return frames


def corrupt(probs, n_swaps=3, swap_strength=0.4):
    """给 logits 加噪声——模拟实际模型的不完美输出。"""
    random.seed(1)
    out = [list(p) for p in probs]
    for _ in range(n_swaps):
        i = random.randrange(len(out))
        j1 = random.randrange(len(out[i]))
        j2 = random.randrange(len(out[i]))
        out[i][j1] -= swap_strength
        out[i][j2] += swap_strength
    return out


def main():
    target = "hello world"
    print(f"=== Step 1: 构建逐帧 CTC 输出 ===")
    print(f"  目标: {target!r}")
    probs = build_frame_probs(target, duration_per_char=3, blank_runs=1)
    print(f"  帧数: {len(probs)}, 词表: {len(VOCAB)} (索引 0=blank)")

    print(f"\n=== Step 2: 贪心解码（合并重复，去除 blank）===")
    greedy = ctc_greedy(probs)
    print(f"  贪心: {greedy!r}")

    print(f"\n=== Step 3: 束搜索解码（束宽 8）===")
    beam = ctc_beam(probs, beam_width=8)
    print(f"  束搜索: {beam!r}")
    print(f"  注意：简化的束搜索未处理重复合并状态")
    print(f"  完整实现需使用前缀树 + blank 状态跟踪")

    print(f"\n=== Step 4: 给 logits 加噪声——束搜索应优于贪心 ===")
    corrupted = corrupt(probs, n_swaps=6, swap_strength=0.6)
    g2 = ctc_greedy(corrupted)
    b2 = ctc_beam(corrupted, beam_width=16)
    print(f"  贪心: {g2!r}")
    print(f"  束搜索: {b2!r}")

    print(f"\n=== Step 5: WER 演示 ===")
    ref = "hello world this is a test"
    cases = {
        "完美": "hello world this is a test",
        "一次替换": "hello world this is the test",
        "一次删除": "hello world this a test",
        "一次插入": "hello world this is a big test",
        "垃圾": "bye everyone nothing here",
    }
    for label, hyp in cases.items():
        print(f"  {label:<14} WER={wer(ref, hyp):.3f}  hyp={hyp!r}")

    print(f"\n=== Step 6: 2026 年 LibriSpeech test-clean 榜单 ===")
    print("  | 模型                | WER  | 参数量 |")
    table = [
        ("Parakeet-TDT-1.1B", 1.40, "1.1B"),
        ("Canary-1B Flash",   1.48, "1B"),
        ("Whisper-L-v3-turbo", 1.58, "809M"),
        ("Seamless M4T v2",   1.70, "2.3B"),
        ("wav2vec 2.0 Large", 1.92, "317M"),
    ]
    for name, w, p in table:
        print(f"  | {name:<21} | {w:.2f} | {p:<6} |")


if __name__ == "__main__":
    main()
