# NIAH（大海捞针）——定制针测试 + 多针变体
# 对应课程：阶段 05 · 28

import random, re

FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a stitch in time saves nine birds of a feather flock together "
    "every cloud has a silver lining actions speak louder than words "
).split()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    """在干草堆的 depth_ratio 处插入一根针。"""
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio 必须在 [0, 1]，收到 {depth_ratio}")
    filler = tokenize(filler_text)
    needle_t = tokenize(needle)
    body_len = max(total_tokens - len(needle_t), 0)
    while len(filler) < body_len:
        filler = filler + filler
    filler = filler[:body_len]
    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler[:insert_at] + needle_t + filler[insert_at:]
    return " ".join(haystack)


def build_multi_needle(filler_text, needles, total_tokens):
    """多针变体：分散在不同深度。"""
    depths = [0.1, 0.4, 0.7]
    result = []
    filler = tokenize(filler_text)
    while len(filler) < total_tokens:
        filler = filler + filler
    filler = filler[:total_tokens]
    for depth, needle in zip(depths, needles):
        pos = int(len(filler) * depth)
        result.append(f"Fact: {needle}.")
    return " ".join(result[:3] + [" ".join(filler[len(result):])])


def fake_model_query(haystack, question, needle, seed=0):
    """模拟模型查询——在 haystack 中随机检索，偶尔找到针。"""
    rng = random.Random(seed + hash(question) % 1000)
    tokens = tokenize(haystack)
    needle_tokens = set(tokenize(needle))
    # 简单模拟：随机采样一个 span，看是否包含 needle
    span_start = rng.randint(0, max(0, len(tokens) - 10))
    span = set(tokens[span_start:span_start + 10])
    return needle.lower() in " ".join(span)


def main():
    needle = "神奇密码是 Pineapple42"
    question = "神奇密码是什么？"
    filler = " ".join(FILLER_WORDS * 50)

    print("=== NIAH（大海捞针）测试 ===")
    for length in [100, 500, 2000]:
        for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hs = build_haystack(filler, needle, depth, length)
            ok = fake_model_query(hs, question, needle, seed=42)
            print(f"  长度={length:5d}  深度={depth:.2f}  {'✓' if ok else '✗'}")

    print("\n=== 多针 MRCR 测试 ===")
    needles_list = ["Pineapple42", "Dragonfruit77", "Mango99"]
    hs = build_multi_needle(filler, needles_list, 1000)
    found = sum(1 for n in needles_list if n.lower() in hs.lower())
    print(f"  多针 haystack 长度 ~1000 tokens")
    print(f"  插入 {len(needles_list)} 根针，确认全部存在: {found}/{len(needles_list)}")
    print(f"  模型需检索全部 {len(needles_list)} 根针——单针成功≠多针成功。")

    print("\n注意：此玩具仅演示测试结构。真实 NIAH 需在真实 LLM 上运行。")


if __name__ == "__main__":
    main()
