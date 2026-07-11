# RAG 分块策略——固定、递归、语义三种分块 + 检索评估
# 对应课程：阶段 05 · 23

import hashlib, math, re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embed(text, dim=256):
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hashlib.md5(tok.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def chunk_fixed(text, size, overlap=0):
    if size <= 0: raise ValueError("size 必须为正数")
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size: return [text]
    for sep in seps:
        if sep not in text: continue
        parts, chunks, buf = text.split(sep), [], ""
        for p in parts:
            if len(p) > size:
                if buf: chunks.append(buf); buf = ""
                chunks.extend(chunk_recursive(p, size, seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size: buf = candidate
            else: chunks.append(buf); buf = p
        if buf: chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)


def main():
    text = (
        "The first chapter introduces the main characters. "
        "It describes their backgrounds and motivations. "
        "\n\n"
        "Chapter two dives into the conflict. "
        "The protagonist faces their first major challenge. "
        "They must choose between loyalty and ambition. "
        "\n\n"
        "The final chapter resolves the story. "
        "All threads come together in a dramatic conclusion. "
        "The reader is left with a sense of hope."
    )

    print("=== 固定分块 (size=100, overlap=0) ===")
    for i, c in enumerate(chunk_fixed(text, 100)):
        print(f"  [{i}] ({len(c)} 字符) {c[:80]}...")

    print(f"\n=== 递归分块 (size=100) ===")
    for i, c in enumerate(chunk_recursive(text, 100)):
        print(f"  [{i}] ({len(c)} 字符) {c[:80]}...")

    print(f"\n=== 三种策略的块数量对比 ===")
    print(f"  固定(100,0):    {len(chunk_fixed(text, 100))} 块")
    print(f"  固定(100,20):   {len(chunk_fixed(text, 100, 20))} 块")
    print(f"  递归(100):      {len(chunk_recursive(text, 100))} 块")
    print("  递归保持段落/句子边界完整——固定会从句子中间截断。")


if __name__ == "__main__":
    main()
