# 哈希嵌入——stdlib-only 稠密嵌入 + 余弦相似度
# 对应课程：阶段 05 · 22
# 非 Transformer 级别——但展示了嵌入的形态：分词→向量→归一化→点积

import hashlib, math, re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_token(token, dim, seed=0):
    h = hashlib.md5(f"{seed}:{token}".encode()).digest()
    return int.from_bytes(h[:4], "big") % dim


def hash_embed(text, dim=256):
    """随机哈希投影——每个 token 映射到向量中的一个位置，带随机符号。"""
    vec = [0.0] * dim
    for tok in tokenize(text):
        idx = hash_token(tok, dim)
        sign = 1.0 if hash_token(tok, 2, seed=1) == 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError(f"维度不匹配: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def main():
    docs = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "cats and dogs are common pets",
        "machine learning is a field of artificial intelligence",
        "deep learning uses neural networks with many layers",
    ]

    queries = [
        ("the feline is on the floor", 0),       # 与"猫在垫子上"最相关
        ("AI and neural nets", 3),                # 与"机器学习"最相关
    ]

    embs = [hash_embed(d, dim=128) for d in docs]

    print("=== 哈希嵌入（128 维）===")
    print("文档:")
    for i, d in enumerate(docs):
        print(f"  [{i}] {d}")

    print(f"\n查询相似度排名:")
    for q, expected in queries:
        q_emb = hash_embed(q, dim=128)
        scores = [(i, cosine(q_emb, e)) for i, e in enumerate(embs)]
        scores.sort(key=lambda x: -x[1])
        print(f"\n  查询: '{q}' (期望最近={expected})")
        for i, s in scores:
            mark = "←" if i == expected else ""
            print(f"    [{i}] {s:.3f} {mark}  {docs[i][:50]}")

    print("\n注意：随机哈希投影不是 Transformer 嵌入。")
    print("但它展示了任务形态：文本→向量→余弦排序。")


if __name__ == "__main__":
    main()
