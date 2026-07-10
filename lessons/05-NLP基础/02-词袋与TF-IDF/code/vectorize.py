# vectorize.py — 从零实现词袋模型与 TF-IDF
# 依赖：无（纯标准库实现）
# 安装：无需额外安装（scikit-learn 部分需要 pip install scikit-learn）
# 对应课程：阶段 05 · 02（词袋与 TF-IDF）

import math
import re
from typing import List, Dict, Tuple


# ============================================================
# 1. 简易分词器（英文）
# ============================================================

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def tokenize(text: str) -> List[str]:
    """简易英文分词——小写化 + 正则匹配。"""
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ============================================================
# 2. 构建词表
# ============================================================

def build_vocab(docs: List[List[str]]) -> Dict[str, int]:
    """从分词后的文档列表构建词表。

    返回 {词语: 索引} 映射。插入顺序决定了索引——第一个出现的词索引为 0。
    scikit-learn 默认按字母排序，这里用出现顺序以便教学理解。
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


# ============================================================
# 3. 词袋模型（BoW）
# ============================================================

def bag_of_words(docs: List[List[str]],
                 vocab: Dict[str, int]) -> List[List[int]]:
    """词袋模型——将每个文档转为词频向量。

    返回形状为 (n_docs, vocab_size) 的整数矩阵。
    每行是一个文档，每列是词表中的一个词。
    matrix[i][j] = 词 j 在文档 i 中出现的次数。
    词序信息完全丢弃——这正是"词袋"的含义。
    """
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix


# ============================================================
# 4. 词频（TF）与文档频率（DF）
# ============================================================

def term_frequency(doc_bow: List[int], doc_length: int) -> List[float]:
    """词频——每个词在文档中出现的次数 / 文档总词数。

    归一化的目的：消除文档长度偏差。一篇 1000 词的文档
    天然比一篇 50 词的文档有更大的词频，不归一化则长文档
    在相似度计算中占据主导地位。
    """
    if doc_length == 0:
        return [0.0] * len(doc_bow)
    return [c / doc_length for c in doc_bow]


def document_frequency(bow_matrix: List[List[int]]) -> List[int]:
    """文档频率——每个词出现在多少个文档中（至少出现一次即计入）。

    返回长度为 vocab_size 的列表。df[j] = 包含词 j 的文档数。
    """
    if not bow_matrix:
        return []
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


# ============================================================
# 5. 逆文档频率（IDF）
# ============================================================

def inverse_document_frequency(df: List[int], n_docs: int) -> List[float]:
    """逆文档频率——文档频率的倒数取对数。

    使用 scikit-learn 默认的平滑公式：
        idf = log((N + 1) / (df + 1)) + 1

    两个平滑技巧：
    - (N+1)/(df+1) 避免 log(x/0)（当 df=0 时也不会爆炸）
    - 末尾 +1 确保出现在全部文档中的词 IDF=1 而非 0
      一个无处不在的词应该被降权，但不应该被彻底归零
    """
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]


# ============================================================
# 6. TF-IDF
# ============================================================

def tfidf(bow_matrix: List[List[int]]) -> List[List[float]]:
    """TF-IDF——将词袋矩阵转换为 TF-IDF 加权矩阵。

    公式：TF-IDF(w, d) = TF(w, d) × IDF(w)

    TF 衡量"这个词在这篇文档中有多重要"
    IDF 衡量"这个词在整个语料库中有多稀缺"
    两个信号相乘 = 既高频又稀缺的词获得最高权重
    """
    n_docs = len(bow_matrix)
    if n_docs == 0:
        return []
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([t * i for t, i in zip(tf, idf)])
    return out


# ============================================================
# 7. L2 归一化
# ============================================================

def l2_normalize(matrix: List[List[float]]) -> List[List[float]]:
    """L2 归一化——将每个文档向量缩放到单位长度。

    归一化后，任意两个向量的余弦相似度退化为点积：
        cosine(a, b) = a·b / (||a|| × ||b||) = a·b / (1 × 1) = a·b
    """
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0.0 for x in row])
    return out


# ============================================================
# 8. 余弦相似度
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度——前提是输入向量已经 L2 归一化。

    返回值范围 [-1, 1]。1 表示完全同向（内容相同），0 表示正交（无关）。
    """
    return sum(x * y for x, y in zip(a, b))


# ============================================================
# 9. 中文分词 + TF-IDF 组合（jieba + 自实现）
# ============================================================

def tokenize_zh(text: str) -> List[str]:
    """中文分词——使用 jieba。

    如果 jieba 未安装，回退为逐字切分（char-level）。
    逐字切分会损失语义信息，仅用于演示时避免报错。
    """
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 回退：逐字切分（仅用于演示）
        return list(text)


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === 英文演示 ===
    raw_en = [
        "The cat sat on the mat.",
        "The dog sat on the mat.",
        "The cat ran across the room.",
        "A cat and a dog played together in the room.",
    ]
    docs_en = [tokenize(r) for r in raw_en]
    vocab_en = build_vocab(docs_en)
    bow_en = bag_of_words(docs_en, vocab_en)
    tfidf_en = l2_normalize(tfidf(bow_en))

    # 按字母排序展示（便于阅读）
    words_en = sorted(vocab_en, key=lambda w: vocab_en[w])
    print("=" * 60)
    print("英文词袋模型 + TF-IDF")
    print("=" * 60)
    print(f"词表 ({len(vocab_en)} 个词): {words_en}")
    print()

    for i, (raw, vec) in enumerate(zip(raw_en, tfidf_en)):
        # 只显示非零项
        nonzero = [(words_en[j], f"{v:.3f}") for j, v in enumerate(vec) if v > 0.01]
        print(f"文档 {i}: \"{raw}\"")
        print(f"  TF-IDF 非零项: {nonzero}")
        print()

    # 余弦相似度矩阵
    print("余弦相似度矩阵（L2 归一化后 = 点积）:")
    for i in range(len(tfidf_en)):
        row = [f"{cosine_similarity(tfidf_en[i], tfidf_en[j]):.2f}"
               for j in range(len(tfidf_en))]
        print(f"  d{i}: {row}")

    # === 关键对比：IDF 的效果 ===
    print("\n--- IDF 权重分析 ---")
    df_en = document_frequency(bow_en)
    idf_en = inverse_document_frequency(df_en, len(docs_en))
    print(f"{'词语':<10} {'DF':>4} {'IDF':>6} {'解读'}")
    print("-" * 50)
    for w, idx in sorted(vocab_en.items(), key=lambda x: x[1]):
        interpretation = (
            "无处不在，权重低" if df_en[idx] == len(docs_en)
            else "稀有词，权重高" if df_en[idx] == 1
            else "中等"
        )
        print(f"{w:<10} {df_en[idx]:>4} {idf_en[idx]:>6.2f} {interpretation}")

    # === 中文演示 ===
    print("\n" + "=" * 60)
    print("中文 TF-IDF（jieba 分词 + 自实现 TF-IDF）")
    print("=" * 60)
    raw_zh = [
        "我喜欢用 Python 做自然语言处理",
        "Python 是一门很好的编程语言",
        "自然语言处理是人工智能的重要方向",
        "我喜欢用 Python 编程",
    ]
    try:
        import jieba
        docs_zh = [tokenize_zh(r) for r in raw_zh]
        vocab_zh = build_vocab(docs_zh)
        bow_zh = bag_of_words(docs_zh, vocab_zh)
        tfidf_zh = l2_normalize(tfidf(bow_zh))
        df_zh = document_frequency(bow_zh)
        idf_zh = inverse_document_frequency(df_zh, len(docs_zh))

        words_zh = sorted(vocab_zh, key=lambda w: vocab_zh[w])
        print(f"词表 ({len(vocab_zh)} 个词): {words_zh}")
        print()

        for i, (raw, vec) in enumerate(zip(raw_zh, tfidf_zh)):
            nonzero = [(words_zh[j], f"{v:.3f}")
                       for j, v in enumerate(vec) if v > 0.01]
            print(f"文档 {i}: \"{raw}\"")
            print(f"  TF-IDF 非零项: {nonzero}")
            print()

        print("IDF 权重分析（中文）:")
        print(f"{'词语':<12} {'DF':>4} {'IDF':>6} {'解读'}")
        print("-" * 50)
        for w, idx in sorted(vocab_zh.items(), key=lambda x: x[1]):
            interpretation = (
                "无处不在" if df_zh[idx] == len(docs_zh)
                else "稀有词" if df_zh[idx] == 1
                else "中等"
            )
            print(f"{w:<12} {df_zh[idx]:>4} {idf_zh[idx]:>6.2f} {interpretation}")

    except ImportError:
        print("jieba 未安装，中文演示跳过。安装：pip install jieba")

    # === scikit-learn 对比 ===
    print("\n" + "=" * 60)
    print("scikit-learn 对比（三行代码完成同样的事情）")
    print("=" * 60)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        tfidf_sklearn = vectorizer.fit_transform(raw_en)
        print(f"特征词: {vectorizer.get_feature_names_out().tolist()}")
        print(f"稀疏矩阵形状: {tfidf_sklearn.shape}")
        print(f"非零元素数: {tfidf_sklearn.nnz}")
        print(f"\nTF-IDF 矩阵 (dense):")
        print(tfidf_sklearn.toarray().round(3))

        # 验证：我们的实现和 scikit-learn 的余弦相似度是否一致？
        print("\n--- 我们的实现 vs scikit-learn 余弦相似度对比 ---")
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
        ours = [cosine_similarity(tfidf_en[i], tfidf_en[j])
                for i in range(4) for j in range(4)]
        theirs = sklearn_cosine(tfidf_sklearn).flatten()
        # 由于实现细节（smooth_idf, 分词差异等），值不会完全相同
        # 但相对大小应该一致
        print(f"我们的 cosine(d0, d1) = {cosine_similarity(tfidf_en[0], tfidf_en[1]):.4f}")
        print(f"sklearn cosine(d0, d1) = {theirs[1]:.4f}")

    except ImportError:
        print("scikit-learn 未安装，跳过。安装：pip install scikit-learn")


if __name__ == "__main__":
    main()
