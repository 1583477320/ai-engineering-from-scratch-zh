# word2vec.py — 从零实现 Skip-gram + 负采样
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 05 · 03（词嵌入 Word2Vec）

import re
import numpy as np
from typing import List, Dict, Tuple, Set

# ============================================================
# 1. 简易分词器
# ============================================================

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    """简易英文分词——小写化 + 正则匹配。"""
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ============================================================
# 2. 构建词表
# ============================================================

def build_vocab(docs: List[List[str]]) -> Dict[str, int]:
    """同阶段 05 · 02——每个不重复词分配唯一索引。"""
    vocab: Dict[str, int] = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


# ============================================================
# 3. Skip-gram 训练对生成
# ============================================================

def skipgram_pairs(docs: List[List[str]], window: int = 2) -> List[Tuple[str, str]]:
    """从文档生成 (中心词, 上下文词) 训练对。

    窗口大小 window=2 时，每个中心词与其前后各 2 个上下文词配对。
    例："the cat sat on mat"，中心词 "sat" 产生：
       (sat, the), (sat, cat), (sat, on), (sat, mat)
    """
    pairs: List[Tuple[str, str]] = []
    for doc in docs:
        for i, center in enumerate(doc):
            left = max(0, i - window)
            right = min(len(doc), i + window + 1)
            for j in range(left, right):
                if i != j:
                    pairs.append((center, doc[j]))
    return pairs


# ============================================================
# 4. Sigmoid
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活——将任意实数映射到 (0, 1)。

    裁剪到 [-20, 20] 防止 exp 溢出。
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# 5. 嵌入表初始化
# ============================================================

def init_embeddings(vocab_size: int, dim: int, seed: int = 0):
    """初始化两个嵌入表。

    W  ——中心词嵌入表（训练后保留的就是这个）
    W' ——上下文词嵌入表（训练后通常丢弃，有时与 W 平均）

    用小随机数初始化——太大则梯度饱和，太小则学习缓慢。
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime


# ============================================================
# 6. 负采样训练（单对）
# ============================================================

def train_pair(W: np.ndarray,
               W_prime: np.ndarray,
               c_idx: int,
               ctx_idx: int,
               neg_indices: List[int],
               lr: float) -> None:
    """对一对 (中心词, 上下文词) 做一步梯度更新。

    核心思想：
    - 正样本：让 W[center]·W'[context] 的 sigmoid 接近 1
    - 负样本：让 W[center]·W'[random] 的 sigmoid 接近 0

    这等价于二元逻辑回归——将"这两个词是否共现"作为分类目标。
    相比全词表 Softmax（10 万类），负采样只做 k+1 次二分类（k 通常为 5-20）。
    """
    v_c = W[c_idx]                     # 中心词向量 (dim,)
    u_pos = W_prime[ctx_idx]           # 正样本上下文向量 (dim,)
    u_negs = W_prime[neg_indices]      # 负样本上下文向量 (k_neg, dim)

    # 正样本误差：我们希望 sigmoid 接近 1
    pos_err = sigmoid(v_c @ u_pos) - 1.0
    # 负样本误差：我们希望 sigmoid 接近 0
    neg_errs = sigmoid(u_negs @ v_c)

    # 梯度回传：中心词向量同时接收正负样本的梯度
    grad_center = pos_err * u_pos + neg_errs @ u_negs

    # 更新参数
    W_prime[ctx_idx] -= lr * pos_err * v_c
    for i, neg_idx in enumerate(neg_indices):
        W_prime[neg_idx] -= lr * neg_errs[i] * v_c
    W[c_idx] -= lr * grad_center


# ============================================================
# 7. 完整训练循环
# ============================================================

def train(docs: List[List[str]],
          dim: int = 16,
          window: int = 2,
          k_neg: int = 5,
          epochs: int = 200,
          lr: float = 0.05,
          seed: int = 0) -> Tuple[Dict[str, int], np.ndarray]:
    """训练 Skip-gram Word2Vec 模型。

    Args:
        docs: 分词后的文档列表
        dim: 嵌入维度（教学用 16，生产用 100-300）
        window: 上下文窗口大小
        k_neg: 每个正样本配多少个负样本
        epochs: 全数据迭代次数
        lr: 学习率
        seed: 随机种子

    Returns:
        (vocab, W): 词表和中心词嵌入表
    """
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    W, W_prime = init_embeddings(vocab_size, dim, seed)
    pairs = skipgram_pairs(docs, window=window)
    rng = np.random.default_rng(seed)

    print(f"词表大小: {vocab_size}, 训练对: {len(pairs)}, "
          f"嵌入维度: {dim}, 负采样数: {k_neg}, 轮次: {epochs}")

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]

            # 随机采样 k_neg 个负样本（排除正样本和中心词本身）
            candidates = rng.integers(0, vocab_size, size=k_neg * 2)
            negs = [int(n) for n in candidates
                    if n != ctx_idx and n != c_idx][:k_neg]

            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)

    return vocab, W


# ============================================================
# 8. 最近邻查询
# ============================================================

def nearest(vocab: Dict[str, int],
            W: np.ndarray,
            target_vec: np.ndarray,
            topk: int = 5,
            exclude: Set[int] = None) -> List[Tuple[str, float]]:
    """查找与目标向量余弦相似度最高的 topk 个词。

    Args:
        vocab: 词表
        W: 嵌入矩阵
        target_vec: 目标向量（可以是已有嵌入，也可以是计算出来的向量）
        topk: 返回前几个
        exclude: 排除的索引集合（防止返回查询词本身）
    """
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}

    # L2 归一化所有嵌入
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)

    # 点积 = 余弦相似度（因为都已归一化）
    sims = W_norm @ target
    order = np.argsort(-sims)

    out = []
    for idx in order:
        i = int(idx)
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


# ============================================================
# 9. 类比推理
# ============================================================

def analogy(vocab: Dict[str, int],
            W: np.ndarray,
            a: str, b: str, c: str,
            topk: int = 5) -> List[Tuple[str, float]]:
    """经典类比：a : b :: c : ?

    计算公式：v = W[b] - W[a] + W[c]
    直觉：(b - a) 捕获了 a 到 b 的"关系方向"，
    将此方向加到 c 上，应得到与 b 同类的词。

    例如：king - man + woman ≈ queen
         巴黎 - 法国 + 日本 ≈ 东京
    """
    vec = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, vec, topk=topk,
                   exclude={vocab[a], vocab[b], vocab[c]})


# ============================================================
# 演示主程序
# ============================================================

def main():
    # 构建玩具语料——故意让猫/狗相关词交替出现
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "a cat chased a mouse",
        "a dog chased a cat",
        "the kitten slept on the mat",
        "the puppy slept on the rug",
        "cats and dogs are pets",
        "kittens and puppies are young",
        "cats chase mice",
        "dogs chase squirrels",
        "the cat ate food",
        "the dog ate food",
        "a kitten played with yarn",
        "a puppy played with a ball",
        "the cat is fluffy",
        "the dog is friendly",
        "kittens are cute",
        "puppies are energetic",
    ] * 15  # 重复以增加训练量

    docs = [tokenize(s) for s in corpus]
    vocab, W = train(docs, dim=16, window=2, k_neg=5, epochs=150,
                     lr=0.05, seed=42)

    print("\n" + "=" * 60)
    print("最近邻查询——验证语义聚类")
    print("=" * 60)
    for word in ["cat", "dog", "kitten", "puppy", "chased", "sat"]:
        idx = vocab[word]
        top = nearest(vocab, W, W[idx], topk=4, exclude={idx})
        print(f"{word}:")
        for w, s in top:
            print(f"  {w:12s} {s:.3f}")
        print()

    # 验证：cat 的最近邻应该包含 kitten 或 dog
    cat_idx = vocab["cat"]
    cat_neighbors = [w for w, _ in nearest(vocab, W, W[cat_idx],
                                           topk=10, exclude={cat_idx})]
    animal_words = {"kitten", "dog", "puppy", "cat"}
    found = animal_words & set(cat_neighbors)
    print(f"cat 的邻居中包含动物词: {found}")

    # 类比测试（玩具语料上效果可能不明显，展示方法）
    print("\n" + "=" * 60)
    print("类比推理——king - man + woman ≈ ?")
    print("（玩具语料上精度有限，仅展示方法）")
    print("=" * 60)
    if "king" in vocab and "man" in vocab and "woman" in vocab:
        result = analogy(vocab, W, "man", "king", "woman", topk=5)
        for w, s in result:
            print(f"  {w:12s} {s:.3f}")

    # 中文演示：如果 jieba 可用
    print("\n" + "=" * 60)
    print("中文词向量演示")
    print("=" * 60)
    try:
        import jieba

        zh_corpus = [
            "我喜欢学习机器学习",
            "深度学习是人工智能的重要方向",
            "机器学习需要大量的数据",
            "人工智能正在改变世界",
            "神经网络是深度学习的基础",
            "我喜欢人工智能这个领域",
            "数据和算力推动了深度学习的发展",
            "机器学习算法需要好的特征",
        ] * 30

        zh_docs = [list(jieba.cut(s)) for s in zh_corpus]
        zh_vocab, zh_W = train(zh_docs, dim=8, window=2, k_neg=3,
                               epochs=100, lr=0.1, seed=42)

        for word in ["机器学习", "深度学习", "人工智能"]:
            if word in zh_vocab:
                idx = zh_vocab[word]
                top = nearest(zh_vocab, zh_W, zh_W[idx],
                             topk=4, exclude={idx})
                print(f"{word}:")
                for w, s in top:
                    print(f"  {w:12s} {s:.3f}")
                print()
    except ImportError:
        print("jieba 未安装，中文演示跳过。安装：pip install jieba")

    print("\n完成！完整的从零实现参见 code/word2vec.py")


if __name__ == "__main__":
    main()
