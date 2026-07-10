# pos_tagger.py — 词性标注：最频标签基线 + Bigram HMM + Viterbi 解码
# 依赖：无（纯标准库实现）
# 安装：无需额外安装
# 对应课程：阶段 05 · 07（词性标注与句法分析）

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set


# ============================================================
# 1. 训练数据——玩具标注集
# ============================================================
# DET=限定词, NOUN=名词, VERB=动词, ADJ=形容词, ADV=副词,
# ADP=介词, AUX=助动词, PRON=代词

TRAIN_DATA = [
    (["The", "cat", "sat", "on", "the", "mat"],
     ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]),
    (["A", "dog", "ran", "across", "the", "road"],
     ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]),
    (["Cats", "chase", "mice"],
     ["NOUN", "VERB", "NOUN"]),
    (["Dogs", "bark", "loudly"],
     ["NOUN", "VERB", "ADV"]),
    (["The", "mat", "is", "red"],
     ["DET", "NOUN", "AUX", "ADJ"]),
    (["A", "red", "cat", "sat"],
     ["DET", "ADJ", "NOUN", "VERB"]),
    (["She", "ran", "quickly"],
     ["PRON", "VERB", "ADV"]),
    (["The", "big", "dog", "ate", "food"],
     ["DET", "ADJ", "NOUN", "VERB", "NOUN"]),
]


# ============================================================
# 2. 最频标签基线（Most Frequent Tag）
# ============================================================

def train_mft(examples: List[Tuple[List[str], List[str]]]):
    """对每个词，记住它在训练数据中最常出现的词性。

    未见过的词 → 回退到全局最高频词性。
    这个基线在 Brown Corpus 上达到 ~85% 准确率——
    是任何严肃模型的地板，不是天花板。
    """
    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    all_tags: Counter = Counter()

    for tokens, tags in examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1

    # 每个词选最高频词性
    word_best = {w: c.most_common(1)[0][0]
                 for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]

    return word_best, default_tag


def predict_mft(tokens: List[str],
                word_best: Dict[str, str],
                default_tag: str) -> List[str]:
    """MFT 预测——看不到的词一律回退默认词性。"""
    return [word_best.get(t.lower(), default_tag) for t in tokens]


# ============================================================
# 3. Bigram HMM + Viterbi 解码
# ============================================================

def train_hmm(examples: List[Tuple[List[str], List[str]]],
              alpha: float = 0.01):
    """训练 Bigram HMM。

    两个概率表：
    - transitions[prev_tag][cur_tag]：P(tag_i | tag_{i-1})
    - emissions[tag][word]：P(word | tag)

    alpha 是加性平滑参数——防止零概率爆炸。
    """
    transitions: Dict[str, Counter] = defaultdict(Counter)
    emissions: Dict[str, Counter] = defaultdict(Counter)
    tags: Set[str] = set()
    vocab: Set[str] = set()

    for tokens, ts in examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def _log_prob(table: Dict[str, Counter],
              given: str,
              key: str,
              smooth_denom: float,
              alpha: float) -> float:
    """对数概率 + 拉普拉斯平滑。"""
    return math.log(
        (table[given].get(key, 0) + alpha) / smooth_denom
    )


def viterbi(tokens: List[str],
            transitions: Dict[str, Counter],
            emissions: Dict[str, Counter],
            tags: Set[str],
            vocab: Set[str],
            alpha: float = 0.01) -> List[str]:
    """Viterbi 算法——找到最高概率的标签序列。

    动态规划：O(n × |T|²)，n 是句子长度，|T| 是标签数。

    核心洞察：第 i 个位置的最优标签序列可以由第 i-1 个位置
    的最优标签序列推出——因为 HMM 的马尔可夫性质保证
    P(tag_i | tag_{i-1}, tag_{i-2}, ...) = P(tag_i | tag_{i-1})。
    """
    tags_list = list(tags)
    n = len(tokens)

    # V[i][j]：以标签 j 结尾的前 i 个词元的最优路径得分
    V = [[0.0] * len(tags_list) for _ in range(n)]
    # back[i][j]：回溯指针——V[i][j] 是从哪个标签 k 转移来的
    back = [[0] * len(tags_list) for _ in range(n)]

    # 初始化：第 0 个词元
    for j, tag in enumerate(tags_list):
        em_denom = (sum(emissions[tag].values())
                    + alpha * (len(vocab) + 1))
        tr_denom = (sum(transitions["<BOS>"].values())
                    + alpha * (len(tags_list) + 1))
        tr = _log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = _log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    # 递推：第 1 到 n-1 个词元
    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = (sum(emissions[tag].values())
                        + alpha * (len(vocab) + 1))
            em = _log_prob(emissions, tag, tokens[i].lower(),
                           em_denom, alpha)

            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = (sum(transitions[prev_tag].values())
                            + alpha * (len(tags_list) + 1))
                tr = _log_prob(transitions, prev_tag, tag,
                               tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    # 回溯——从最后一个词元的最优标签往前推导
    last_best = max(range(len(tags_list)),
                    key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])

    return [tags_list[j] for j in reversed(path)]


# ============================================================
# 4. 评估
# ============================================================

def evaluate(y_true: List[List[str]],
             y_pred: List[List[str]]) -> Dict:
    """词元级准确率 + 每标签精确率/召回率/F1。

    词性标注的评估通常用词元级准确率——与 NER 不同，
    因为标签在每个词元上的分布相对均匀，不存在 O 标签
    占据 85% 的问题。
    """
    correct = 0
    total = 0
    tag_tp: Dict[str, int] = defaultdict(int)
    tag_fp: Dict[str, int] = defaultdict(int)
    tag_fn: Dict[str, int] = defaultdict(int)

    for true_tags, pred_tags in zip(y_true, y_pred):
        for t, p in zip(true_tags, pred_tags):
            total += 1
            if t == p:
                correct += 1
                tag_tp[t] += 1
            else:
                tag_fp[p] += 1
                tag_fn[t] += 1

    accuracy = correct / total if total else 0.0
    all_tags = set(tag_tp) | set(tag_fp) | set(tag_fn)
    per_tag = {}
    for tag in sorted(all_tags):
        tp = tag_tp[tag]
        fp = tag_fp[tag]
        fn = tag_fn[tag]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * p * r / (p + r)) if p + r else 0.0
        per_tag[tag] = {"precision": round(p, 3),
                        "recall": round(r, 3),
                        "f1": round(f1, 3)}

    return {"accuracy": round(accuracy, 3), "per_tag": per_tag}


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === MFT 基线 ===
    word_best, default_tag = train_mft(TRAIN_DATA)
    print("=" * 60)
    print("最频标签基线（MFT）")
    print("=" * 60)
    print(f"已知词数: {len(word_best)}")
    print(f"默认词性: {default_tag}")
    # 展示对几个关键词的"记忆"
    for word in ["the", "cat", "sat", "red"]:
        tag = word_best.get(word, default_tag)
        print(f"  '{word}' → 记为 {tag}")

    # === HMM + Viterbi ===
    transitions, emissions, tags, vocab = train_hmm(TRAIN_DATA)
    print(f"\n{'='*60}")
    print("Bigram HMM + Viterbi")
    print(f"{'='*60}")
    print(f"标签数: {len(tags)}, 词表大小: {len(vocab)}")

    # 展示转移矩阵中最强的模式
    print("\n最强的转移概率:")
    for prev in ["<BOS>", "DET", "ADJ", "NOUN"]:
        if prev in transitions:
            top = transitions[prev].most_common(3)
            print(f"  {prev:6s} → {top}")

    # === 测试 ===
    print(f"\n{'='*60}")
    print("MFT vs HMM 对比")
    print(f"{'='*60}")
    test_sentences = [
        "The cat chased the dog".split(),
        "A red mat is here".split(),
        "Dogs bark".split(),
        "She ran quickly".split(),
        "The big dog ate food".split(),  # 全在训练数据中——验证学习效果
    ]

    y_true_all = []
    y_pred_mft_all = []
    y_pred_hmm_all = []

    for sent in test_sentences:
        mft = predict_mft(sent, word_best, default_tag)
        hmm = viterbi(sent, transitions, emissions, tags, vocab)
        print(f"\n词元:   {sent}")
        print(f"MFT:    {mft}")
        print(f"HMM:    {hmm}")

        # 收集评估数据（用 MFT 作为伪真实标签对比——仅演示用途）
        y_pred_mft_all.append(mft)
        y_pred_hmm_all.append(hmm)

    # === 关键洞察 ===
    print(f"\n{'='*60}")
    print("关键洞察")
    print(f"{'='*60}")
    print("1. MFT 的局限：'red' 可以是 ADJ（a red cat）或 NOUN（the red of sunset）")
    print("   MFT 记住了最频词性，但不知道当前句中 'red' 到底应该是什么")
    print("2. HMM 的改进：转移概率学到 DET→ADJ→NOUN 是高频序列")
    print("   'A red cat' → HMM 利用上下文修正了单靠词频可能标错的位置")
    print("3. HMM 的盲区：'saw' 在 'I saw the movie'（动词）和 'I bought a saw'（名词）中")
    print("   转移概率都是 DET→?，无法区分——这需要 CRF 或 BiLSTM 的全局特征")


if __name__ == "__main__":
    main()
