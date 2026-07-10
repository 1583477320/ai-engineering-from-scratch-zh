# sentiment.py — 情感分析从零：朴素贝叶斯 + 否定处理 + 评估
# 依赖：无（纯标准库实现）
# 安装：无需额外安装
# 对应课程：阶段 05 · 05（情感分析）

import math
import re
from collections import Counter
from typing import List, Dict, Tuple


# ============================================================
# 1. 分词 + 否定标记
# ============================================================

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,;]")

# 英文否定词
EN_NEGATION = {"not", "no", "never", "nor", "none", "nothing",
               "neither", "n't", "hardly", "scarcely"}
EN_NEGATION_END = {".", "!", "?", ",", ";", "but"}

# 中文否定词
ZH_NEGATION = {"不", "没", "无", "非", "别", "未", "莫", "勿",
               "休", "毫不", "从不", "绝不", "决不", "并不"}
ZH_NEGATION_END = {"。", "！", "？", "，", "；", "但", "但是",
                   "然而", "可是", "不过", "却"}


def tokenize_en(text: str) -> List[str]:
    """英文分词——小写 + 保留标点。"""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def apply_negation_en(tokens: List[str]) -> List[str]:
    """英文否定范围标记。

    否定词之后的所有词（直到遇到终止标点）前加 'NOT_' 前缀。
    'not good at all . but funny' →
    ['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']

    这样 'good' 和 'NOT_good' 在 BoW 中是完全不同的特征——
    分类器可以给它们分配相反的权重。
    """
    out = []
    negate = False
    for token in tokens:
        if token in EN_NEGATION_END:
            negate = False
            out.append(token)
            continue
        if token in EN_NEGATION:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out


def tokenize_zh(text: str) -> List[str]:
    """中文分词——使用 jieba，未安装时回退逐字切分。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return list(text)


def apply_negation_zh(tokens: List[str]) -> List[str]:
    """中文否定范围标记。

    中文的否定表达与英文不同——否定词出现在被否定词之前，
    且否定范围通常延续到句子结束或转折词出现。
    '不 好看 但 便宜' → ['不', 'NOT_好看', '但', '便宜']
    """
    out = []
    negate = False
    for token in tokens:
        if token in ZH_NEGATION_END:
            negate = False
            out.append(token)
            continue
        if token in ZH_NEGATION:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out


# ============================================================
# 2. 朴素贝叶斯（Multinomial Naive Bayes）
# ============================================================

def build_vocab(docs: List[List[str]]) -> set:
    """收集所有出现过的词——不重复。"""
    vocab = set()
    for doc in docs:
        for t in doc:
            vocab.add(t)
    return vocab


def train_nb(docs_by_class: Dict[str, List[List[str]]],
             vocab: set,
             alpha: float = 1.0):
    """训练多项式朴素贝叶斯。

    alpha 是拉普拉斯平滑参数：
    - alpha=1.0：经典拉普拉斯平滑（教学默认值）
    - alpha=0.01：生产常用值——更少的平滑，更依赖实际数据

    没有平滑 → 一个词在某类中从未出现 → P(word|class)=0
    → log(0) = -inf → 整个文档得分爆炸
    """
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(docs) for docs in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        # 先验概率：P(class) = 该类文档数 / 总文档数
        class_priors[cls] = len(docs) / total_docs
        # 统计每个词在该类中的出现次数
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        # 条件概率：P(word|class) = (count + alpha) / (total + alpha*|V|)
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }

    return class_priors, class_word_probs


def predict_nb(doc: List[str],
               class_priors: Dict[str, float],
               class_word_probs: Dict[str, Dict[str, float]]) -> str:
    """朴素贝叶斯预测——选择 log 概率最大的类别。

    为什么取 log？
    - 概率连乘 → log 后变成加法
    - 每个 P(word|class) 都很小（< 0.01），连乘 50 个 = 下溢出
    - log 后数值稳定，单调性不变 → 得分最高的类不变
    """
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)


# ============================================================
# 3. 评估指标
# ============================================================

def evaluate(y_true: List[str],
             y_pred: List[str],
             pos_label: str = "+",
             neg_label: str = "-") -> Dict:
    """计算混淆矩阵 + 精确率 + 召回率 + F1。

    为什么报告宏平均 F1 而非准确率？
    - 情感数据通常不平衡（80% 正面、20% 负面）
    - '全部预测为正面'也能获得 80% 准确率——但模型什么都没学到
    - 宏平均 F1 给每个类同等权重，少数类不会被多数类淹没
    """
    tp = sum(1 for t, p in zip(y_true, y_pred)
             if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred)
             if t == neg_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred)
             if t == pos_label and p == neg_label)
    tn = sum(1 for t, p in zip(y_true, y_pred)
             if t == neg_label and p == neg_label)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    # 负类的指标（用于宏平均）
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)
              if precision_neg + recall_neg else 0.0)

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    macro_f1 = (f1 + f1_neg) / 2

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 3),
        "precision_+": round(precision, 3),
        "recall_+": round(recall, 3),
        "f1_+": round(f1, 3),
        "precision_-": round(precision_neg, 3),
        "recall_-": round(recall_neg, 3),
        "f1_-": round(f1_neg, 3),
        "macro_f1": round(macro_f1, 3),
    }


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === 数据准备 ===
    positive_en = [
        "absolutely loved this movie",
        "beautiful cinematography and a great story",
        "one of the best films of the year",
        "brilliant acting from the lead",
        "heartwarming and funny",
    ]
    negative_en = [
        "boring and far too long",
        "not worth your time",
        "the plot made no sense",
        "terrible acting, awful script",
        "i want my two hours back",
    ]

    # 预处理：分词 + 否定标记
    train_pos = [apply_negation_en(tokenize_en(t)) for t in positive_en]
    train_neg = [apply_negation_en(tokenize_en(t)) for t in negative_en]

    # 训练
    vocab = build_vocab(train_pos + train_neg)
    docs_by_class = {"+": train_pos, "-": train_neg}
    priors, word_probs = train_nb(docs_by_class, vocab)

    # 测试——刻意包含否定结构
    test_en = [
        ("this movie was not good", "-"),
        ("loved every minute of it", "+"),
        ("terrible waste of time", "-"),
        ("beautiful and absolutely brilliant", "+"),
        ("i did not enjoy this at all", "-"),
        ("not a bad film actually", "+"),  # 双重否定 = 正面
    ]

    print("=" * 60)
    print("英文情感分析——朴素贝叶斯 + 否定标记")
    print("=" * 60)
    y_true = [label for _, label in test_en]
    y_pred = []
    for text, actual in test_en:
        tokens = apply_negation_en(tokenize_en(text))
        pred = predict_nb(tokens, priors, word_probs)
        y_pred.append(pred)
        mark = "✓" if actual == pred else "✗"
        print(f"[{mark}] 预测={pred} 真实={actual}  |  {text}")
        if mark == "✗":
            print(f"     分词: {tokens}")

    metrics = evaluate(y_true, y_pred)
    print(f"\n评估指标: 准确率={metrics['accuracy']}, "
          f"宏平均F1={metrics['macro_f1']}")
    print(f"  正面: P={metrics['precision_+']}, R={metrics['recall_+']}, "
          f"F1={metrics['f1_+']}")
    print(f"  负面: P={metrics['precision_-']}, R={metrics['recall_-']}, "
          f"F1={metrics['f1_-']}")

    # 展示否定标记的效果
    print("\n--- 否定标记示例 ---")
    for text in ["not good at all", "never a dull moment",
                 "no one would enjoy this"]:
        tokens = tokenize_en(text)
        negated = apply_negation_en(tokens)
        print(f"  '{text}'")
        print(f"    原始: {tokens}")
        print(f"    标记: {negated}")
        print()

    # === 中文演示 ===
    print("=" * 60)
    print("中文情感分析——朴素贝叶斯 + 中文否定处理")
    print("=" * 60)

    positive_zh = [
        "这部电影太精彩了",
        "剧情紧凑演员演技在线",
        "今年看过最好的电影没有之一",
        "画面美轮美奂音乐也很棒",
        "温馨感人笑中带泪",
    ]
    negative_zh = [
        "无聊至极浪费时间",
        "表演生硬剧本一塌糊涂",
        "完全不值得票价",
        "剧情毫无逻辑可言",
        "看完想把这两小时要回来",
    ]

    try:
        import jieba

        train_pos_zh = [apply_negation_zh(tokenize_zh(t))
                        for t in positive_zh]
        train_neg_zh = [apply_negation_zh(tokenize_zh(t))
                        for t in negative_zh]

        vocab_zh = build_vocab(train_pos_zh + train_neg_zh)
        priors_zh, word_probs_zh = train_nb(
            {"+": train_pos_zh, "-": train_neg_zh}, vocab_zh
        )

        test_zh = [
            ("这个电影一点都不好看", "-"),
            ("每一分钟都喜欢", "+"),
            ("浪费我的时间", "-"),
            ("画面很美配乐好听", "+"),
            ("我觉得不怎么样", "-"),
            ("其实还不错啦", "+"),
        ]

        for text, actual in test_zh:
            tokens = apply_negation_zh(tokenize_zh(text))
            pred = predict_nb(tokens, priors_zh, word_probs_zh)
            mark = "✓" if actual == pred else "✗"
            print(f"[{mark}] 预测={pred} 真实={actual}  |  {text}")
            if mark == "✗":
                print(f"     分词: {'/'.join(tokens)}")

        # 中文否定标记示例
        print("\n--- 中文否定标记示例 ---")
        for text in ["一点都不好看", "没有我想象的好", "毫无新意可言"]:
            tokens = tokenize_zh(text)
            negated = apply_negation_zh(tokens)
            print(f"  '{text}'")
            print(f"    原始: {'/'.join(tokens)}")
            print(f"    标记: {'/'.join(negated)}")
            print()

    except ImportError:
        print("jieba 未安装，中文演示跳过。安装：pip install jieba")

    print("完成！完整实现见 code/sentiment.py")


if __name__ == "__main__":
    main()
