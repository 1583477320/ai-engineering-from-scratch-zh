# ner.py — 命名实体识别从零：BIO标注 + 词典 + 手工特征
# 依赖：无（纯标准库实现）
# 安装：无需额外安装（spaCy/HuggingFace 部分需要 pip install）
# 对应课程：阶段 05 · 06（命名实体识别）

from typing import List, Tuple, Dict, Set

# ============================================================
# 1. 英文实体词典（Gazetteer）
# ============================================================

ORG_GAZETTEER: Set[str] = {
    "Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon",
    "Netflix", "Anthropic", "Tesla", "NVIDIA", "IBM", "Intel",
}
GPE_GAZETTEER: Set[str] = {
    "US", "USA", "UK", "India", "Germany", "France", "Japan",
    "China", "Canada", "Brazil", "Korea", "Australia",
}
PRODUCT_GAZETTEER: Set[str] = {
    "iPhone", "Android", "Windows", "ChatGPT", "Claude", "Gemini",
    "iPad", "MacBook", "Surface",
}

# 中文实体词典
ZH_ORG: Set[str] = {
    "苹果", "谷歌", "微软", "华为", "阿里巴巴", "腾讯", "百度",
    "字节跳动", "美团", "京东", "小米", "比亚迪",
}
ZH_GPE: Set[str] = {
    "北京", "上海", "深圳", "杭州", "美国", "中国", "日本",
    "英国", "德国", "法国", "硅谷", "粤港澳大湾区",
}
ZH_PERSON: Set[str] = {
    "马云", "马化腾", "李彦宏", "任正非", "雷军", "张一鸣",
    "黄仁勋", "Sam Altman", "Elon Musk",
}


# ============================================================
# 2. BIO 标注转换
# ============================================================

def spans_to_bio(tokens: List[str],
                 spans: List[Tuple[int, int, str]]) -> List[str]:
    """将 (start, end, type) 范围转换为 BIO 标签序列。

    例：tokens=["New","York","City","mayor"]
        spans=[(0, 3, "GPE")]
        → ["B-GPE", "I-GPE", "I-GPE", "O"]
    """
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens: List[str],
                 labels: List[str]) -> List[Tuple[int, int, str]]:
    """将 BIO 标签序列还原为实体范围。

    这是 NER 评估的关键——实体级 F1 要求预测的 start/end/type
    与真实标注完全一致。
    """
    spans: List[Tuple[int, int, str]] = []
    current: Tuple[int, int, str] | None = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current is not None:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current is not None \
                and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current is not None:
                spans.append(current)
                current = None

    if current is not None:
        spans.append(current)
    return spans


# ============================================================
# 3. 手工特征（CRF/BiLSTM 之前的标准做法）
# ============================================================

def word_shape(word: str) -> str:
    """词形特征——大小写结构是专有名词的强信号。

    "iPhone"  → "xXxxxx"
    "USA"    → "XXX"
    "IBM"    → "XXX"
    "2024Q3" → "ddddXd"
    """
    out: List[str] = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def token_features(token: str,
                   prev_token: str = "",
                   next_token: str = "") -> Dict[str, object]:
    """为一元 token 构造手工特征字典。

    这些特征在 CRF 时代是核心武器。每个特征都是一条线索——
    首字母大写 → 可能是专有名词；词形 XXX → 可能是缩写；
    前一个词是"Inc." → 当前词很可能是公司名。
    """
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower() if len(token) >= 3 else token.lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


# ============================================================
# 4. 基于词典的 NER（Rule-based + Gazetteer）
# ============================================================

def rule_based_ner(tokens: List[str]) -> List[str]:
    """基于实体词典的规则 NER。

    这是最简单也最脆弱的方案。高精确率（查到的都对）、零召回率
    （查不到的全漏）。最大的问题：无法消歧——"Apple"是公司还是水果？
    词典不知道。
    """
    labels: List[str] = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels


# ============================================================
# 5. 实体级评估
# ============================================================

def entity_f1(y_true: List[Tuple[int, int, str]],
              y_pred: List[Tuple[int, int, str]]) -> Dict:
    """计算实体级精确率、召回率、F1。

    为什么不是 token 级 F1？
    "New York City" → 预测为 "New York" → token 级给了 2/3 的分数
    → 实体级直接判定为错误（span 不完全匹配）。
    实体级 F1 才是你关心的指标——你需要的不是"大致位置"，而是完整的实体名。
    """
    true_set = set(y_true)
    pred_set = set(y_pred)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === BIO 标注 + 词典演示 ===
    print("=" * 60)
    print("英文 NER——基于词典 + BIO 标注")
    print("=" * 60)
    sentence = "Apple sued Google over iPhone sales in the US .".split()
    labels = rule_based_ner(sentence)
    spans = bio_to_spans(sentence, labels)

    print(f"词元:   {sentence}")
    print(f"BIO标签: {labels}")
    print("实体:")
    for start, end, kind in spans:
        entity = " ".join(sentence[start:end])
        print(f"  [{start}:{end}] {kind:8s} \"{entity}\"")

    # === 词形特征演示 ===
    print("\n--- 词形特征（大小写 = 专有名词的强信号）---")
    for tok in ["Apple", "iPhone", "IBM", "USA-2024", "apple", "NVIDIA"]:
        print(f"  {tok:12s} → shape: {word_shape(tok)}")

    # === 特征字典示例 ===
    print("\n--- 手工特征字典（CRF 时代的核心武器）---")
    tokens = "Apple CEO Tim Cook visited the EU .".split()
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        feats = token_features(tok, prev, nxt)
        print(f"  {tok:8s}: {feats}")

    # === BIO 往返一致性验证 ===
    print("\n--- BIO 往返（spans → BIO → spans）---")
    tokens2 = "The New York City mayor visited OpenAI .".split()
    gold_spans: List[Tuple[int, int, str]] = [
        (1, 4, "GPE"),    # New York City
        (6, 7, "ORG"),    # OpenAI
    ]
    bio = spans_to_bio(tokens2, gold_spans)
    recovered = bio_to_spans(tokens2, bio)
    print(f"  词元:      {tokens2}")
    print(f"  BIO标签:   {bio}")
    print(f"  原始实体:  {gold_spans}")
    print(f"  恢复实体:  {recovered}")
    print(f"  往返一致:  {gold_spans == recovered}")

    # === 实体级评估演示 ===
    print("\n--- 实体级 F1 vs Token 级 F1 ---")
    true_spans: List[Tuple[int, int, str]] = [
        (0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "PRODUCT")
    ]
    # 模拟：模型漏掉了 PRODUCT，把第二个 ORG 标成了 MISC
    pred_spans: List[Tuple[int, int, str]] = [
        (0, 1, "ORG"), (2, 3, "MISC")
    ]
    metrics = entity_f1(true_spans, pred_spans)
    print(f"  真实: {true_spans}")
    print(f"  预测: {pred_spans}")
    print(f"  实体级 F1: {metrics}")
    print(f"  注意: 只有 (0,1,ORG) 是精确匹配——2 个错误中 1 个是类型错、1 个完全漏掉")

    # === 中文 NER 演示 ===
    print("\n" + "=" * 60)
    print("中文 NER——词典匹配 + 挑战说明")
    print("=" * 60)
    zh_sentence = "百度 CEO 李彦宏在北京发布了新产品"
    zh_tokens = list(zh_sentence)  # 逐字（未分词）

    # 简单词典标注（演示用——生产环境需要先分词再做序列标注）
    zh_labels: List[str] = []
    for ch in zh_tokens:
        if ch == "李" and zh_tokens[zh_labels.count("B-PER") == 0:]:
            pass  # 这只是一个粗糙演示
        # 逐字标注无法处理多字实体，仅用于展示
        zh_labels.append("O")

    print(f"  原文: {zh_sentence}")
    print("  中文 NER 的三大挑战:")
    print("    1. 没有大小写信号——'苹果'是公司还是水果？纯靠文本本身")
    print("    2. 需要先分词再做序列标注——分词错误 = NER 错误")
    print("    3. 中文人名/地名没有形式标志——不能靠首字母大写判断")

    try:
        import jieba
        zh_words = list(jieba.cut(zh_sentence))
        print(f"\n  jieba 分词: {'/'.join(zh_words)}")
        # 词典标注
        zh_word_labels: List[str] = []
        for w in zh_words:
            assigned = False
            if w in ZH_ORG:
                zh_word_labels.append("B-ORG")
                assigned = True
            elif w in ZH_GPE:
                zh_word_labels.append("B-GPE")
                assigned = True
            elif w in ZH_PERSON:
                zh_word_labels.append("B-PER")
                assigned = True
            if not assigned:
                zh_word_labels.append("O")
        print(f"  词典标注: {dict(zip(zh_words, zh_word_labels))}")
    except ImportError:
        print("  jieba 未安装，跳过中文演示")

    print("\n完成！完整实现见 code/ner.py")


if __name__ == "__main__":
    main()
