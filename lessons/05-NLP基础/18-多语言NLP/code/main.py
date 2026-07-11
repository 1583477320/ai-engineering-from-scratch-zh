# 多语言 NLP —— 语言相似度与源语言选择（qWALS 风格）
# 对应课程：阶段 05 · 18

LANGUAGE_FEATURES = {
    "english":  {"word_order": "SVO", "script": "Latin",   "family": "Germanic"},
    "german":   {"word_order": "SVO", "script": "Latin",   "family": "Germanic"},
    "french":   {"word_order": "SVO", "script": "Latin",   "family": "Romance"},
    "spanish":  {"word_order": "SVO", "script": "Latin",   "family": "Romance"},
    "italian":  {"word_order": "SVO", "script": "Latin",   "family": "Romance"},
    "hindi":    {"word_order": "SOV", "script": "Devanagari", "family": "Indic"},
    "marathi":  {"word_order": "SOV", "script": "Devanagari", "family": "Indic"},
    "bengali":  {"word_order": "SOV", "script": "Bengali",    "family": "Indic"},
    "urdu":     {"word_order": "SOV", "script": "Arabic",     "family": "Indic"},
    "arabic":   {"word_order": "VSO", "script": "Arabic",     "family": "Semitic"},
    "japanese": {"word_order": "SOV", "script": "Kanji",      "family": "Japonic"},
    "chinese":  {"word_order": "SVO", "script": "Hanzi",      "family": "Sino-Tibetan"},
}


def similarity(a, b):
    """计算两种语言在三个特征上的相似度——qWALS 的教学简化版。"""
    fa, fb = LANGUAGE_FEATURES[a], LANGUAGE_FEATURES[b]
    matches = sum(1 for k in fa if fa[k] == fb[k])
    return matches / len(fa)


def rank_source_languages(target, candidates):
    """对候选源语言按与目标语言的相似度降序排列。"""
    scored = [(c, similarity(target, c)) for c in candidates if c != target]
    scored.sort(key=lambda x: -x[1])
    return scored


def simulate_transfer_accuracy(target, source):
    """模拟跨语言迁移准确率。基础 45% + 相似度 × 最大增益 45%。"""
    return min(0.95, 0.45 + similarity(target, source) * 0.45)


def main():
    candidates = list(LANGUAGE_FEATURES)
    targets = ["marathi", "urdu", "arabic", "japanese", "chinese"]

    print("=== 源语言选择（qWALS 风格相似度）===")
    for target in targets:
        ranking = rank_source_languages(target, candidates)[:4]
        print(f"\n  目标语言: {target}")
        for source, sim in ranking:
            acc = simulate_transfer_accuracy(target, source)
            best = "← 最佳" if sim == ranking[0][1] else ""
            print(f"    源={source:10s}  相似度={sim:.2f}  模拟准确率={acc:.0%}  {best}")

    print("\n关键洞察：对 Marathi，Hindi 是比 English 更好的源语言。")
    print("对中文，English(SVO)也比 Japanese(SOV)更适合——语序比文字系统更重要。")
    print("注意：真实相似度来自 qWALS/lang2vec——不是三个特征的玩具。")


if __name__ == "__main__":
    main()
