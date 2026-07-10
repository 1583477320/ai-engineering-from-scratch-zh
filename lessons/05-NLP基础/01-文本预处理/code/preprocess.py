# preprocess.py — 英文与中文文本预处理流水线
# 依赖：jieba>=0.42
# 安装：pip install jieba
# 对应课程：阶段 05 · 01（文本预处理）

import re
from typing import List, Dict, Tuple, Optional, Callable


# ============================================================
# 1. 英文正则分词器
# ============================================================

# 三条规则，按优先级匹配：
# (1) 带内部撇号的单词（don't, it's）
# (2) 纯数字
# (3) 单个标点符号
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")


def tokenize_en(text: str) -> List[str]:
    """英文正则分词——用最少代码覆盖最常见场景。"""
    return WORD_RE.findall(text)


# ============================================================
# 2. 中文分词——正向/逆向最大匹配
# ============================================================

# 演示用词典（实际使用中至少需要数万条）
CHINESE_DICT = {
    "我们", "喜欢", "学习", "自然语言", "自然", "语言", "处理",
    "人工智能", "人工", "智能", "北京", "大学", "北京大学",
    "中国", "中国人", "人民", "中华人民共和国",
    "机器", "机器学习", "深度", "深度学习", "模型",
    "分词", "词元", "文本", "预处理",
    # 歧义演示用
    "研究", "研究生", "生命", "命",
}


def forward_max_match(text: str, dictionary: set) -> List[str]:
    """正向最大匹配（FMM）——从左到右，每次取最长词。

    这是中文分词最基础的算法。速度 O(n·max_len)，简单但不一定最优。
    歧义处理：当"北京大学"和"北京"/"大学"并存时，FMM 选"北京大学"。
    """
    max_len = max(len(w) for w in dictionary) if dictionary else 1
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        # 从最大长度开始尝试匹配
        for window in range(max_len, 0, -1):
            candidate = text[i:i + window]
            if candidate in dictionary:
                tokens.append(candidate)
                i += window
                matched = True
                break
        if not matched:
            # 单字作为独立词元
            tokens.append(text[i])
            i += 1
    return tokens


def reverse_max_match(text: str, dictionary: set) -> List[str]:
    """逆向最大匹配（BMM）——从右到左，每次取最长词。

    中文分词中，BMM 准确率通常略高于 FMM。
    原因：中文的偏正结构（修饰语在前，中心语在后），从后往前切更有优势。
    例如"研究生命"→ FMM 可能切成"研究生/命"，BMM 切为"研究/生命"。
    """
    max_len = max(len(w) for w in dictionary) if dictionary else 1
    tokens = []
    i = len(text)
    while i > 0:
        matched = False
        for window in range(max_len, 0, -1):
            start = i - window
            if start < 0:
                continue
            candidate = text[start:i]
            if candidate in dictionary:
                tokens.insert(0, candidate)
                i = start
                matched = True
                break
        if not matched:
            tokens.insert(0, text[i - 1])
            i -= 1
    return tokens


# ============================================================
# 3. Porter 词干提取器——Step 1a
# ============================================================

def stem_step_1a(word: str) -> str:
    """Porter 词干提取算法 Step 1a。

    完整的 Porter 算法有 5 个阶段。Step 1a 覆盖了英文最常见的后缀规则。
    注意：中文没有屈折形态变化，不需要词干提取——这是英文 NLP 特有的步骤。
    """
    if word.endswith("sses"):
        return word[:-2]       # caresses → caress
    if word.endswith("ies"):
        return word[:-2]       # ponies → poni（Step 1b 会进一步修正）
    if word.endswith("ss"):
        return word            # caress → caress（已经是词干）
    if word.endswith("s") and len(word) > 1:
        return word[:-1]       # cats → cat
    return word


# ============================================================
# 4. 基于查表的词形还原器
# ============================================================

LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
    ("am", "VERB"): "be",
    ("are", "VERB"): "be",
}


def lemmatize(word: str, pos: str = "NOUN") -> str:
    """词形还原——将词的变体还原为词典形式。

    需要词性标注（POS Tag）才能正确处理。
    查表优先，规则回退。规则只能覆盖有规律的变化（如 -ing, -s）。
    不规则形式（went → go, better → good）必须依赖查表。
    """
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    # 规则回退
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()


# ============================================================
# 5. 演示用词性标注器
# ============================================================

def demo_pos_tagger(tokens: List[str]) -> List[Tuple[str, str]]:
    """简易词性标注——仅用于演示。生产环境请使用 spaCy 或 NLTK。"""
    verbs = {"running", "ran", "runs", "were", "was", "is", "are", "am",
             "watched", "played", "studied"}
    adjs = {"better", "best", "good", "bad", "worse", "worst"}
    result = []
    for t in tokens:
        low = t.lower()
        if low in verbs:
            result.append((t, "VERB"))
        elif low in adjs:
            result.append((t, "ADJ"))
        else:
            result.append((t, "NOUN"))
    return result


# ============================================================
# 6. 预处理流水线
# ============================================================

def preprocess_en(text: str,
                  pos_tagger: Optional[Callable] = None) -> Dict[str, List[str]]:
    """英文预处理流水线：分词 → 词干提取 → 词形还原。"""
    tokens = tokenize_en(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}


def preprocess_zh(text: str,
                  dictionary: Optional[set] = None) -> Dict[str, List[str]]:
    """中文预处理流水线：分词（正向/逆向最大匹配对比）。

    中文不需要词干提取和词形还原——没有屈折形态变化。
    核心挑战在于分词歧义消解——这是 NLTK/spaCy 不擅长、但 jieba/HanLP 专攻的问题。
    """
    dic = dictionary if dictionary is not None else CHINESE_DICT
    fmm_tokens = forward_max_match(text, dic)
    bmm_tokens = reverse_max_match(text, dic)
    return {"fmm_tokens": fmm_tokens, "bmm_tokens": bmm_tokens}


# ============================================================
# 演示主程序
# ============================================================

def main():
    # === 英文演示 ===
    en_text = "The cats were running at 3pm."
    result_en = preprocess_en(en_text, pos_tagger=demo_pos_tagger)
    print("=== 英文预处理 ===")
    print(f"输入:   {en_text}")
    print(f"分词:   {result_en['tokens']}")
    print(f"词干:   {result_en['stems']}")
    print(f"词形还原: {result_en['lemmas']}")

    # 对比：词干 vs 词形还原
    print("\n词干提取 vs 词形还原：")
    word = "running"
    print(f"  '{word}' → 词干: '{stem_step_1a(word)}', 词形还原: '{lemmatize(word, 'VERB')}'")
    word = "better"
    print(f"  '{word}' → 词干: '{stem_step_1a(word)}', 词形还原: '{lemmatize(word, 'ADJ')}'")

    # === 中文演示 ===
    print("\n=== 中文分词 ===")
    zh_text = "我们喜欢学习自然语言处理"
    result_zh = preprocess_zh(zh_text)
    print(f"输入:         {zh_text}")
    print(f"正向最大匹配: {'/'.join(result_zh['fmm_tokens'])}")
    print(f"逆向最大匹配: {'/'.join(result_zh['bmm_tokens'])}")

    # 歧义演示
    print("\n=== 分词歧义 ===")
    ambiguous = "研究生命"
    result_amb = preprocess_zh(ambiguous)
    print(f"输入:         {ambiguous}")
    print(f"正向最大匹配: {'/'.join(result_amb['fmm_tokens'])}  ← 可能错误")
    print(f"逆向最大匹配: {'/'.join(result_amb['bmm_tokens'])}  ← 更合理")

    # === jieba 分词（工业工具） ===
    print("\n=== jieba 分词对比 ===")
    try:
        import jieba
        jieba_tokens = list(jieba.cut(zh_text))
        print(f"jieba 分词: {'/'.join(jieba_tokens)}")
        jieba_amb = list(jieba.cut(ambiguous))
        print(f"歧义处理:   {'/'.join(jieba_amb)}")
    except ImportError:
        print("jieba 未安装，跳过。安装：pip install jieba")


if __name__ == "__main__":
    main()
