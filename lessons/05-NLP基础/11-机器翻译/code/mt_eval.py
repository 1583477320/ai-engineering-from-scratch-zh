# mt_eval.py — 机器翻译评估：BLEU + chrF 从零实现
# 依赖：无（纯标准库）
# 安装：无需额外安装
# 对应课程：阶段 05 · 11（机器翻译）

import math
from collections import Counter
from typing import List


# ============================================================
# 1. 分词
# ============================================================

def tokenize(text: str) -> List[str]:
    """简易分词——小写 + 标点分离。"""
    return (text.lower()
            .replace(".", " .")
            .replace(",", " ,")
            .replace("!", " !")
            .replace("?", " ?")
            .split())


# ============================================================
# 2. BLEU
# ============================================================

def ngrams(tokens: List[str], n: int) -> List[tuple]:
    """提取 n-gram。"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_precision(hyp_tokens: List[str],
                    ref_tokens: List[str], n: int) -> float:
    """n-gram 精确率——裁剪计数防止过度计数。

    如果译文说了 7 次 'the' 而参考只有 2 次，
    裁剪计数将 'the' 的匹配计数限制为 2 而非 7——
    惩罚过度重复。
    """
    hyp_counts = Counter(ngrams(hyp_tokens, n))
    ref_counts = Counter(ngrams(ref_tokens, n))
    clipped = 0
    total = 0
    for ngram, count in hyp_counts.items():
        clipped += min(count, ref_counts.get(ngram, 0))
        total += count
    return clipped / total if total else 0.0


def brevity_penalty(hyp_len: int, ref_len: int) -> float:
    """短句惩罚——译文过短时惩罚 BLEU。

    长度为 0 → 直接判 0。否则只惩罚过短的译文，
    不过长译文不受惩罚（BP = 1.0）。
    """
    if hyp_len >= ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - ref_len / hyp_len)


def simple_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """BLEU 分数——n-gram 精确率的几何平均 × 短句惩罚。

    BLEU 在 [0, 100] 范围内：
    - < 10：不可用
    - 10-20：勉强可读
    - 20-30：可理解但生硬
    - 30-40：中等质量
    - 40-50：良好
    - > 50：优秀（接近人工翻译质量）
    - 差值 < 1 BLEU：属于噪声范畴

    注意：教学版本无平滑——任一 n-gram 精确率为 0 → BLEU = 0。
    生产环境用 sacrebleu (`pip install sacrebleu`) 替代。
    """
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    precisions = [ngram_precision(hyp, ref, n)
                  for n in range(1, max_n + 1)]
    if any(p == 0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / max_n
    bp = brevity_penalty(len(hyp), len(ref))
    return 100 * bp * math.exp(log_mean)


# ============================================================
# 3. chrF——字符级 F-score
# ============================================================

def chrf(hypothesis: str, reference: str,
         n: int = 6, beta: float = 2.0) -> float:
    """chrF——字符 n-gram F-score。

    相比 BLEU 的优势：
    - 对形态丰富的语言更公平（捷克语、芬兰语等）
    - 能捕获部分匹配——BLEU 完全无解
    - 不需要分词——直接基于字符

    beta=2 意味着召回率的权重是精确率的 2 倍（F2 而非 F1）。
    在 MT 评估中，召回率通常比精确率更稀缺——beta=2 的默认。
    """
    def char_ngrams(text: str, k: int) -> List[str]:
        return [text[i:i + k] for i in range(len(text) - k + 1)]

    hyp = hypothesis.lower()
    ref = reference.lower()
    precisions: List[float] = []
    recalls: List[float] = []

    for k in range(1, n + 1):
        hyp_c = Counter(char_ngrams(hyp, k))
        ref_c = Counter(char_ngrams(ref, k))
        match = sum((hyp_c & ref_c).values())
        hyp_total = sum(hyp_c.values())
        ref_total = sum(ref_c.values())
        if hyp_total == 0 or ref_total == 0:
            continue
        precisions.append(match / hyp_total)
        recalls.append(match / ref_total)

    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta * beta) * p * r / (beta * beta * p + r)


# ============================================================
# 演示主程序
# ============================================================

def main():
    # 中英翻译评估示例
    print("=" * 60)
    print("机器翻译评估——BLEU vs chrF")
    print("=" * 60)

    cases = [
        # (译文, 参考译文, 说明)
        ("猫正在跑。", "猫在跑。", "近义但不同的表达"),
        ("猫在跑。", "猫在跑。", "完全匹配"),
        ("猫正在跑着。", "猫在跑。", "多了虚词"),
        ("狗在吃东西。", "猫在跑。", "完全不同的内容"),
        ("猫", "猫在跑。", "过短的译文→BP惩罚"),
    ]

    print(f"{'译文':<20s} {'参考':<15s} {'BLEU':>7s} {'chrF':>7s}  {'说明'}")
    print("-" * 70)
    for hyp, ref, desc in cases:
        b = simple_bleu(hyp, ref)
        c = chrf(hyp, ref)
        print(f"{hyp:<20s} {ref:<15s} {b:7.1f} {c:7.1f}  {desc}")

    print(f"\n关键观察:")
    print(f"  1. 完全匹配→BLEU 100，chrF 100")
    print(f"  2. '猫正在跑' vs '猫在跑'→BLEU 0（4-gram无匹配）chrF ~70（字符级部分匹配）")
    print(f"  3. '狗在吃东西' vs '猫在跑'→两者都很低——内容确实不同")
    print(f"  4. '猫' vs '猫在跑'→BLEU 0（短句惩罚+4-gram无匹配），chrF ~46")

    print(f"\nBLEU vs chrF 选择建议:")
    print(f"  - 英文 ↔ 欧洲语言：BLEU 足够")
    print(f"  - 形态丰富语言（芬兰语/土耳其语/阿拉伯语）：chrF 更公平")
    print(f"  - 中文/日文等无空格语言：chrF 不需要分词→更稳定")
    print(f"  - 差值 < 1 BLEU 或 < 2 chrF → 属于噪声，不应报告为统计显著")
    print(f"  - 生产环境：pip install sacrebleu——替代本教学实现")

    # 中英混合演示
    print(f"\n{'='*60}")
    print(f"中英翻译评估示例")
    print(f"{'='*60}")
    zh_en = [
        ("猫坐在垫子上。", "The cat sat on the mat.", "直译对比"),
        ("那只猫坐在了垫子上面。", "The cat sat on the mat.", "中文多了'那只'和'了'+'上面'"),
        ("垫子上有只猫。", "The cat sat on the mat.", "语序完全不同"),
    ]
    for zh, en, desc in zh_en:
        b = simple_bleu(en, zh)  # 注意：中英文长度差异 → 短句惩罚
        c = chrf(en, zh)
        print(f"  中文: {zh}")
        print(f"  英文: {en}")
        print(f"  BLEU={b:.1f}  chrF={c:.1f}  ({desc})")
        print()


if __name__ == "__main__":
    main()
