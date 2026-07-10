# summarize.py — 文本摘要：TextRank 抽取式 + ROUGE 评估
# 依赖：无（纯标准库）
# 安装：无需额外安装
# 对应课程：阶段 05 · 12（文本摘要）

import math
import re
from collections import Counter
from typing import List


# ============================================================
# 1. 分句
# ============================================================

def sentence_split(text: str) -> List[str]:
    """按句号、问号、感叹号分句。"""
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*", text.strip())
            if s.strip()]


# ============================================================
# 2. 句子相似度（基于词频的余弦相似度近似）
# ============================================================

def similarity(s1: str, s2: str) -> float:
    """两句的词汇重叠度——分母用 log 归一化抑制长句优势。"""
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    return intersection / denom if denom else 0.0


# ============================================================
# 3. TextRank——基于图的抽取式摘要
# ============================================================

def textrank(text: str,
             top_k: int = 3,
             damping: float = 0.85,
             iterations: int = 50,
             epsilon: float = 1e-4) -> List[str]:
    """TextRank 抽取式摘要。

    将文章建模为图：节点=句子，边=相似度。
    运行 PageRank 算法——"被高分的相似句子指向的句子也是高分的"。
    迭代到收敛或达到最大次数。选分数最高的 top_k 句保留原文顺序。

    优点：输出永远语法正确（原文原句）。
    风险：零散分布在文章各处的关键信息可能被遗漏。
    """
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    # 构建相似度矩阵
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    # PageRank 迭代
    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += (damping * sim[i][j]
                                      / total_out * scores[i])
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    # 选 top_k 并恢复原文顺序
    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]


# ============================================================
# 4. ROUGE-N——召回率导向的摘要评估
# ============================================================

def rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """ROUGE-N 召回率 = 参考中的 n-gram 有多少在摘要中出现。

    为什么是召回率而非精确率？
    摘要评估关心的是"参考中的关键内容有没有被覆盖"——
    漏掉关键信息比多写了几个词更严重。
    """
    def ngrams(tokens, k):
        return Counter(tuple(tokens[i:i + k])
                       for i in range(len(tokens) - k + 1))

    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    hyp_ngrams = ngrams(hyp_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    return overlap / sum(ref_ngrams.values())


# ============================================================
# 演示主程序
# ============================================================

def main():
    # 英文演示
    article = (
        "Researchers at a Canadian university published a paper on "
        "efficient transformers. The paper introduces a new attention "
        "variant that runs in linear time. The authors trained models "
        "up to 1 billion parameters on public data. Benchmarks show "
        "the new attention matches standard attention on most tasks. "
        "The authors released training code and weights on GitHub. "
        "Several research labs have already replicated the main results. "
        "The paper has been accepted at NeurIPS."
    )
    reference = (
        "Researchers introduced a linear-time attention variant, "
        "trained up to 1B parameters, matched standard attention on "
        "benchmarks, and released code and weights."
    )

    print("=" * 60)
    print("TextRank 抽取式摘要")
    print("=" * 60)
    summary = textrank(article, top_k=3)
    for i, s in enumerate(summary):
        print(f"  [{i+1}] {s}")

    joined = " ".join(summary)
    print(f"\nROUGE 评估:")
    for n in [1, 2]:
        print(f"  ROUGE-{n}: {rouge_n(joined, reference, n=n):.3f}")
    print("  (生产环境使用 `pip install rouge-score` ——含F-measure和stemming)")

    # 中文演示
    print(f"\n{'='*60}")
    print("中文 TextRank（以词为单元）")
    print("=" * 60)
    zh_article = (
        "加拿大一所大学的研究人员发表了一篇关于高效Transformer的论文。"
        "该论文提出了一种新的注意力变体，可以在线性时间内运行。"
        "作者使用公开数据训练了多达10亿参数的模型。"
        "基准测试显示，新的注意力机制在大多数任务上达到了标准注意力的水平。"
        "作者在GitHub上发布了训练代码和模型权重。"
        "多个研究实验室已经复现了主要结果。"
        "该论文已被NeurIPS接收。"
    )
    zh_ref = (
        "研究人员提出了一种线性时间注意力变体，训练了10亿参数模型，"
        "在基准测试上达到了标准注意力水平，并开源了代码和权重。"
    )

    try:
        import jieba
        # 将中文文本按词切分后做 TextRank
        jieba_article = " ".join(jieba.cut(zh_article))
        jieba_ref = " ".join(jieba.cut(zh_ref))

        zh_summary = textrank(jieba_article, top_k=3)
        print("摘要句:")
        for i, s in enumerate(zh_summary):
            print(f"  [{i+1}] {s.strip()}")

        joined_zh = " ".join(zh_summary)
        print(f"\nROUGE 评估（jieba 分词后）:")
        for n in [1, 2]:
            print(f"  ROUGE-{n}: {rouge_n(joined_zh, jieba_ref, n=n):.3f}")
    except ImportError:
        print("jieba 未安装，中文演示跳过。pip install jieba")

    # 关键对比
    print(f"\n{'='*60}")
    print("抽取式 vs 生成式：核心差异")
    print("=" * 60)
    print("┌──────────┬─────────────────┬──────────────────┐")
    print("│          │ 抽取式(TextRank) │ 生成式(BART/Pegasus)│")
    print("├──────────┼─────────────────┼──────────────────┤")
    print("│ 输出来源 │ 原文句子（原封不动）│ 模型生成（可能新词）│")
    print("│ 语法保证 │ 100% 正确         │ 可能不通顺          │")
    print("│ 事实准确 │ 100%（不创造内容） │ 可能幻觉——编造信息  │")
    print("│ 压缩能力 │ 低——句子级别选择   │ 高——可任意缩短      │")
    print("│ 适用场景 │ 新闻/论文/法律    │ 对话/标题/自由格式   │")
    print("└──────────┴─────────────────┴──────────────────┘")


if __name__ == "__main__":
    main()
