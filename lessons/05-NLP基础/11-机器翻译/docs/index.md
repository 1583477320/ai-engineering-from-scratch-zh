# 机器翻译

> NLP 三十年的金主。翻译质量可以衡量，人与机器之间的差距顽固——每一步前进都因为这个差距而诞生。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 10（注意力机制）、阶段 05 · 04（GloVe、FastText、子词）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 12（文本摘要）— 同样是 Seq2Seq 但目标和评估完全不同

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 BLEU 和 chrF 评估指标——理解裁剪计数、短句惩罚、字符 F-score 的设计动机
- [ ] 使用 NLLB-200 或 mBART 调用预训练多语言翻译模型
- [ ] 解释为什么 BLEU<1 的差异属于噪声——以及什么时候该用 chrF 替代 BLEU
- [ ] 理解中英翻译评估中 BLEU 的系统性低估问题——分词不一致是根源

---

## 1. 问题

"我想你。" → 法文："Tu me manques." 字面意思是"你对我来说是缺失的"——没有一个词级别的对齐能幸存。

机器翻译是迫使 NLP 发明了编码器-解码器、注意力、Transformer、以及最终的整个 LLM 范式的任务。每一步前进都是因为翻译质量能被可衡量地评估，而人类水平与机器水平之间的差距顽固地存在。

本课跳过历史课，直接教 2026 年的 working pipeline：预训练多语言编码器-解码器（NLLB-200 / mBART）、子词分词器、束搜索解码、BLEU + chrF 评估、以及仍在悄悄混入生产环境的那几个失败模式。

---

## 2. 概念

### 2.1 现代 MT 的流水线

```
源语言文本 → SentencePiece BPE 分词 → 编码器 → 交叉注意力 → 解码器（束搜索） → 去分词 → 目标语言文本
```

三个操作性的选择决定了真实的翻译质量：

- **分词器。** SentencePiece BPE 在混合多语言语料上训练。跨语言的共享词表是 NLLB 能实现零样本语言对翻译的前提
- **模型大小。** NLLB-200 蒸馏版 600M——笔记本可跑。3.3B——生产默认。54.5B——研究天花板
- **解码策略。** 束宽 4-5 是通用内容的默认。用长度惩罚避免过短输出。术语一致性需求下使用约束解码

### 2.2 中英翻译的特殊挑战

| 挑战 | 说明 |
|---|---|
| 语序差异 | 中文 SVO vs 英文 SVO 大体一致但状语位置不同——'我昨天在图书馆看书' → 'I read a book at the library yesterday' |
| 量词缺失 | 英文需要冠词（a/the），中文不需要——翻译凭空添加了英文中没有显式来源的信息 |
| 时态不对应 | 中文无动词时态变化——'我吃饭'可以翻译为 I eat / I ate / I am eating / I will eat，取决于上下文 |
| 成语/典故 | '画蛇添足' → 字面翻译完全无意义。需要跨语言语义映射 |

### 2.3 BLEU——n-gram 精确率 × 短句惩罚

BLEU = 1~4 gram 精确率的几何平均 × 短句惩罚。核心设计决策：

- **裁剪计数。** 译文出现 7 次"the"但参考只有 2 次 → 匹配计数裁剪为 2。惩罚无意义的重复
- **短句惩罚。** 如果只输出"猫"而参考是"猫在跑" → BLEU 天然被短句惩罚拖低。防止模型通过极短输出来获得高分
- **< 1 BLEU = 噪声。** 不同参考译文之间的 BLEU 波动通常在 2-5 的范围内。差值 < 1 BLEU 在统计上无法区分不同的系统

### 2.4 chrF——字符级的公平度量

chrF 计算的是**字符级** n-gram 的 F-score（通常 n=6, beta=2——召回率权重更高）。相比 BLEU：

- 对形态丰富的语言（捷克语、芬兰语、土耳其语）更公平——'running' vs 'run' 在 BLEU 中是 0 匹配，在 chrF 中是部分匹配（共享 'run' 子串）
- 对中文等无空格语言更稳定——不需要分词，自然绕过了"这个词边界对不对"的问题
- 对中译英评估尤其有价值——中文"了/着/过"等虚词在 BLEU 中完全不给分（没有英文对应），但在 chrF 中至少是字符级的部分重叠

---

## 3. 从零实现

### 第 1 步：BLEU——n-gram 精确率 + 短句惩罚

```python
def ngram_precision(hyp_tokens, ref_tokens, n):
    """n-gram 精确率——裁剪计数防止过度计数。"""
    hyp_counts = Counter(ngrams(hyp_tokens, n))
    ref_counts = Counter(ngrams(ref_tokens, n))
    clipped = sum(min(count, ref_counts.get(ng, 0))
                  for ng, count in hyp_counts.items())
    return clipped / max(1, sum(hyp_counts.values()))

def brevity_penalty(hyp_len, ref_len):
    """短句惩罚——译文过短时惩罚 BLEU。"""
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / max(1, hyp_len))

def simple_bleu(hypothesis, reference, max_n=4):
    """BLEU = 几何平均 (P1...P4) × 短句惩罚。"""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    precisions = [ngram_precision(hyp, ref, n) for n in range(1, max_n + 1)]
    if any(p == 0 for p in precisions):
        return 0.0  # 教学版无平滑——任一 n-gram P=0 → BLEU=0
    log_mean = sum(math.log(p) for p in precisions) / max_n
    return 100 * brevity_penalty(len(hyp), len(ref)) * math.exp(log_mean)
```

### 第 2 步：chrF——字符级 F-score

```python
def chrf(hypothesis, reference, n=6, beta=2.0):
    """chrF = 字符 n-gram F_beta score。beta=2 → 召回率权重 2x 精确率。"""
    def char_ngrams(text, k):
        return [text[i:i + k] for i in range(len(text) - k + 1)]

    hyp, ref = hypothesis.lower(), reference.lower()
    precisions, recalls = [], []
    for k in range(1, n + 1):
        hc, rc = Counter(char_ngrams(hyp, k)), Counter(char_ngrams(ref, k))
        match = sum((hc & rc).values())
        if sum(hc.values()) and sum(rc.values()):
            precisions.append(match / sum(hc.values()))
            recalls.append(match / sum(rc.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0
```

### 第 3 步：BLEU vs chrF 的互补性

```python
>>> simple_bleu("猫正在跑。", "猫在跑。")    # → 0.0（4-gram 无匹配）
>>> chrf("猫正在跑。", "猫在跑。")           # → ~70（字符级大量重叠）
```

**BLEU 高 + chrF 高** = 精准且全面。**BLEU 低 + chrF 高** = 词义对但表达不地道。两个指标互补——单独任何一个都不能全面评价翻译质量。

完整代码见 `code/mt_eval.py`。

---

## 4. 工业工具

### 4.1 NLLB-200——200 种语言的开箱翻译

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="zho_Hans")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "猫正在跑。"
inputs = tok(src, return_tensors="pt")
out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
    num_beams=5, length_penalty=1.0, max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
# "The cats are running."
```

三个关键点：`src_lang` 告诉分词器用哪种语言的分割。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的约定——mBART 和 M2M-100 有自己的，不可互换。

### 4.2 评估——用 sacrebleu 而非自实现

```bash
pip install sacrebleu
```

```python
from sacrebleu.metrics import BLEU, CHRF
bleu = BLEU()
chrf = CHRF(word_order=2)
print(bleu.corpus_score(hypotheses, [references]))
print(chrf.corpus_score(hypotheses, [references]))
```

Sacrebleu 内置了标准的平滑方法（避免单句 BLEU=0）、标准分词器（语言感知的）、和标准的签名（方便复现——报告 BLEU 时一并报告 sacrebleu 的签名）。

### 4.3 2026 年 MT 框架选择

| 场景 | 选择 |
|---|---|
| 通用多语言、零样本语言对 | NLLB-200 (蒸馏 600M 或 3.3B) |
| 英-欧/英-亚、有领域数据微调 | mBART-50 |
| 仅中英、有百万级平行语料 | 微调 mBART 或 NLLB |
| 研究级最高准确率 | NLLB-54.5B 或 GPT-4 零样本 + 术语约束 |
| 设备端离线翻译 | 自训练的小型 Transformer（~50M 参数） |

---

## 5. 知识连线

MT 是本阶段前 10 课的完整集成——分词（01）→ BPE（04）→ Seq2Seq（09）→ 注意力（10）。评估指标 BLEU/chrF 是 NLP 中少数可重复的客观指标之一——这也是为什么 MT 被用作新架构的"测试场地"。

---

## 6. 常见错误

### 错误 1：用自实现的 BLEU 交叉比较论文数字

**现象：** "我们的 BLEU 比论文高 3 个点" → 检查发现用了不同的分词器。

**原因：** BLEU 对分词、平滑、参考译文数量高度敏感。同一输出、不同分词器 → BLEU 相差 2-5 点。论文中的 BLEU 数字只有用完全相同的 sacrebleu 签名复现时才是可比。

**修复：** 始终用 sacrebleu 并报告签名（`sacrebleu.BLEU().get_signature()`）。这个签名记录了所有评价参数——任何人可用它复现完全相同的 BLEU 数字。

### 错误 2：中英翻译评估不用 chrF

**现象：** 仅报告 BLEU。中文译文 BLEU 系统性低于英文译文 BLEU——但翻译质量可能并不差。

**原因：** BLEU 依赖精确的词边界匹配。中文的词边界由 jieba/分词器决定——分词选择直接影响 BLEU。同一句中文可以用 3 种不同的分词方式产生 3 个不同的 BLEU 分数。

**修复：** 始终同时报告 chrF——它绕过词边界问题。中英翻译评估的标准做法是 BLEU + chrF 联合。

---

## 7. 面试考点

### Q1：BLEU 为什么用几何平均而非算术平均？（难度：⭐⭐）

**参考答案：**
几何平均对极端值敏感——只要有一个 n-gram 的精确率为 0，BLEU 就是 0。这正是 BLEU 的设计意图——4-gram 精确率为 0 意味着"没有一个连续的 4-gram 匹配"，这在翻译中不是小问题——它意味着语序或搭配很可能是错的。算术平均会给你 75 分（其他 3 个 n-gram 很好），掩盖了 4-gram 的完全失败。

### Q2：中译英 vs 英译中的评估策略有什么不同？（难度：⭐⭐⭐）

**参考答案：**
中译英：BLEU 相对可靠——英文输出有天然词边界，分词一致性好。但 chrF 仍然有价值——对动词时态变化（run/runs/ran/running）给予部分分数。英译中：BLEU 系统性偏低——因为中文分词是概率性的，不同分词器的选择直接影响 BLEU。加上中文量词（一个/一只/一头）在英文中无对应——英文 BLEU 高了但中文表达可能不地道。**最佳实践：双向翻译评估中始终同时报告 BLEU + chrF，中→英侧用 chrF 作为校准参考。**

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| BLEU | "翻译的分数" | n-gram 精确率的几何平均 × 短句惩罚。对词序敏感。30=可用，40=良好，50=优秀 |
| chrF | "BLEU 的替代品" | 字符 n-gram F-score。对形态变化宽容，不依赖分词——无空格语言的首选 |
| 裁剪计数 | "防止重复骗分" | 译文 7 次"the"→参考 2 次→匹配计数上限=2。惩罚无意义重复 |
| 短句惩罚 | "你输出太短了" | 译文过短→BP<1。防止"只输出一个词"获得高精确率 |
| sacrebleu | "BLEU 的标准实现" | 内置平滑、标准分词器、可复现签名。生产环境不要自己写 BLEU |

---

## 📚 小结

机器翻译是 Seq2Seq + 注意力 + 子词分词器的完整集成——也是 NLP 三十年架构演进的主要驱动力。BLEU 和 chrF 互补评估——BLEU 要求连续匹配和语序一致性，chrF 对形态变化和词边界不确定性宽容。差值 < 1 BLEU 属于噪声。中英翻译永远同时报告两个指标。

---

## ✏️ 练习

1. 【理解】找 3 句中文→英文的机器翻译结果。手工打分（1-5），与 BLEU 和 chrF 对比。找出 BLEU 与人工评分差异最大的那句，解释原因。

2. 【实现】在 `simple_bleu` 中加入平滑——当 n-gram 精确率为 0 时，用 epsilon=0.01 替代 0。比较平滑前后在短句上的 BLEU 变化。

3. 【实验】用 NLLB-200 翻译 10 句中英对照新闻。报告 sacrebleu BLEU 和 chrF。分析 BLEU 最低的 2 句——是翻译真的差还是评估指标的问题？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| BLEU + chrF 从零实现 | `code/mt_eval.py` | 裁剪计数+短句惩罚+字符F-score，含中英文评估示例 |

---

## 📖 参考资料

1. [论文] Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation". ACL, 2002. https://aclanthology.org/P02-1040/ — BLEU 论文
2. [论文] Popović. "chrF: character n-gram F-score for automatic MT evaluation". WMT, 2015. https://aclanthology.org/W15-3049/ — chrF 论文
3. [官方文档] sacrebleu. https://github.com/mjpost/sacrebleu — 生产环境的标准评估工具
4. [模型] NLLB Team. "No Language Left Behind". https://arxiv.org/abs/2207.04672 — NLLB-200 论文

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、中英翻译分析、工程最佳实践、常见错误、面试考点等均为原创内容。
