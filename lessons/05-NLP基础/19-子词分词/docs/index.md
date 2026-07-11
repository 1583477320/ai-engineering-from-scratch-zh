# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到没见过的词就噎住。字符级分词器把序列炸成碎片。子词分词器在两者之间取得了平衡。每一个现代 LLM 都搭载一个。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05 · 01（文本预处理）、阶段 05 · 04（GloVe / FastText / 子词）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 10 · 01（分词器从零）— 本课的概念在阶段 10 会被实现为完整的 BPE tokenizer

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 BPE、WordPiece、Unigram 三种算法各自的合并/剪枝原理——谁贪心、谁最大化似然、谁概率剪枝
- [ ] 区分 SentencePiece（训练词表）和 tiktoken（使用词表）——什么场景选哪个
- [ ] 理解字节级 BPE 为什么能消灭 `[UNK]`——256 字节基础词表，任何 UTF-8 输入皆可编码
- [ ] 为中文 LLM 选择 tokenizer 策略——BPE vs Unigram vs 逐字，各自的词表效率和下游影响

---

## 1. 问题

你的词表有 5 万个词。用户输入了"untokenizable"。你的分词器返回 `[UNK]`。模型对这个词没有任何信号。更糟糕的是——语料库中 90% 分位的文档包含 40 个稀有词，意味着每篇文档扔掉了 40 个信息位。

**子词分词解决了这个问题。** 常见词保持为单个 token。稀有词分解为有意义的碎片：`untokenizable` → `un` + `token` + `izable`。训练数据覆盖一切——因为任何字符串最终都是字节序列。

2026 年每一个前沿 LLM 都搭载三选一算法（BPE、Unigram、WordPiece）中的一种，被装在三选一库（tiktoken、SentencePiece、HF Tokenizers）中。你不能不选一个。

---

## 2. 概念——三种算法对照

### 2.1 BPE（Byte-Pair Encoding）

**从字符级词表出发。** 统计每一对相邻 token。合并最频繁的一对 → 新 token。重复直到达到目标词表大小。

使用者：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

### 2.2 字节级 BPE

**同样的算法但在原始字节（256 个基础 token）而非 Unicode 字符上操作。** 保证零 `[UNK]`——任何字节序列都可编码。GPT-2 使用 50,257 token（256 字节 + 50,000 次合并 + 1 特殊 token）。

### 2.3 Unigram

**从巨大的词表出发。** 给每个 token 分配一个 unigram 概率。迭代剪枝——移除对语料对数似然增加最小的 token。**推理时是概率性的**——可以对同一输入采样不同分词（子词正则化，对数据增强有价值）。

使用者：T5、mBART、ALBERT、XLNet、Gemma。

### 2.4 WordPiece

**合并使训练语料似然最大化的一对**——而非原始频率最高的那对。与 BPE 贪心的频率选择不同：BPE 问"哪一对最频繁？"，WordPiece 问"哪一对在一起出现的概率比随机更高？"

使用者：BERT、DistilBERT、ELECTRA。

### 2.5 三种算法对照

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| **原理** | 贪心合并最高频相邻对 | 合并使似然最大化的对 | 从大词表迭代剪枝损失最小的 token |
| **方向** | 自底向上（字符→子词） | 自底向上 | 自顶向下（大词表→剪枝） |
| **推理** | 确定性——按合并顺序 apply | 确定性 | **概率性**——可采样不同分词 |
| **`##` 前缀** | 无（或 `Ġ`） | `##` 标记接续子词 | 无（或 `▁`）|

### 2.6 SentencePiece vs tiktoken

| | SentencePiece | tiktoken |
|---|---|---|
| **角色** | **训练**词表 + 编码 | **编码**（使用预训练词表） |
| **输入** | 原始 Unicode 文本——不需要预分词 | 预构建的词表文件 |
| **空白处理** | 将空格编码为 `▁`（U+2581） | 基于字节——空白本身是字节序列 |
| **中文友好度** | **高**——直接操作于 Unicode 字符流，对无空格语言天然友好 | 高——但针对英文做了更多优化 |

**经验法则：**
- **训练新词表：** SentencePiece（多语言、无预分词）或 HF Tokenizers
- **快速推理、GPT 兼容词表：** tiktoken（`cl100k_base`、`o200k_base`）
- **两者都要：** HF Tokenizers——一个库，训练 + 部署

---

## 3. 从零实现

### 第 1 步：BPE 从零

```python
from collections import Counter

def train_bpe(corpus, num_merges):
    """corpus = {'low': 5, 'lower': 2, ...} 词-频字典"""
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq           # 频率加权——高频词中的pair更重要
        if not pairs: break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)        # 在全部词表中应用合并
    return merges
```

**算法编码的三个事实。** `</w>` 标记词尾——"low"作为后缀和"lower"作为前缀保持独立。频率加权使高频词中的 pair 更早胜出。合并列表是有序的——推理时按训练顺序 apply。

### 第 2 步：用学到的合并规则编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:          # 必须按学习顺序
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现是 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用merge-rank 查找 + 优先队列，在近线性时间内运行。

### 第 3 步：SentencePiece 实战

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",              # 或 "unigram"
    character_coverage=0.9995,     # 英文~0.9995，中文/日文可降到0.995
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制稀有字符的保留程度——覆盖率越低，越多罕见字被映射到 `<unk>`。CJK（中日韩）语言建议调低到 0.995 以控制词表增长。

### 第 4 步：tiktoken——OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))         # [127340, 101028]
print(len(enc.encode("Hello, world!")))    # 4
```

仅编码。快（Rust 后端）。与 GPT-4/5 分词完全一致——用于 token 计数、费用估算和上下文窗口预算。

完整 BPE 实现见 `code/`（本课代码在阶段 04 的 `glove_fasttext_bpe.py` 中）。

---

## 4. 2026 年仍在混入生产的陷阱

- **Tokenizer 漂移。** 训练时用词表 A，部署时对上了词表 B。Token ID 不同——模型输出垃圾。CI 中校验 `tokenizer.json` 的 hash
- **空白歧义。** BPE 中"hello" vs " hello"生成不同的 token。始终显式指定 `add_special_tokens` 和 `add_prefix_space`
- **多语言训练不足。** 英文主导的语料产出的词表将非拉丁文字切分成 5-10 倍的 token。同样的 prompt 在日文/阿拉伯文上花费 5-10 倍于英文的 token 成本。`o200k_base` 部分修复了这个问题
- **Emoji 切分。** 单个 emoji 可能需要 5 个 token。做上下文预算时检查 emoji 处理

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| 从头训练单语言模型 | HF Tokenizers (BPE) |
| 从头训练多语言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 部署 OpenAI 兼容 API | tiktoken (`o200k_base` for GPT-4+) |
| 领域特定词表（代码/数学/蛋白质） | 在领域语料上训练自定义 BPE，与基础词表合并 |
| 边缘推理、小模型 | Unigram（小词表效果更好） |

**词表大小是一个扩缩决策，不是一个常数。** 粗略启发式：< 10 亿参数 → 32K。1-10B → 50-100K。多语言/前沿 → 200K+。

### 中文子词分词特别建议

- **"字"本身已经是高信息密度单元。** 英文需要从 26 个字母逐步合并——"machine learning"从字符到词需要多次合并。中文"机器学习"只需合并"机器"+"学习"两步——因为"字"本身就已经承载了语素级别的信息
- **中文 tokenizer 词表可小于英文。** 32K 中文词表 ≈ 英文 50K 的效果——因为中文 token 的"信息密度"更高。一个汉字 ≈ 一个英文词在 BPE 的词表效率统计中大致成立
- **简繁不可混用同一 tokenizer。** 训练语料是简体 → 繁体输入会被切得非常碎。需要繁简同时支持 → 训练 tokenizer 的语料中混合简繁文本
- **中文 BPE vs Unigram：** BPE 的热门合并（"学习"、"机器"）对中文高频词尤其有效。Unigram 的概率剪枝为中文保留了更丰富的词语变体——对子词正则化（数据增强）有价值

---

## 6. 知识连线

子词分词是连接经典 NLP 预处理（阶段 05 · 01）和 LLM 架构（阶段 10）的桥梁：

- **阶段 04（GloVe/FastText）→** FastText 的子词 n-gram 思想被 BPE 及其变体吸收——"子词"从学术场景进入了每一个 API 调用
- **阶段 10（大语言模型从零）→** 阶段 10 · 01 将从零实现一个完整的 BPE tokenizer——本课的概念和算法是那个实现的直接前置
- **阶段 18（多语言 NLP）→** 本课的多语言 tokenizer 选择和分词税分析是阶段 18 中"capacity spillover tax"的直接技术根源

---

## 7. 面试考点

### Q1：BPE 和 WordPiece 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
BPE 合并最频繁的相邻对（纯频率）。WordPiece 合并使训练语料似然最大化的对——`count(a,b) / (count(a) × count(b))` 最大的对。这意味着 WordPiece 倾向于合并"在一起出现的概率比随机更高"的对——比 BPE 更抗拒将偶然共现的词对合并。BERT 的 `##` 前缀是 WordPiece 的视觉标记——`##` 表示这个子词是接续前面 token 的。

### Q2：从零训练中文 LLM——选 BPE 还是 Unigram？（难度：⭐⭐⭐）

**参考答案：**
Unigram 的两个优势：**(1)** 概率性分词——训练时可以采样多种分词（子词正则化），提高模型对分词噪声的鲁棒性。中文的多种合理分词（"机器学习" vs "机器/学习"都可以是对的）正好受益于此。**(2)** 剪枝使词表更紧凑——Unigram 的去冗余机制在中文高频字上尤其有效。选 Unigram + `character_coverage=0.995`。如果训练语料以简体为主但需要支持繁体——确保语料中混合 10-20% 繁体文本。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| BPE | "字节对编码" | 贪心合并最高频字符对直到目标词表大小。GPT-2/3/4、Llama 的底层 |
| 字节级 BPE | "永远不会 OOV" | BPE 在原始 256 字节上操作；GPT-2 使用 50,257 token 的词表 |
| Unigram | "概率分词器" | 从大候选集用对数似然剪枝；T5、Gemma 使用 |
| SentencePiece | "处理空白的那个" | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁` |
| tiktoken | "很快的那个" | OpenAI 的 Rust 后端 BPE 编码器——用于预构建词表。不训练 |
| 合并列表 | "神奇的排序" | 有序的 `(a, b) → ab` 合并序列；推理时按序 apply |
| 字符覆盖率 | "多罕见才算太罕见？" | tokenizer 必须覆盖的训练语料中字符的比例；典型 ~0.9995 |

---

## 📚 小结

BPE、WordPiece、Unigram 是 2026 年每一个 LLM 搭载的三种子词分词算法。BPE 贪心合并频率最高的对，Unigram 概率剪枝，WordPiece 最大化似然。SentencePiece 训练词表（中文友好——Unicode 原生），tiktoken 使用词表（GPT 兼容——极速推理）。

字节级 BPE 从根源消灭了 OOV——256 字节基础词表，任何 UTF-8 输入皆可编码。为中文 LLM 选择 tokenizer：Unigram + `character_coverage=0.995` + 简繁混合语料。

---

## ✏️ 练习

1. 【理解】在玩具语料上训练 500 次合并的 BPE。编码 3 个留出词——多少个恰好 1 个 token vs > 1 个 token？

2. 【实现】对比 `cl100k_base`、`o200k_base` 和你训练的 32K SentencePiece BPE 在 100 句英文 Wikipedia 上的 token 数。报告各方案的压缩比。

3. 【实验】用 BPE、Unigram 和 WordPiece 分别训练同一语料。在下游情感分类器中使用各自的 tokenizer。报告三种方案的 F1——选择差异是否超过 1 个点？

4. 【思考】你的中文 tokenizer 在处理用户输入"OpenAI发布了GPT-5"时——中英混合——效果如何？英文部分被切碎了多少？如何改进训练语料来缓解这个问题？

---

## 📖 参考资料

1. [论文] Sennrich, Haddow, Birch. "Neural Machine Translation of Rare Words with Subword Units". ACL, 2016. https://arxiv.org/abs/1508.07909 — BPE 论文
2. [论文] Kudo. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates". ACL, 2018. https://arxiv.org/abs/1804.10959 — Unigram 论文
3. [论文] Kudo, Richardson. "SentencePiece: A simple and language independent subword tokenizer". EMNLP, 2018. https://arxiv.org/abs/1808.06226 — SentencePiece 论文
4. [官方文档] Hugging Face — Summary of the tokenizers. https://huggingface.co/docs/transformers/tokenizer_summary — BPE/WordPiece/Unigram 的实用对比
5. [GitHub] OpenAI tiktoken. https://github.com/openai/tiktoken — cookbook + 编码列表

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文子词分词建议、简繁 tokenizer 训练策略、工程最佳实践、常见错误、面试考点等均为原创内容。
