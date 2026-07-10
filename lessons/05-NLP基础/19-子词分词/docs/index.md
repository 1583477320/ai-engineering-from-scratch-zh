# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到没见过的词就噎住。字符级分词器把序列炸成碎片。子词分词器在两者之间取得了平衡。每一个现代 LLM 都搭载一个。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 05 · 01、05 · 04 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 区分 BPE、WordPiece、Unigram 三种子词算法——各自的合并/剪枝原理和使用者
- [ ] 理解字节级 BPE 为什么能消灭 `[UNK]`——基础词表 = 256 个字节，任何 UTF-8 输入皆可编码
- [ ] 解释 SentencePiece vs tiktoken——一个训练词表，一个使用词表

---

## 1. 问题

你的词表有 5 万个词。用户输入了"untokenizable"。你的分词器返回 `[UNK]`。模型对这个词没有任何信号。更糟糕的是——语料库中 90% 分位的文档包含 40 个稀有词，意味着每篇文档扔掉了 40 个信息位。

子词分词解决了这个问题。常见词保持为单个 token。稀有词分解为有意义的碎片：`untokenizable` → `un` + `token` + `izable`。训练数据覆盖一切——因为任何字符串最终都是字节序列。

2026 年每一个前沿 LLM 都基于三选一算法（BPE、Unigram、WordPiece）被装在三选一库中（tiktoken、SentencePiece、HF Tokenizers）。你不能不选一个。

---

## 2. 三种算法对照

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| **原理** | 贪心合并最高频相邻对 | 合并使语料似然最大化的对 | 从大词表迭代剪枝使损失最小的 token |
| **方向** | 自底向上（字符→子词） | 自底向上 | 自顶向下（大词表→剪枝） |
| **推理** | 确定性——按合并顺序 apply | 确定性 | 概率性——可采样不同分词（子词正则化） |
| **使用者** | GPT-2/3/4, Llama, Mistral, Qwen | BERT, DistilBERT | T5, mBART, ALBERT, Gemma |
| **中文** | 逐字起步——基础单元是汉字 | 同 BPE | 逐字起步 + 概率剪枝 |

### 字节级 BPE——`[UNK]` 的终结者

基础词表 = 256 个可能的字节（而非 Unicode 字符）。这确保了任何输入——任何语言、任何 emoji、任何二进制数据——都能被编码为已知的 token 序列。GPT-2 的词表 = 256 个字节 + 50,000 次合并 + 1 个特殊 token = 50,257。

### SentencePiece vs tiktoken

| | SentencePiece | tiktoken |
|---|---|---|
| **角色** | **训练**词表 + 编码 | **编码**（使用已训练好的词表） |
| **输入** | 原始 Unicode 文本——不需要预分词 | 预构建的词表文件 |
| **空白处理** | 将空格编码为 `▁`（U+2581） | 基于字节——空白本身是字节序列的一部分 |
| **中文友好度** | 高——直接操作于 Unicode 字符流，对无空格语言天然友好 | 高——但针对英文做了更多优化 |

---

## 3. 中文子词分词的特殊考虑

- **以字为基础的 BPE 对中文天然合理。** 中文的"字"已经是比"字母"更高的语义单元——不像英文需要从 26 个字母逐步合并。"机器学习"作为 4 个字 → BPE 合并"机器"+"学习"只需要 2 次合并（英文字串可能需要 10+ 次）
- **中文 tokenizer 的词表大小选择：** 32K 词表在中文上相当于英文 50K 的效果——因为中文 token 的"信息密度"更高。"一个汉字 ≈ 一个英文词"在 BPE 的词表效率统计中大致成立
- **不要混用简繁 tokenizer。** 如果训练 tokenizer 的语料是简体中文 → 繁体输入会被切得非常碎（每个繁体字被当作一个独立 token 或分解为多个字节）。如果需要繁简同时支持 → 在 tokenizer 训练语料中混合简繁文本

---

## 4. 选择决策树

```
从头训练自己的 LLM？
├─ 是 → SentencePiece BPE 或 Unigram（推荐 Unigram：子词正则化对数据增强有价值）
└─ 否 → 微调现有模型？
         └─ 用模型出厂 tokenizer。永远不要换。

只是编码文本做分类/检索？
├─ 英文为主 → tiktoken (cl100k_base, o200k_base)
├─ 中文为主 → 加载对应 LLM 的 tokenizer（Qwen/Llama tokenizer）
└─ 多语言   → NLLB tokenizer 或 XLM-R tokenizer
```

---

## 参考资料

1. Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units". ACL, 2016. (BPE for NLP)
2. Kudo and Richardson. "SentencePiece: A simple and language independent subword tokenizer". EMNLP, 2018.

---

> 本课程参考了 AI Engineering From Scratch 的课程体系，在此基础上进行了重构和原创内容的扩充。中文子词分词分析为原创内容。
