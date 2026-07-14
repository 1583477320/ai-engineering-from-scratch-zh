# 分词器——BPE、WordPiece、SentencePiece

> 大语言模型不认识中文，也不认识英文。它只认识整数。分词器决定了这些整数是在承载含义，还是在浪费算力。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05（NLP 基础）| **时间：** ~90 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 02（构建分词器）— 从库使用到从零训练 | 阶段 10 · 03（数据管道）— 分词是预处理的第一步

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 BPE、WordPiece 和 Unigram 分词算法——对比它们的合并策略
- [ ] 解释词表大小如何影响模型效率——太小导致序列变长，太大浪费嵌入参数
- [ ] 分析不同语言和代码的分词效率——找出特定分词器失效的地方
- [ ] 使用 tiktoken 和 sentencepiece 库对文本进行分词并检查生成的词元 ID

---

## 1. 问题

你的大语言模型不认识中文。它不认识任何语言。它只认识数字。

从 "你好，世界！" 到 [15496, 11, 995, 0] 之间的桥梁，就是分词器。每个字、每个空格、每个标点符号，在进入模型之前都必须被转换成一个整数。这个转换不是中性的——它会将某些假设"烧结"到模型中，后续再也无法撤销。

做得不好，你的模型会把算力浪费在编码常见词上。"不幸的是"变成了 4 个词元，而不是 1 个。你的 128K 上下文窗口对中文用户来说实际只有 30K。做好这件事，同样的上下文窗口能容纳两倍的有效信息。

你在 ChatGPT 中输入的每一个中文字、调用 API 的每一笔费用、模型生成的每一个字——都由分词器决定。分词不是预处理。**分词是架构。**

---

## 2. 概念

### 2.1 三种方案——两种失败，一种胜出

| 方案 | 原理 | 优点 | 缺点 | 结果 |
|------|------|------|------|------|
| **词级分词** | 按空格和标点拆分 | 简单 | 词表爆炸，OOV 多 | ❌ 失败 |
| **字符级分词** | 每个字符一个 token | 无 OOV | 序列太长，效率低 | ❌ 失败 |
| **子词分词** | 常见词完整，罕见词拆分 | 词表可控，无 OOV | 需要训练 | ✅ 胜出 |

子词分词是所有现代 LLM 的选择。GPT-2、GPT-4、BERT、Llama 3、Claude——全部使用子词分词。

### 2.2 BPE——字节对编码

BPE 是一个贪婪压缩算法。从单个字符开始。统计训练语料中每个相邻字符对的频率。合并最频繁的对为新 token。重复直到达到目标词表大小。

```
语料（含频率）：
  "lower"  ×5
  "lowest" ×2
  "newest" ×6

步骤 0: 从字符开始
  l o w e r      (×5)
  l o w e s t    (×2)
  n e w e s t    (×6)

步骤 1: 统计相邻对
  (e,s): 8    (s,t): 8    (l,o): 7    (o,w): 7
  (w,e): 13   (e,r): 5    (n,e): 6    ...

步骤 2: 合并最频繁对 (w,e) → "we"
  l o we r       (×5)
  l o we s t     (×2)
  n e we s t     (×6)

步骤 3: 合并 (we,s) → "wes" → 继续...

步骤 4: 合并 (wes,t) → "west"
  l o we r       (×5)
  l o west       (×2)
  n e west       (×6)

...重复直到目标词表大小
```

**合并表就是分词器。** 编码新文本时，按学习的顺序应用合并。训练语料决定了哪些合并存在——这个选择永久塑造了模型"看到"的世界。

### 2.3 字节级 BPE（GPT-2, GPT-4）

标准 BPE 在 Unicode 字符上操作。字节级 BPE 在原始字节（0-255）上操作。基础词表正好 256 个条目。任何输入——文本、二进制、损坏的——都可以被分词，永远不产生未知 token。

GPT-2 引入了这种方法。OpenAI 的 tiktoken 库实现了字节级 BPE，词表大小：

- GPT-2: 50,257 词元
- GPT-3.5/GPT-4: ~100,256 词元（cl100k_base）
- GPT-4o: 200,019 词元（o200k_base）

### 2.4 WordPiece（BERT）

WordPiece 看起来像 BPE 但合并策略不同。不是看原始频率，而是最大化训练数据的似然：

```
BPE 合并标准：  count(A, B)
WordPiece 合并标准：count(AB) / (count(A) × count(B))
```

BPE 问："哪个对出现最频繁？" WordPiece 问："哪个对的共现比随机预期更频繁？" 这个微妙的区别产生了不同的词表。

WordPiece 使用 "##" 前缀表示续接子词：`"unhappiness" → ["un", "##happi", "##ness"]`。BERT 使用 WordPiece，词表 30,522。

### 2.5 SentencePiece（Llama, T5）

SentencePiece 将输入视为原始 Unicode 字符流，包括空格。没有预分词步骤。没有语言特定的词边界规则。对中文、日文、泰文等无空格分词的语言真正语言无关。

支持两种算法：
- **BPE 模式**：与标准 BPE 相同的合并逻辑
- **Unigram 模式**：从大词表开始，迭代移除对似然影响最小的 token——BPE 的反向操作

### 2.6 词表大小的权衡

| 模型 | 词表大小 | 分词器类型 | 平均每个英文词的 token 数 |
|------|---------|-----------|------------------------|
| BERT | 30,522 | WordPiece | ~1.4 |
| GPT-2 | 50,257 | 字节级 BPE | ~1.3 |
| Llama 2 | 32,000 | SentencePiece BPE | ~1.4 |
| GPT-4 | ~100,256 | 字节级 BPE | ~1.2 |
| Llama 3 | 128,256 | 字节级 BPE | ~1.1 |
| GPT-4o | 200,019 | 字节级 BPE | ~1.0 |

**趋势：词表大小在增长。** 更大的词表压缩更激进，推理更快，但嵌入层参数更多。

### 2.7 多语言税

以英文为主训练的分词器对其他语言非常不友好。韩文在 GPT-2 分词器中平均每词 2-3 个 token。中文可能更糟。这意味着一个中文用户实际上拥有英文用户一半大小的上下文窗口——付同样的钱但信息密度更低。

这就是为什么 Llama 3 将词表从 32K 翻了四倍到 128K——更多 token 分配给非英文脚本，跨语言压缩更公平。

---

## 3. 从零实现

### 第 1 步：字符级分词器（基线）

```python
class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]
    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
```

"hello" 变成 [104, 101, 108, 108, 111]。每个字符一个 token。这是我们改进的基线。

### 第 2 步：从零实现 BPE

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
        return self

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")
```

### 第 3 步：编码-解码往返测试

```python
corpus = "The cat sat on the mat. The cat ate the rat."
tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=40)

for sentence in ["The cat sat on the mat.", "unhappiness", "tokenization"]:
    encoded = tokenizer.encode(sentence)
    decoded = tokenizer.decode(encoded)
    print(f"'{sentence}' → {len(encoded)} tokens → 往返: {'通过' if decoded == sentence else '失败'}")
```

### 第 4 步：与 tiktoken 对比

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

texts = ["The cat sat on the mat.", "unhappiness", "Hello, world!"]
for text in texts:
    our = tokenizer.encode(text)
    tik = enc.encode(text)
    print(f"'{text}' → 我们: {len(our)} tokens, tiktoken: {len(tik)} tokens")
```

tiktoken 使用相同的算法但训练在数百 GB 文本上——100K 次合并。你训练 40 次合并的玩具分词器无法与之竞争。但机制完全相同。

---

## 4. 工具

### 4.1 tiktoken（OpenAI）

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Tokenizers convert text to integers")
print(f"Tokens: {tokens}")
print(f"Pieces: {[enc.decode([t]) for t in tokens]}")
```

tiktoken 用 Rust 写成，Python 绑定。每秒编码数百万词元。

### 4.2 Hugging Face tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>", "<unk>"])
tokenizer.train(["corpus.txt"], trainer=output)
```

Rust 底层——在 GB 级语料上训练只需秒级。

### 4.3 加载 Llama 的分词器

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokens = tok.encode("Tokenizers are the unsung heroes of LLMs")
print(f"Token IDs: {tokens}")
print(f"词表大小: {tok.vocab_size}")  # 128,256
```

### 4.4 工具对比

| 工具 | 算法 | 速度 | 适用场景 |
|------|------|------|---------|
| tiktoken | 字节级 BPE | 极快 | GPT 系列推理 |
| HF tokenizers | BPE/WordPiece/Unigram | 极快 | 自定义分词器训练 |
| SentencePiece | BPE/Unigram | 快 | 多语言分词 |
| 从零实现 BPE | BPE | 慢 | 教学理解 |

---

## 5. LLM 视角

### 5.1 分词器在大语言模型中的体现

分词器直接影响：
- **计费**：GPT-4 API 按词元收费——同样文本，更好分词器更便宜
- **上下文容量**：中文在低质量分词器中可能只有一半的等效上下文
- **生成速度**：每步生成一个词元——词表越大每步越快，但嵌入层更重

### 5.2 分词器是 LLM 的"眼睛"

分词器的质量决定了模型"看到"的世界。如果中文被拆成单个字（每个字 3 个 UTF-8 字节 → 3 个 token），模型必须学习将这些碎片重新组合。如果中文被作为完整词处理，模型直接"看到"语义单元。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你让 Claude 数 "strawberry" 中有几个 'r' 时，它经常数错。这不是因为它不聪明——**是因为它看到的不是字母，而是词元**。"strawberry" 可能被分成了 ["st", "raw", "berry"]，它根本看不到单个的 'r'。理解这一点，你就理解了 LLM 的很多"愚蠢"行为其实是分词器的锅。

---

## 6. 工程最佳实践

### 6.1 词表大小选择

| 场景 | 推荐词表大小 | 原因 |
|------|------------|------|
| 资源受限 | 8K-16K | 嵌入层参数少 |
| 通用模型 | 32K | GPT-2/LLama 2 标准 |
| 多语言 | 64K-128K | 覆盖更多语言 |
| 极致推理 | 128K-200K | 每词更少 token，推理更快 |

### 6.2 中文场景特别建议

- 训练分词器时，确保中文语料占比 ≥ 30%，否则中文会被过度切分
- 使用 `bert-base-chinese` 时注意：它是字级的，不产生子词
- 中文+代码混合场景，优先使用 Llama 3 或 GPT-4 的分词器（128K+ 词表）

### 6.3 踩坑经验

- **训练分词器时未设 byte_fallback=True**：部署时遇到生僻字直接报错
- **padding_side="left" 还是 "right"**：自回归模型用 left，编码器模型用 right
- **不要用训练集平均长度作为 max_length**：留 20% 余量，否则截断率太高

---

## 7. 常见错误

### 错误 1：忘记除以 √d_k

**现象：** 训练初期 loss 不下降，梯度为 NaN。

**修复：** 确保 Attention 计算时除以 √d_k。

### 错误 2：分词器词表与模型嵌入层不匹配

**现象：** 加载模型报维度错误。

**原因：** 分词器的词表大小必须与模型嵌入矩阵的行数完全一致。

```python
# ❌ 词表不匹配
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 词表 50257
model = GPT2LMHeadModel.from_pretrained("gpt2", vocab_size=30000)  # 嵌入层 30000

# ✓ 保持一致
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")  # 自动匹配
```

---

## 8. 面试考点

### Q1：为什么所有现代 LLM 都使用子词分词？（难度：⭐⭐）

**参考答案：**
词级分词需要巨大词表且无法处理未登录词。字符级分词序列太长——"unhappiness" 需要 11 个 token，模型必须消耗注意力容量学习字符组合。子词分词取两者之长：常见词完整保留（"the"=1 token），罕见词拆分为有意义的子词（"unhappiness"=3 token）。词表可控（32K-200K），序列长度合理，无 OOV 问题。

### Q2：BPE 和 WordPiece 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
BPE 合并最频繁的相邻对——问"哪个对出现最频繁？" WordPiece 合并似然比最高的对——问"哪个对的共现比随机预期更频繁？"。这导致 WordPiece 更偏好有语言学意义的合并（如 "un" + "happiness"），而 BPE 可能合并纯统计上的高频对。

### Q3：词表大小的选择如何影响 LLM 的效率？（难度：⭐⭐⭐）

**参考答案：**
词表大小影响三个维度：(1) 嵌入层参数量——128K 词表 + 4096 维 = 524M 参数，32K 词表 = 131M 参数，差 400M；(2) 序列长度——更大词表将更多常见词编码为单个 token，序列更短，注意力计算 O(n²) 减少；(3) 推理速度——每步生成一个词元，序列更短意味着更少的步数。趋势是增大词表——GPT-2(50K)→GPT-4(100K)→GPT-4o(200K)。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 词元 (Token) | "就是一个词" | 模型词表中的一个单元——可以是字符、子词、词、甚至多词片段 |
| BPE | "一种压缩算法" | 字节对编码——反复合并最频繁的相邻词元对，直到达到目标词表大小 |
| WordPiece | "BERT 的分词器" | 像 BPE 但合并标准是 count(AB)/(count(A)×count(B))——最大化似然 |
| SentencePiece | "一个分词器库" | 语言无关的分词器——在原始 Unicode 上操作，支持 BPE 和 Unigram 算法 |
| 词表大小 | "模型认识多少词" | 所有不重复词元的总数——GPT-2 有 50,257 个，Llama 3 有 128,256 个 |
| 多语言税 | "非英文用得贵" | 分词器以英文为主训练时，中文/韩文/阿拉伯文需要 2-5 倍词元表达同样的意思 |
| 字节级 BPE | "GPT 的分词器" | 在原始字节（0-255）上操作的 BPE——保证任何输入都不产生未知词元 |
| 肥度 (Fertility) | "每个词多少个 token" | 输出词元数/输入词数——1.0 是完美，3.0 意味着模型三倍辛苦 |

---

## 📚 小结

分词器 = 文字→数字的桥梁。BPE 是最常用的算法——从字符开始迭代合并。字节级 BPE 保证任何语言都不产生未知 token。词表大小是关键权衡——更大词表推理更快但嵌入层更大。多语言税让中文用户在低质量分词器上吃亏。理解分词器就是理解 LLM "看到"的世界。下一课你将从零构建一个生产级分词器。

---

## ✏️ 练习

1. **【实现】** 从零实现 BPE 分词器（本课的 `BPETokenizer`）。打印每步合并过程——观察 "t" + "h" → "th" → "the" 是如何逐步形成的。
2. **【实现】** 添加特殊 token `<pad>`、`<eos>`、`<unk>`，分配 ID 0、1、2。实现预分词步骤——在 BPE 之前按空格拆分。
3. **【实验】** 实现 WordPiece 合并标准。在同一语料上分别训练 BPE 和 WordPiece（相同合并次数），对比生成的词表——哪个产生更有语言学意义的子词？
4. **【实验】** 构建多语言分词器效率基准：取 10 句英文、中文、韩文、阿拉伯文。用 tiktoken (cl100k_base) 分词，测量平均词元/字符比。量化每种语言的"多语言税"。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| BPE 分词器 | `code/bpe.py` | 从零实现的字节级 BPE 分词器 |
| 分词效率分析器 | `outputs/prompt-tokenizer-analyzer.md` | 分析任意文本和模型分词效率的提示词 |

---

## 📖 参考资料

1. [论文] Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units". ACL, 2016. https://arxiv.org/abs/1508.07909 — 将 BPE 引入 NLP 的开创性论文
2. [论文] Kudo & Richardson. "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing". EMNLP, 2018. https://arxiv.org/abs/1808.06226
3. [GitHub] OpenAI tiktoken: https://github.com/openai/tiktoken — GPT-3.5/4 使用的 Rust BPE 实现
4. [官方文档] Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers
5. [论文] Wu et al. "Google's Neural Machine Translation System". arXiv, 2016. https://arxiv.org/abs/1609.08144 — WordPiece 的应用

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
