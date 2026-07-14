# 构建分词器——从零训练

> 第 01 课给你一个玩具。这一课给你一个武器——处理 Unicode、空白规范化、特殊 token 的生产级分词器。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01（分词器）| **时间：** ~90 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 01（分词器原理）— 从理解到实现 | 阶段 10 · 03（数据管道）— 分词器是数据管道的第一步

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建一个生产级 BPE 分词器——处理 Unicode、空白规范化、特殊 token
- [ ] 实现字节级回退——分词器可以编码任何输入（包括 emoji、中文、代码）而没有未知 token
- [ ] 添加预分词正则表达式——在 BPE 合并之前按词边界拆分文本
- [ ] 在语料上训练自定义分词器并评估压缩率——对比 tiktoken 在多语言文本上的表现

---

## 1. 问题

第 01 课的 BPE 分词器可以处理英文。现在扔给日文试试。或者 emoji。或者混合了制表符和空格的 Python 代码。它坏了。

不是因为 BPE 有错——是因为实现不完整。生产级分词器要处理任意编码的原始字节、在拆分前规范化 Unicode、管理永不合并的特殊 token、将预分词与子词拆分串联——且足够快不拖慢训练管道。

GPT-2 有 50,257 个词元。Llama 3 有 128,256 个。那些词表背后的合并表训练在数百 GB 文本上。围绕它的机器——规范化、预分词、特殊 token 注入、聊天模板——才是区分玩具和武器的关键。

---

## 2. 概念

### 2.1 五阶段流水线

| 阶段 | 功能 | 为什么重要 |
|------|------|-----------|
| 规范化 | NFKC Unicode、可选小写化 | "fi" 连字变成两个字符——不做这个同词不同 token |
| 预分词 | BPE 之前按词边界拆分 | 防止 BPE 跨词边界合并 |
| BPE 合并 | 应用学习到的合并规则 | 核心压缩——字节变子词 |
| 特殊 token | 注入 BOS/EOS/PAD/聊天标记 | 固定 ID，永不参与合并 |
| ID 映射 | token 字符串到整数 ID | 模型看到整数不是字符串 |

### 2.2 字节级 BPE

将每个字节值（0-255）视为有效 token。基础词表正好 256 个条目。任何文件都可以被分词——永远不产生未知 token。

中文字符是 3 个 UTF-8 字节。日文 3-4 字节。emoji 4 字节。BPE 在这些字节序列中发现模式的方式与英文 ASCII 完全相同。**字节是字节。**

### 2.3 预分词

GPT-2 使用正则表达式按词边界拆分——缩写、单词、数字、标点、空白。前导空格保留在词上。Llama 使用 SentencePiece 跳过正则——将原始字节流视为一个长序列，让 BPE 自己决定边界。

### 2.4 特殊 token

特殊 token 有固定 ID，永不参与 BPE 合并——在合并之前精确匹配并替换为固定 ID。

### 2.5 聊天模板

模型看到的是平铺的 token 序列，不是 JSON。每个模型的聊天模板格式不同——格式错误就产出垃圾。

### 2.6 速度

Python 太慢。tiktoken（OpenAI）用 Rust 写成，HuggingFace tokenizers 也是 Rust。实现 10-100 倍加速。

### 2.7 词表大小的权衡

| 词表大小 | 序列长度 | 嵌入层参数 | 推荐场景 |
|----------|---------|-----------|---------|
| 8K | 长 | 小 | 资源受限 |
| 32K | 短 | 大 | GPT/LLaMA 级别 |
| 64K-128K | 更短 | 更大 | 多语言、高性能 |

---

---

## 3. 从零实现

### 第 1 步：字节级编码

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

### 第 2 步：BPE 训练核心

```python
from collections import Counter

def train_bpe(text, num_merges):
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for i in range(num_merges):
        pairs = Counter()
        for j in range(len(tokens) - 1):
            pairs[(tokens[j], tokens[j + 1])] += 1
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_id = 256 + i
        # 应用合并
        merged = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best:
                merged.append(new_id)
                j += 2
            else:
                merged.append(tokens[j])
                j += 1
        tokens = merged
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

    return merges, vocab
```

### 第 3 步：HuggingFace tokenizers 库（推荐方式）

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

### 第 4 步：多语言测试

```python
test_texts = [
    "Hello World",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
]
for text in test_texts:
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {len(ids)} tokens -> {ids.tokens}")
```

---

## 4. 工具

### 4.1 HuggingFace tokenizers（推荐）

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>"])
tokenizer.train(["corpus.txt"], trainer)
```

Rust 底层——在 GB 级语料上训练只需秒级。

### 4.2 SentencePiece

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="bpe_model",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,  # 中文推荐 0.9995
)
sp = spm.SentencePieceProcessor(model_file="bpe_model.model")
tokens = sp.encode("你好世界", out_type=str)
```

### 4.3 工具对比

| 工具 | 算法 | 速度 | 适用场景 |
|------|------|------|---------|
| tiktoken | 字节级 BPE | 极快 | GPT 系列推理 |
| HF tokenizers | BPE/WordPiece/Unigram | 极快 | 自定义分词器训练 |
| SentencePiece | BPE/Unigram | 快 | 多语言、跨平台 |

---

## 5. LLM 视角

### 5.1 分词器对 LLM 的影响

- **计费**：GPT-4 API 按词元收费——更好的分词器意味着更少词元、更少费用
- **上下文容量**：中文在低质量分词器中可能只有一半的等效上下文
- **训练效率**：词表大小直接影响嵌入层参数量（128K × 4096 = 524M）

### 5.2 为什么中文需要特别关注

GPT-2 词表 50K 中英文占绝大多数。中文字符被拆成 UTF-8 字节处理——每个汉字 3 个 token。中文用户的 128K 上下文实际只有 ~40K 有效容量。

### 5.3 聊天模板是关键

模型被训练在特定的聊天格式上。格式错误 = 输出垃圾。Llama 3、ChatGPT、Qwen 使用不同的特殊 token 和格式。

---

## 6. 工程最佳实践

### 6.1 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| vocab_size | 32K-128K | 根据语言需求选择 |
| min_frequency | 2-5 | 过滤低频对，防止过拟合 |
| character_coverage | 0.9995（中文） | 确保所有字符被覆盖 |

### 6.2 中文场景特别建议

- 使用 SentencePiece 并设置 character_coverage=0.9995
- 训练时中文语料占比 ≥ 30%
- 考虑字级初始词表——汉字作为独立 token

### 6.3 踩坑经验

- **忘记加特殊 token**：模型无法识别序列边界
- **padding_side 搞反**：自回归模型用 left，编码器模型用 right
- **词表不匹配**：分词器词表大小必须与模型嵌入层行数一致

---

## 7. 常见错误

### 错误 1：训练后忘记测试往返

```python
# 必须验证 encode → decode 往返正确
decoded = tokenizer.decode(tokenizer.encode(text))
assert decoded == text
```

### 错误 2：特殊 token 参与合并

特殊 token 应被精确匹配并固定 ID——绝不参与 BPE 合并。

---

## 8. 面试考点

### Q1：生产级分词器需要处理哪些边缘情况？（难度：⭐⭐）

**参考答案：**
(1) 未知字节序列——字节级 BPE 保证任何输入可编码；(2) Unicode 规范化——连字、全角字符、重音字符需要统一处理；(3) 特殊 token 不合并——BOS/EOS/聊天标记需要固定 ID；(4) 不同语言的词边界——中文无空格，需要字符级或 SentencePiece；(5) 聊天模板——每个模型格式不同，格式错误导致输出垃圾。

### Q2：GPT-2 的字节级 BPE 为什么能在所有语言上工作？（难度：⭐⭐⭐）

**参考答案：**
任何文本都可以编码为 UTF-8 字节序列——英文 1 字节/字符，中文 3 字节/字符，emoji 4 字节。字节级 BPE 的基础词表是 0-255 的 256 个字节值。BPE 合并在这些字节上操作——发现中文字符的 3 字节模式、英文单词的常见组合。算法不关心语言——它只看到字节序列。这使得同一个分词器可以处理任何语言、任何编码、甚至二进制数据。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 字节级 BPE | "在字节上做 BPE" | 基础词表 256 个字节值——处理任何输入无未知 token |
| 预分词 | "BPE 之前拆分" | 正则/规则拆分——防止 BPE 跨词边界合并 |
| NFKC 规范化 | "Unicode 清洗" | 标准分解+兼容组合——"fi"连字变"fi"两个字符 |
| 聊天模板 | "消息变 token 的方式" | 将 role/content 消息列表转换为平铺 token 序列的精确格式 |
| 特殊 token | "控制 token" | 绕过 BPE 的保留 ID——BOS、EOS、PAD、聊天标记 |
| 肥度 (Fertility) | "每个词多少 token" | 输出 token 数/输入词数——GPT-4 英文 ~1.3，韩文 2-3 |

---

## 📚 小结

生产级分词器是五阶段流水线——规范化、预分词、BPE 合并、特殊 token、ID 映射。字节级 BPE 处理任何语言无未知 token。特殊 token 永不参与合并。聊天模板必须精确匹配训练格式。训练自定义分词器用 HF tokenizers 库（Rust 底层），评估压缩率对比 tiktoken。

---

## ✏️ 练习

1. **【实现】** 为分词器添加 `get_token_bytes(id)` 方法——查看任意 ID 的原始字节。用它检查最常用的合并 token 实际代表什么。
2. **【实现】** 添加 Llama 风格的预分词——按空格和数字拆分但保留前导空格。对比 GPT-2 正则和 Llama 方法在同一语料上的词表差异。
3. **【实验】** 实现聊天模板方法——将 `{"role": ..., "content": ...}` 消息列表转换为 Llama 3 格式的 token 序列。与 HuggingFace 实现对比。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 完整分词器 | `code/tokenizer.py` | 字节级 BPE + 预分词 + 特殊 token + 聊天模板 |
| 构建提示词 | `outputs/prompt-tokenizer-builder.md` | 构建和调试生产分词器的提示词 |

---

## 📖 参考资料

1. [GitHub] OpenAI tiktoken: https://github.com/openai/tiktoken — Rust BPE 实现
2. [GitHub] HuggingFace tokenizers: https://github.com/huggingface/tokenizers — Rust 分词器库
3. [论文] Kudo & Richardson. "SentencePiece". EMNLP, 2018. https://arxiv.org/abs/1808.06226
4. [论文] Llama 3 Team. "The Llama 3 Herd of Models". arXiv, 2024. https://arxiv.org/abs/2407.21783 — 128K 词表细节
5. [GitHub] GPT-2 分词器源码: https://github.com/openai/gpt-2/blob/master/src/encoder.py

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
