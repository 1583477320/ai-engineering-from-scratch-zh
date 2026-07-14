# 数据管道——从原始文本到训练批次

> 模型是镜子。它反映出你喂给它的任何东西。喂垃圾，它会用完美的流畅度反映出垃圾。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-02（分词器）| **时间：** ~90 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 01-02（分词器）— 分词是管道的第一步 | 阶段 10 · 04（预训练）— 管道的输出供预训练使用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建流式数据管道——分词、分块、打乱、批处理 TB 级文本而无需全部加载到内存
- [ ] 实现数据质量过滤——去重、语言检测、内容过滤
- [ ] 创建固定长度训练序列——正确的注意力掩码和文档边界处理
- [ ] 分析管道吞吐量——确保数据加载器不会拖慢 GPU 训练速度

---

## 1. 问题

你有了分词器。现在你需要数据。

Llama 3 在 15 万亿 token 上预训练。这些文本无法全部加载到内存。你不能等 10 分钟加载一批数据。你需要一个能流式处理、边读边处理、边处理边喂给 GPU 的管道。

预训练数据管道的四个核心操作：

```
原始文本 → [分词] → [分块] → [打乱] → [批处理] → GPU 训练
```

管道必须保证：
- **连续性**：模型看到的每条训练序列 = 固定长度的 token 序列（如 2048）
- **文档边界**：不同文档之间用特殊 token（`<EOS>`）分隔
- **随机性**：打乱顺序防止模型记住局部模式
- **吞吐量**：每秒处理的 token 数 ≥ GPU 每秒消耗的 token 数

---

## 2. 概念

### 2.1 流式处理 vs 批处理

| 方法 | 适用场景 | 内存需求 | 吞吐量 |
|------|---------|---------|--------|
| **批处理** | 小数据集（<10GB） | O(数据大小) | 低 |
| **流式处理** | 大数据集（>100GB） | O(1) | 高 |
| **内存映射** | 中等数据集 | O(索引大小) | 中 |

Llama 3 使用流式处理——每次只读取一个 chunk（如 10MB），分词后丢弃原始文本。

### 2.2 数据质量过滤

Llama 3 使用的过滤规则：

```python
def quality_filter(text, min_words=10, max_ratio=0.3):
    """基本质量过滤。"""
    words = text.split()
    # 太短
    if len(words) < min_words:
        return False
    # 特殊字符比例太高（可能是乱码）
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special / len(text) > max_ratio:
        return False
    # 重复内容检测（简单版）
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        return False  # 太多重复
    return True
```

### 2.3 去重

重复数据会让模型过拟合——在 Common Crawl 上，约 30-60% 的内容是重复的。去重方法：
- **精确去重**：哈希整个文档
- **模糊去重**：MinHash/LSH——检测近似重复
- **n-gram 去重**：检测子串级别的重复

### 2.4 固定长度分块

```python
def chunk_tokens(token_ids, chunk_size=2048, eos_id=2):
    """将 token 序列分块为固定长度。"""
    chunks = []
    current = []
    for token_id in token_ids:
        current.append(token_id)
        if len(current) >= chunk_size:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks
```

### 2.5 注意力掩码

因果语言模型需要下三角注意力掩码——每个位置只能看到之前的位置：

```python
def create_causal_mask(seq_len):
    """创建因果掩码。"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
    return mask
```

### 2.6 吞吐量分析

GPU 训练的瓶颈可能在数据加载。分析指标：
- **每秒处理 token 数**：目标 > GPU 消耗速度
- **GPU 利用率**：如果 < 80%，可能数据加载太慢
- **num_workers**：数据加载进程数，太多会竞争内存

---

## 3. 从零实现

### Step 1：流式文本读取

```python
def stream_text(file_path, chunk_size=1024*1024):
    """流式读取文本文件——每次读取 1MB。"""
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
```

### Step 2：分词 + 分块

```python
def tokenize_and_chunk(file_path, tokenizer, chunk_size=2048):
    """流式分词并分块。"""
    buffer = []
    for text_chunk in stream_text(file_path):
        tokens = tokenizer.encode(text_chunk)
        buffer.extend(tokens)
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            yield {"input_ids": chunk, "labels": chunk[1:] + [0]}
```

### Step 3：创建 PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            "input_ids": torch.tensor(chunk["input_ids"], dtype=torch.long),
            "labels": torch.tensor(chunk["labels"], dtype=torch.long),
        }
```

### Step 4：数据质量过滤

```python
import hashlib

def deduplicate(files, seen_hashes=None):
    """精确去重——基于文档哈希。"""
    if seen_hashes is None:
        seen_hashes = set()
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            yield content
```

---

## 4. 工具

### 4.1 HuggingFace Datasets

```python
from datasets import load_dataset

# 加载并流式处理
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", streaming=True)

for sample in dataset["train"]:
    text = sample["text"]
    # 分词、分块...
    if len(text) < 10:
        continue  # 过滤太短的文本
```

### 4.2 Ray Data

```python
import ray

# 大规模分布式数据处理
ds = ray.data.read_text("s3://my-bucket/corpus/*.jsonl")
ds = ds.map(lambda row: {"tokens": tokenizer.encode(row["text"])})
ds = ds.flat_map(lambda row: chunk_tokens(row["tokens"]))
```

### 4.3 工具对比

| 工具 | 适用场景 | 特点 |
|------|---------|------|
| HuggingFace Datasets | 单机/小规模 | 简单易用，流式支持 |
| Ray Data | 分布式 | 自动并行，容错 |
| WebDataset | 流式 | Tar 格式，I/O 优化 |

---

## 5. LLM 视角

### 5.1 数据质量对 LLM 的影响

- **"垃圾进，垃圾出"**：模型会反映训练数据中的偏见、错误、低质量内容
- **去重至关重要**：重复数据让模型"死记硬背"——在基准测试上得分高但泛化差
- **数据混合比例**：英文/代码/数学的比例直接影响模型能力

### 5.2 Llama 3 的数据管道

15 万亿 token 的处理流程：
1. 原始爬取 → 语言检测（过滤非英文）→ 质量过滤
2. 去重（MinHash）→ 敏感内容过滤 → 领域分类
3. 分词 → 分块 → 打乱 → 批处理
4. 混合比例：网页 50%、代码 25%、学术 15%、其他 10%

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你输入"给我写一首诗"时，模型能写出来——是因为它在训练数据中见过大量诗歌。数据管道决定了模型"见过什么"。如果训练数据中没有中文诗歌，模型就写不好中文诗。

---

## 6. 工程最佳实践

### 6.1 管道优化

- **num_workers=4-8**：每个 worker 独立加载和分词
- **pin_memory=True**：加速 CPU 到 GPU 的数据传输
- **prefetch_factor=2**：预加载下一个 batch
- **使用 numpy 做分词**：比 Python 循环快 10-50 倍

### 6.2 中文场景特别建议

- 中文文本需要 UTF-8 编码处理
- 中文无空格分词——使用 SentencePiece 或 BPE
- 中文数据去重可能需要 jieba 分词后做 n-gram 去重

### 6.3 踩坑经验

- **文档边界丢失**：不加 `<EOS>` 会导致模型跨文档生成
- **填充 token 泄漏**：`<PAD>` 作为输入时不应计算损失
- **数据泄露**：测试集和训练集有重叠——严格按时间/文档去重

---

## 7. 常见错误

### 错误 1：不处理文档边界

**现象：** 模型在生成时跨文档拼接——从一个文档续写另一个文档的内容。

**修复：** 在每个文档末尾添加 `<EOS>` token，分块时不跨文档边界。

### 错误 2：padding token 参与损失计算

```python
# ❌ 错误：所有位置都计算损失
loss = F.cross_entropy(logits, labels)

# ✓ 正确：忽略 padding 位置
loss = F.cross_entropy(logits, labels, ignore_index=pad_token_id)
```

### 错误 3：数据加载成为瓶颈

**现象：** GPU 利用率只有 30-50%——大部分时间在等数据。

**修复：** 增加 num_workers、启用 pin_memory、预处理并缓存分词结果。

---

## 8. 面试考点

### Q1：为什么预训练需要去重？（难度：⭐⭐）

**参考答案：**
重复数据有两个危害：(1) 模型会过拟合——对重复内容"死记硬背"，在基准测试上得分高但泛化差；(2) 训练效率降低——重复样本没有提供新信息但消耗相同计算。Common Crawl 上约 30-60% 内容是重复的。Llama 3 使用 MinHash 做模糊去重——检测近似重复而非完全相同。

### Q2：如何确保数据管道不成为训练瓶颈？（难度：⭐⭐⭐）

**参考答案：**
(1) 流式处理——不加载全部数据到内存，每次只读取一个 chunk；(2) 多进程加载——num_workers=4-8 并行加载和分词；(3) 预处理并缓存——分词结果存为 numpy/memmap 格式，避免重复分词；(4) pin_memory=True——加速 CPU→GPU 传输；(5) 监控 GPU 利用率——如果 <80%，说明数据加载太慢。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 流式处理 | "边读边处理" | 每次只读取一个 chunk（如 10MB），处理后丢弃——内存占用 O(1) |
| 去重 | "删除重复" | 检测并移除重复/近似重复文档——防止模型过拟合 |
| 分块 | "切成固定长度" | 将 token 序列切分为固定长度（如 2048）的训练样本 |
| 因果掩码 | "让模型只能往前看" | 下三角掩码——位置 i 只能关注 0..i |
| 文档边界 | "文档之间怎么分" | 用 `<EOS>` token 标记文档结束——防止跨文档生成 |
| 吞吐量 | "数据管道多快" | 每秒处理的 token 数——需 ≥ GPU 消耗速度 |

---

## 📚 小结

数据管道是预训练的基础设施——流式读取、分词、分块、打乱、批处理。数据质量过滤（去重、语言检测、内容过滤）直接影响模型质量。固定长度分块 + 文档边界处理确保训练的正确性。吞吐量必须匹配 GPU 消费速度——否则 GPU 在空等数据。下一课你将在这些数据上训练一个 Mini GPT。

---

## ✏️ 练习

1. **【实现】** 构建流式数据管道：读取一个文本文件，分词、分块为 2048 token 的序列，输出 PyTorch DataLoader。
2. **【实验】** 在 Wikipedia 子集上实现 MinHash 去重——对比去重前后的训练 loss 收敛速度。
3. **【实验】** 测量不同 num_workers（0, 1, 4, 8）下的数据加载吞吐量，画出吞吐量 vs workers 曲线。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 流式数据管道 | `code/data_pipeline.py` | 流式读取 + 分词 + 分块 + DataLoader |

---

## 📖 参考资料

1. [论文] Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models". arXiv, 2023. https://arxiv.org/abs/2307.09288 — 数据管道细节
2. [官方文档] HuggingFace Datasets: https://huggingface.co/docs/datasets
3. [论文] Lee et al. "Deduplicating Training Data Makes Language Models Better". ACL, 2022. https://arxiv.org/abs/2107.06499

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
