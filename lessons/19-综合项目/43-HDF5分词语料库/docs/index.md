# 综合项目43——HDF5 分词语料库（HDF5 Tokenized Corpus）

> 下载的语料必须以训练器能以线速流式传输的布局落地。磁盘上的 JSONL 无法承受 16 个数据加载器工作进程。HDF5 带可调整大小的分块整数数据集可以。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将文档流式传输到可调整大小的 HDF5 整数数据集
- 跨多个 HDF5 文件分片写入，避免单点故障
- 通过 HDF5 的分块布局和页缓存实现高效读取
- 实现产生固定长度训练序列的滑动窗口数据加载器

---

## 1. 问题

现代大语言模型训练以每秒数十万样本的速度跨数十个工作进程读取词元。JSONL 在磁盘上无法承受第一次冷缓存页错误：JSON 解析器慢、文档边界不可寻址、搜索到特定样本需要扫描整个文件。

更具体地说，有三个问题：

**问题一：解析开销。** JSONL 的每一行都需要 JSON 解析——`json.loads()` 在每秒百万次调用的压力下成为瓶颈。HDF5 存储的是纯整数张量，读取时直接以 NumPy 数组形式返回，无需解析。

**问题二：随机访问。** 训练时数据加载器需要在全局词元流中随机选择起始位置。JSONL 中找第 N 个文档需要扫描 N 行（除非预先构建索引）。HDF5 支持 `dataset[start:stop]` 的 O(1) 切片访问。

**问题三：分片与容错。** 一个 200GB 的 HDF5 文件写入到一半时进程崩溃——整个文件可能不可读。正确的做法是分片写入，每个分片独立，通过清单文件索引。

---

## 2. 核心概念

### 2.1 HDF5 的核心优势

**HDF5**（层次化数据格式版本 5）是一种为大规模数值数据设计的文件格式和库：

- **分块存储**：数据集在内部被分成固定大小的块，读取时只需要从磁盘加载包含请求范围的块
- **可调整大小**：`maxshape=(None,)` 创建的数据集可以动态增长
- **内存映射读取**：读取时利用操作系统的页缓存，无需将整个文件加载到内存
- **并行 I/O**：多个工作进程可以同时从同一个文件读取不同范围

### 2.2 正确的可调整大小策略

词元数据集以 `maxshape=(None,)` 和固定 `chunks=(chunk_size,)` 创建。写入通过缓冲词元到 `chunk_size` 长度的 NumPy 数组进行。缓冲填满时，数据集恰好调整 `chunk_size` 并写入新范围。

```
写入流程：
文档流 → 分词器 → 词元 ID 列表 → 累积缓冲区
                                          ↓
                            缓冲区达到 chunk_size？
                           /                \
                          是                否
                         /                   \
          调整 HDF5 数据集大小 + chunk 写入    继续累积
```

### 2.3 分片写入

单个 HDF5 文件是单点故障。管道并行写入分片：每个输入分片产生一个 HDF5 输出分片。`shards.json` 索引记录每个分片的文件路径、词元计数、文档计数和词元的 sha256。

### 2.4 滑动窗口数据加载器

数据加载器是唯一知道训练序列长度的阶段。它在全局词元流中随机选择起始索引，读取 `window_size + 1` 个词元，返回 `(输入, 目标)` 对。

```python
# 训练序列长度 = window_size + 1（输入为前 window_size 个词元，目标为后 window_size 个词元偏移一位）
# 例如：window_size=4
# 词元流：[10, 20, 30, 40, 50, 60, 70, ...]
# 样本 1: 输入=[10,20,30,40], 目标=[20,30,40,50]
# 样本 2: 输入=[50,60,70,80], 目标=[60,70,80,90]
```

---

## 3. 从零实现

```python
"""HDF5 分词语料库——流式分词+分片写入+内存映射读取。"""
import json, hashlib, numpy as np
from dataclasses import dataclass, field
from typing import List

try:
    import h5py
except ImportError:
    h5py = None

BOUNDARY_TOKEN_ID = 0
TOKEN_DTYPE = np.uint16
CHUNK_SIZE = 8192


@dataclass
class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))


@dataclass
class HDF5ShardWriter:
    """HDF5 分片写入器。

    缓冲词元到 CHUNK_SIZE，批量调整数据集大小并写入。
    """
    path: str
    chunk_size: int = CHUNK_SIZE
    buffer: list = field(default_factory=list)
    tokens_written: int = 0

    def __post_init__(self):
        if h5py is None:
            return
        self.file = h5py.File(self.path, "w", libver="latest")
        self.ds = self.file.create_dataset(
            "tokens",
            shape=(0,),
            maxshape=(None,),
            dtype=TOKEN_DTYPE,
            chunks=(self.chunk_size,),
        )
        self.file.swmr_mode = True  # 单写入器多读取器模式

    def append(self, token_ids):
        """追加词元 ID 列表到 HDF5 数据集。"""
        if h5py is None:
            return
        self.buffer.extend(token_ids)
        while len(self.buffer) >= self.chunk_size:
            chunk = np.array(self.buffer[:self.chunk_size], dtype=TOKEN_DTYPE)
            old_size = self.ds.shape[0]
            self.ds.resize(old_size + self.chunk_size)
            self.ds[old_size:old_size + self.chunk_size] = chunk
            self.tokens_written += self.chunk_size
            self.buffer = self.buffer[self.chunk_size:]
            self.ds.flush()

    def close(self):
        if h5py is None:
            return
        if self.buffer:
            chunk = np.array(self.buffer, dtype=TOKEN_DTYPE)
            old_size = self.ds.shape[0]
            self.ds.resize(old_size + len(chunk))
            self.ds[old_size:old_size + len(chunk)] = chunk
            self.tokens_written += len(chunk)
        self.ds.attrs["token_count"] = self.tokens_written
        sha = hashlib.sha256(np.array(self.ds[:self.tokens_written]).tobytes()).hexdigest()
        self.ds.attrs["sha256"] = sha
        self.file.close()


class SlidingWindowLoader:
    """从 HDF5 数据集产生固定长度序列的滑动窗口加载器。"""

    def __init__(self, h5_path: str, window_size: int):
        self.file = h5py.File(h5_path, "r", swmr=True)
        self.tokens = self.file["tokens"]
        self.window_size = window_size
        self.total_tokens = self.tokens.shape[0]

    def __len__(self):
        return max(0, self.total_tokens - self.window_size)

    def __getitem__(self, idx):
        """返回 (输入, 目标) 对，每个长度为 window_size。"""
        if idx + self.window_size + 1 > self.total_tokens:
            raise IndexError("超出范围")
        chunk = self.tokens[idx:idx + self.window_size + 1]
        return chunk[:self.window_size], chunk[1:self.window_size + 1]


def main():
    if not h5py:
        print("需要 h5py：pip install h5py")
        return 1

    tok = ByteTokenizer()
    docs = [
        "hello world training corpus document one",
        "second document with different content",
        "third document about machine learning",
    ]
    path = "/tmp/test_shard.h5"
    writer = HDF5ShardWriter(path, chunk_size=8)
    for doc in docs:
        ids = tok.encode(doc)
        writer.append(ids + [BOUNDARY_TOKEN_ID])
    writer.close()

    with h5py.File(path, "r") as f:
        total = f["tokens"].attrs["token_count"]
        sha = f["tokens"].attrs["sha256"][:16]
        print(f"词元总数: {total}")
        print(f"sha256: {sha}...")

    loader = SlidingWindowLoader(path, window_size=4)
    for i in range(min(3, len(loader))):
        inp, tgt = loader[i]
        print(f"样本 {i}: 输入={inp.tolist()} 目标={tgt.tolist()}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

## 4. 工业工具

### 4.1 HDF5 生产配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Chunk size | 4096-16384 | 太小增加元数据开销，太大浪费 I/O |
| 压缩 | gzip level 4 | 300-500MB/s 解压速度，减少磁盘空间约 30-40% |
| Swmr | True | 单写入器多读取器模式，训练时不需要锁 |

### 4.2 替代方案对比

| 方案 | 随机访问 | 解析开销 | 分片支持 | 生产采用率 |
|------|---------|---------|---------|-----------|
| HDF5 | O(1) | 无（纯张量） | 内置 | 高（Megatron-LM） |
| JSONL | O(N) | 高（JSON 解析） | 手动 | 低（小数据） |
| Parquet | O(1) | 中 | 内置 | 中（数据处理） |
| MMap | O(1) | 无 | 手动 | 高（LLaMA） |

### 4.3 Megatron-LM 中的 HDF5 使用

NVIDIA 的 Megatron-LM 使用 HDF5 作为其标准数据格式。训练流水线为：原始语料 → 分词 → HDF5 分片 → 分布式训练。每个 GPT 训练任务可能涉及数百个 HDF5 分片，通过一个全局索引文件组织。

---

## 5. 工程最佳实践

### 5.1 分片大小选择

每个 HDF5 分片建议包含 500M-2B 词元。太小（<100M）导致文件数量过多，太大（>10B）失去分片容错的优势。

### 5.2 压缩权衡

```python
# 启用压缩（适合存储）：
ds = f.create_dataset("tokens", shape=(0,), maxshape=(None,),
                      dtype=np.uint16, chunks=(8192,),
                      compression="gzip", compression_opts=4)

# 不启用压缩（适合训练读取速度优先）：
ds = f.create_dataset("tokens", shape=(0,), maxshape=(None,),
                      dtype=np.uint16, chunks=(8192,))
```

压缩减少磁盘空间约 30-40%，但读取时需要解压。如果磁盘 I/O 是瓶颈且磁盘空间充足，可以不压缩。

### 5.3 中文场景特别建议

- **词元类型选择**：中文词元通常使用 `uint16`（最大 65535）足够，但 BPE 词表可能达到 100K+。如果词表 > 65535，需要使用 `uint32`。
- **文档边界标记**：中文 NLP 中文档边界标记尤为重要——不同文档之间不应产生跨文档的滑动窗口样本。在文档结尾插入 `BOUNDARY_TOKEN_ID` 并在采样时跳过跨边界位置。

---

## 6. 常见错误

### 错误 1：反复调整 HDF5 数据集大小

**现象：** HDF5 文件碎片化严重，文件大小远大于实际数据量。

**原因：** 每次 `resize()` 调整一个元素的大小，产生大量元数据碎片。

**修复：** 缓冲到一个 chunk 大小后再批量调整。

### 错误 2：启用 SWMR 后忘记反射

**现象：** 写入器关闭后，读取器仍看到旧数据。

**原因：** SWMR 模式下读取器缓存了元数据。需要在打开文件时使用 `libver="latest"`。

**修复：** `h5py.File(path, "r", swmr=True)` — 写入器必须用 `libver="latest"` 模式创建文件。

### 错误 3：滑动窗口跨越文档边界

**现象：** 训练样本包含两个不相关文档的内容。

**原因：** 滑动窗口在全局词元流上无差别滑动，没有感知文档边界。

**修复：** 在文档之间插入分隔词元，并在数据加载器中检查窗口是否包含边界。如果包含，跳过该样本或重新采样。

---

## 7. 面试考点

### Q1：HDF5 的分块存储与行存储相比有什么优势？（难度：⭐⭐）

**参考答案：** 行存储（如 CSV）读取连续行时高效，但读取列的子集时仍需加载整行。分块存储将数据集分成固定大小的块，读取任意子集时只需加载包含请求范围的块。对于训练随机采样（需要读取任意位置的长度固定的词元片段），分块存储显著减少磁盘 I/O。

### Q2：为什么大型语言模型训练不使用 JSONL 而是 HDF5 或内存映射文件？（难度：⭐⭐⭐）

**参考答案：** 三个核心原因：(1) 解析开销——JSONL 的 `json.loads()` 在大规模吞吐下成为瓶颈；(2) 随机访问——JSONL 是行存格式，随机访问需要扫描或额外索引；(3) 内存映射——HDF5 和内存映射文件利用操作系统的页缓存，多个工作进程可以共享缓存的页面。JSONL 无法实现随机切片读取。

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| HDF5 | 层次化数据格式——专为大尺度数值数据设计的文件格式和库 |
| 分块存储 | 将数据集分成固定大小的块，只读取需要的块 |
| 可调整大小 | 数据集可以动态增长，无需预先分配完整空间 |
| SWMR | 单写入器多读取器模式——训练时多个工作进程安全读取同一文件 |
| 滑动窗口 | 在全局词元流上以固定长度滑动产生训练序列 |

---

## 📚 小结

HDF5 为大语言模型训练提供了高效的词元存储和访问方案。你实现了流式分词写入、分片存储策略和滑动窗口数据加载器。与 JSONL 相比，HDF5 在解析开销、随机访问和内存共享方面具有显著优势。

下一节将构建学习率调度器——将余弦预热调度集成到优化器循环中。

---

## ✏️ 练习

1. 【理解】为什么 HDF5 的分块大小选择对训练性能有显著影响？选择一个不合适的 chunk 大小会导致什么问题？

2. 【实现】在 `SlidingWindowLoader` 中添加文档边界感知——跳过跨越 `BOUNDARY_TOKEN_ID` 的窗口，确保训练样本不包含拼接的不相关文本。

3. 【实验】对一个 10 万词的文本语料，分别用 HDF5（有压缩）和 JSONL 存储，对比文件大小和读取 10 万个随机位置的耗时。

4. 【思考】与内存映射二进制文件（如 LLaMA 使用的格式）相比，HDF5 的优缺点是什么？什么场景下应该选择 MMap 而非 HDF5？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| HDF5 分片写入器 | `code/main.py` | 从零实现的 HDF5 分词写入和数据加载 |
| 可复用数据管道 | `outputs/skill-hdf5-corpus.md` | HDF5 语料库构建与加载配置 |

---

## 📖 参考资料

1. [官方文档] HDF5 官方文档. https://docs.hdfgroup.org/hdf5/
2. [官方文档] h5py 文档. https://docs.h5py.org/en/stable/
3. [论文] Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv 1909.08053. https://arxiv.org/abs/1909.08053
4. [GitHub] Megatron-LM 数据处理. https://github.com/NVIDIA/Megatron-LM
