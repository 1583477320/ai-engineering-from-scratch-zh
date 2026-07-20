# 综合项目43——HDF5分词语料库

> 下载的语料必须以训练器能以线速流式传输的布局落地。磁盘上的JSONL无法承受16个dataloader worker。HDF5带可调整大小的分块整数数据集可以。本课程构建流式分词到可调整大小的HDF5数据集，跨多个文件分片写入，训练时内存映射读取，以及产生固定长度序列的滑动窗口数据加载器。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将文档流式传输到可调整大小的HDF5整数数据集
- 跨多个HDF5文件分片写入
- 通过HDF5的页缓存分块布局读取token
- 实现产生固定长度训练序列的滑动窗口数据加载器

---

## 1. 问题

现代LLM训练以每秒数十万样本的速度跨数十个worker读取token。JSONL在磁盘上无法承受第一次冷缓存页错误：JSON解析器慢、文档边界不可寻址、搜索到特定样本需要扫描文件。HDF5提供分块、可调整大小、纯整数数据集，读取时页缓存友好。

构建问题是使写入端诚实。可调整大小的数据集容易误用：每次写一个文档会导致文件碎片化；一次性调整大小写所有文档时进程崩溃会丢失整个分片。

---

## 2. 核心概念

### 2.1 正确的可调整大小HDF5

token数据集以`maxshape=(None,)`和固定`chunks=(chunk_size,)`创建。写入通过缓冲token到`chunk_size`长度的NumPy数组进行。缓冲填满时，数据集恰好调整`chunk_size`并写入新范围。

### 2.2 分片写入

单个HDF5文件是单点故障。管道并行写入分片：每个输入分片产生一个HDF5输出分片。`shards.json`索引记录每个分片的文件路径、token计数、文档计数和token的sha256。

### 2.3 内存映射读取

训练时每个worker以`swmr=True`模式打开HDF5文件并请求`tokens[start:stop]`。HDF5的分块布局使这是页缓存支持的读取。

### 2.4 滑动窗口数据加载器

数据加载器是唯一知道训练序列长度的阶段。它在全局token流中随机选择起始索引，读取`window_size + 1`个token，返回`(input, target)`。

---

## 3. 从零实现

```python
"""HDF5分词语料库——流式分词+分片写入+内存映射读取。"""
import json, hashlib, numpy as np
from dataclasses import dataclass, field
from typing import List

try:
    import h5py
except ImportError:
    h5py = None

BOUNDARY_TOKEN_ID = 0; TOKEN_DTYPE = np.uint16

@dataclass
class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

@dataclass
class HDF5ShardWriter:
    path: str; chunk_size: int = 8192; buffer: list = field(default_factory=list)
    tokens_written: int = 0

    def __post_init__(self):
        if h5py:
            self.file = h5py.File(self.path, "w", libver="latest")
            self.ds = self.file.create_dataset("tokens", shape=(0,), maxshape=(None,), dtype=TOKEN_DTYPE, chunks=(self.chunk_size,))
            self.file.swmr_mode = True

    def append(self, token_ids):
        self.buffer.extend(token_ids)
        while len(self.buffer) >= self.chunk_size:
            chunk = np.array(self.buffer[:self.chunk_size], dtype=TOKEN_DTYPE)
            self.ds.resize(self.ds.shape[0] + self.chunk_size)
            self.ds[self.tokens_written:self.tokens_written+self.chunk_size] = chunk
            self.tokens_written += self.chunk_size
            self.buffer = self.buffer[self.chunk_size:]
            self.ds.flush()

    def close(self):
        if self.buffer:
            chunk = np.array(self.buffer, dtype=TOKEN_DTYPE)
            self.ds.resize(self.ds.shape[0] + len(chunk))
            self.ds[self.tokens_written:self.tokens_written+len(chunk)] = chunk
            self.tokens_written += len(chunk)
        self.ds.attrs["token_count"] = self.tokens_written
        sha = hashlib.sha256(np.array(self.ds[:self.tokens_written]).tobytes()).hexdigest()
        self.ds.attrs["sha256"] = sha
        self.file.close()

def main():
    if not h5py:
        print("h5py required"); return 1
    tok = ByteTokenizer()
    docs = ["hello world training corpus document one", "second document with different content", "third document about machine learning"]
    path = "/tmp/test_shard.h5"
    writer = HDF5ShardWriter(path)
    total_tokens = 0
    for doc in docs:
        ids = tok.encode(doc)
        writer.append(ids + [BOUNDARY_TOKEN_ID])
        total_tokens += len(ids) + 1
    writer.close()
    with h5py.File(path, "r") as f:
        print(f"token_count: {f['tokens'].attrs['token_count']}")
        print(f"sha256: {f['tokens'].attrs['sha256'][:16]}...")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```
