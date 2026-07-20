# 综合项目42——大语料库下载器（Streaming Corpus Downloader）

> 训练语言模型从第一个词元进入显存之前就开始了。语料必须落在磁盘上、解压缩、去重、可寻址。而恢复——在网络在 41% 处断开之前就已解决。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 用 `urllib` 流式传输远程分片并用 `zstandard` 在线解压缩
- 通过 HTTP Range 请求实现部分下载的断点续传
- 为每个文档构建 MinHash 签名并用 LSH 分桶近似去重
- 发出包含内容哈希、字节大小、文档计数和去重判定的分片清单

---

## 1. 问题

第一次在 200GB 语料上训练时，网络在 41% 处断开。第二次在 78% 处断开。从零开始下载整个语料需要重新计算 40GB 的已验证数据。两个必须从第一分钟设计的失败是**部分下载恢复**和**重复文档移除**。

恢复是 HTTP 问题：服务器必须支持 Range，客户端必须跟踪已验证偏移与磁盘记录的对比。去重是签名问题：精确哈希（MD5、SHA256）只能找到完全相同的文档——而训练语料中大量存在只有一两个词不同的近似重复。这些重复会污染训练数据，使模型过拟合到常见表达。

---

## 2. 核心概念

### 2.1 流式解压缩

`urllib.request.urlopen` 返回类文件对象。用 `zstandard.ZstdDecompressor().stream_reader` 包装，字节从网络流经解压器到文档迭代器，从不将整个压缩/解压分片放入内存。

```mermaid
flowchart LR
    A[远程 URL] -->|HTTP Range| B[网络流]
    B --> C[Zstd 解压器]
    C --> D[文档迭代器]
    D --> E[HDF5 写入器]
```

### 2.2 断点续传

下载器维护一个本地状态文件，记录每个已下载分片的字节偏移。下载开始时先读取该文件，检查远程文件大小是否与本地记录匹配。如果匹配，跳过该分片；否则从上次停止的偏移处恢复。

### 2.3 MinHash + LSH 近似去重

MinHash 在固定空间中估计两个集合的 Jaccard 相似度。对于文档集合，每个文档的 MinHash 签名是一个固定长度的哈希值向量。两个文档的签名越相似，它们越可能是重复文档。

**LSH（局部敏感哈希）**将 k 个签名分量分成 b 个 r 行的带，实现亚线性时间去重。两个文档只要有一对带完全匹配，就判定为近似重复。

```python
# 阈值 s=0.8 用 k=128, b=32, r=4 达到
# 碰撞概率: s^r = 0.8^4 = 0.4096 per band
# 至少一个带碰撞: 1 - (1-0.4096)^32 ≈ 1.0 (高召回率)
```

### 2.4 分片清单

下载器的唯一持久输出是清单。清单保存 URL、解压字节数、文档计数、去重后唯一文档数和最终分片文件的 sha256。下游分词化读取清单。

```json
{
  "url": "https://data.example.com/corpus-00001.zst",
  "decompressed_bytes": 1073741824,
  "total_documents": 125000,
  "unique_documents": 98000,
  "duplicates_removed": 27000,
  "sha256": "a1b2c3d4..."
}
```

---

## 3. 从零实现

```python
"""大语料库下载器——流式+MinHash 去重+清单。"""
import hashlib, json, os, sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def minhash_signature(text: str, k: int = 128, width: int = 5) -> List[int]:
    """为文本生成 MinHash 签名。k 个分量，每个是 k 个哈希值的最小值。"""
    shingles = [text[i:i + width] for i in range(max(0, len(text) - width + 1))]
    if not shingles:
        shingles = [text] if text else [""]
    return [min(hash(f"{seed}:{s}") for s in shingles) for seed in range(k)]


def lsh_bucket(sig: List[int], bands: int = 32, rows: int = 4) -> List[Tuple]:
    """将签名分为 bands 个带，每个带 rows 行。返回带元组列表。"""
    return [tuple(sig[b * rows:(b + 1) * rows]) for b in range(bands)]


def dedup_documents(
    documents: List[str], k: int = 128, bands: int = 32, rows: int = 4
) -> Tuple[List[str], List[str]]:
    """基于 MinHash+LSH 的近似去重。返回 (保留文档, 重复文档)。"""
    seen_bands = set()
    kept = []
    dupes = []
    for doc in documents:
        sig = minhash_signature(doc, k)
        buckets = lsh_bucket(sig, bands, rows)
        if any(b in seen_bands for b in buckets):
            dupes.append(doc[:80])
        else:
            kept.append(doc)
            for b in buckets:
                seen_bands.add(b)
    return kept, dupes


def simulate_download_resume(local_path: str, remote_url: str, chunk_size: int = 1024) -> dict:
    """模拟断点续传逻辑——检查本地状态是否允许跳过。"""
    state_path = local_path + ".state"
    state = {"url": remote_url, "bytes_downloaded": 0, "sha256": hashlib.sha256().hexdigest()}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"  恢复：已下载 {state['bytes_downloaded']} 字节")
    # 模拟写入
    state["bytes_downloaded"] += chunk_size
    state["sha256"] = hashlib.sha256(f"{state['sha256']}{chunk_size}".encode()).hexdigest()
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    return state


def main() -> int:
    # 1. 演示 MinHash + LSH 去重
    print("=== MinHash + LSH 近似去重 ===")
    docs = [
        "the quick brown fox jumps over the lazy dog " * 3,
        "the quick brown fox jumps over the lazy dog " * 3 + " extra text",
        "a completely different document about AI",
        "the quick brown fox jumps over the lazy dog " * 3 + " variant",
        "artificial intelligence is transforming the world",
        "a completely different document about ML",
    ]
    kept, dupes = dedup_documents(docs)
    print(f"原始: {len(docs)}  保留: {len(kept)}  重复: {len(dupes)}")
    for d in dupes:
        print(f"  重复: {d}")
    print()

    # 2. 演示断点续传
    print("=== 断点续传演示 ===")
    os.makedirs("/tmp/corpus_dl", exist_ok=True)
    state1 = simulate_download_resume("/tmp/corpus_dl/shard_001.zst", "https://example.com/corpus/shard_001.zst")
    state2 = simulate_download_resume("/tmp/corpus_dl/shard_001.zst", "https://example.com/corpus/shard_001.zst")
    print(f"  第一次字节: {state1['bytes_downloaded']}  第二次字节: {state2['bytes_downloaded']}")
    print()

    # 3. 演示清单生成
    print("=== 分片清单 ===")
    manifest = {
        "url": "https://example.com/corpus/shard_001.zst",
        "decompressed_bytes": len(" ".join(kept).encode("utf-8")),
        "total_documents": len(docs),
        "unique_documents": len(kept),
        "duplicates_removed": len(dupes),
        "sha256": hashlib.sha256(json.dumps(kept).encode()).hexdigest()[:16],
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. 工业工具

### 4.1 Zstandard 解压

`zstandard` 库是 HuggingFace Datasets 和 OpenWebText 项目使用的标准解压工具：

```python
import zstandard as zstd

# 流式解压——内存友好
dctx = zstd.ZstdDecompressor()
reader = dctx.stream_reader(compressed_file)
# reader 是一个类文件对象，逐块读取
```

### 4.2 Common Crawl 处理

Common Crawl 是最常用的网页语料库，原始数据约 10-20TB。处理流程：

1. 下载 WARC 文件（WARC.gz 压缩）
2. 提取 HTML → 纯文本（使用 `trafilatura` 或 `readability-lxml`）
3. MinHash + LSH 去重（使用 `dask` 并行处理）
4. 分片写入 HDF5

### 4.3 性能对比

| 工具 | 速度 | 内存占用 | 说明 |
|------|------|---------|------|
| 自定义流式 | ~50 MB/s | 低 | 零依赖，完全可控 |
| HuggingFace Datasets | ~100 MB/s | 中 | 自动处理多种格式 |
| `wget` + 后处理 | 最快 | 高 | 下载快但需额外去重步骤 |

---

## 5. 工程最佳实践

### 5.1 流式处理优先

不要将整个分片加载到内存。使用流式管道处理大语料，字节从网络流经解压器、分词器、去重器，最终写入磁盘。内存占用始终是文档级别的，不是分片级别的。

### 5.2 并行处理

使用 `concurrent.futures.ThreadPoolExecutor` 并行下载多个分片。每个线程处理一个 URL。注意 `urllib` 是阻塞的，适合线程池而非进程池。

### 5.3 中文场景特别建议

- **中文去重需要分词**：原始字节级别的去重对中文效果差——"我喜欢你"和"你喜欢我"有大量字符重叠。在 MinHash 前先用分词器（如 jieba）将文档转为词级 n-gram，能显著提高去重质量。
- **分片清单的文档计数**：中文语料的文档边界比英文更难确定（无句号分隔的连续文本）。建议使用段落分隔符或固定长度切分。

---

## 6. 常见错误

### 错误 1：整个分片加载到内存

**现象：** 下载 1GB 的压缩分片时，内存使用跳到 2GB+。

**原因：** 使用 `open(URL).read()` 将整个文件读入内存。

**修复：** 使用流式读取——`response = urllib.request.urlopen(url)`，然后逐块读取和处理。

### 错误 2：MinHash 的 k 值过小

**现象：** 去重率远低于预期，大量近似重复文档未被捕获。

**原因：** k=64 的签名长度不足以区分语义相似但不完全相同的文档。

**修复：** 使用 k=128 或更大。对于训练语料去重，推荐 k=128, b=32, r=4。

### 错误 3：下载完成后忘记清理状态文件

**现象：** 下次下载时错误地跳过了一个已损坏的分片。

**原因：** 状态文件记录了"已下载完成"，但实际上下载是不完整的。

**修复：** 在下载完成后计算 sha256 并与服务器提供的哈希对比，只有匹配时才标记为"完成"。

---

## 7. 面试考点

### Q1：为什么训练语料去重比精确去重更重要？（难度：⭐⭐）

**参考答案：** 精确去重只能找到字节级完全相同的文档。训练语料中大量存在只有一两个词不同的文档——例如同一条新闻的不同版本、同一篇论文的不同抓取时间戳。这些近似重复如果不被移除，会使模型过拟合到常见表达，降低泛化能力。MinHash+LSH 在亚线性时间找到这些近似重复。

### Q2：如何选择 MinHash 的参数 (k, b, r)？（难度：⭐⭐⭐）

**参考答案：** 参数选择遵循公式：$s^r \approx 1 - (1-p)^{1/b}$，其中 $s$ 是相似度阈值（如 0.8），$p$ 是期望碰撞概率，$b$ 是带数，$r$ 是每带行数，$k = b \times r$ 是签名长度。对于 0.8 的阈值，使用 k=128, b=32, r=4 是经验上的好选择。增加 k 可以减少假阳性但增加存储和计算成本。

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| MinHash | 用于估计两个集合相似度的哈希技术——签名越相似，集合越可能重叠 |
| LSH（局部敏感哈希） | 将签名分桶实现亚线性去重的索引结构 |
| 断点续传 | 从上次停止的偏移处恢复下载，不重传已验证的数据 |
| 分片清单 | 下载器输出的 JSON 元数据文件，记录 URL、字节数、文档计数等 |
| Jaccard 相似度 | 两个集合交集大小除以并集大小——MinHash 估计的目标值 |

---

## 📚 小结

大语料库下载器是训练流水线的第一步。你实现了流式解压缩、断点续传、MinHash+LSH 近似去重和清单生成。这些组件是处理 TB 级训练数据的基础，确保语料在进入分词和训练之前是干净、可恢复的。

下一节将把下载的语料转换为 HDF5 格式，构建高效的训练数据集。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么 MinHash 使用多个哈希函数而不是单个大哈希。这与 Bloom Filter 的原理有什么联系？

2. 【实现】将 `main()` 中的模拟下载替换为真正的 HTTP 下载——用 `urllib.request.urlopen` 流式读取一个公开的文本文件，并在线解压。

3. 【实验】对 1000 篇随机生成的中文文本，分别用字节级和词级（分词后）MinHash 去重，对比去重率的差异。

4. 【思考】下载器输出的清单文件与 HuggingFace Datasets 的缓存目录结构相比，有哪些异同？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 语料库下载器 | `code/main.py` | MinHash+LSH 去重+清单生成 |
| 可复用下载配方 | `outputs/skill-corpus-downloader.md` | 流式下载和去重配置 |

---

## 📖 参考资料

1. [论文] Broder. "On the Resemblance and Containment of Documents". 1997. https://cs.bgu.ac.il/~zviAA/hpc/1516_Seminar/Broder-97.pdf
2. [官方文档] Zstandard 库. https://facebook.github.io/zstd/
3. [GitHub] HuggingFace Datasets 数据加载. https://github.com/huggingface/datasets
4. [官方文档] Common Crawl 处理流程. https://commoncrawl.org/
