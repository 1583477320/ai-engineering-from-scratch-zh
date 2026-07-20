# 综合项目42——大语料库下载器（流式下载+去重+清单）

> 训练语言模型从第一个前向传播之前就开始了。语料必须落在磁盘上，解压缩，去重，可寻址，恢复故事在网络在4%处断开之前就已解决。本课程构建流式下载器，拉取压缩分片，用Zstandard在线解压缩，用MinHash加局部敏感哈希指纹近似重复，写入分片清单供管道其余部分信任。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 用`urllib`流式传输远程分片并用`zstandard`在线解压缩
- 通过HTTP Range请求恢复部分下载
- 为每个文档构建MinHash签名并用LSH分桶
- 发出包含内容哈希、字节大小、文档计数和去重判定的分片清单

---

## 1. 问题

第一次在200GB语料上训练时，网络在41%处断开。第二次在78%处断开。两个必须从第一分钟设计的失败是部分下载恢复和重复文档移除。

恢复是HTTP问题：服务器必须支持Range，客户端必须跟踪已验证偏移与磁盘记录的对比。去重是签名问题：精确哈希遗漏近似重复。

---

## 2. 核心概念

### 2.1 流式解压缩

`urllib.request.urlopen`返回类文件对象。用`zstandard.ZstdDecompressor().stream_reader`包装，字节从网络流经解压器到文档迭代器，从不将整个压缩/解压分片放入内存。

### 2.2 MinHash + LSH

MinHash在固定空间中估计两个集合的Jaccard相似度。LSH将k个分量分成b个r行的带，实现亚线性去重。阈值s=0.8用k=128, b=32, r=4达到。

### 2.3 分片清单

下载器的唯一持久输出是清单。清单保存URL、解压字节数、文档计数、去重后唯一文档数和最终分片文件的sha256。下游分词化读取清单。

---

## 3. 从零实现

```python
"""大语料库下载器——流式+MinHash去重+清单。"""
import hashlib, json, os, tempfile, urllib.request
from dataclasses import dataclass, field
from typing import List

def minhash_signature(text, k=128, width=5):
    shingles=[text[i:i+width] for i in range(max(0,len(text)-width+1))]
    return [min(hash(f"{seed}:{s}") for s in shingles or [""]) for seed in range(k)]

def lsh_bucket(sig, bands=32, rows=4):
    return [tuple(sig[b*rows:(b+1)*rows]) for b in range(bands)]

def dedup(documents, k=128, bands=32, rows=4):
    seen_bands=set(); kept=[]; dupes=[]
    for doc in documents:
        sig=minhash_signature(doc,k)
        buckets=lsh_bucket(sig,bands,rows)
        if any(b in seen_bands for b in buckets):
            dupes.append(doc[:60])
        else:
            kept.append(doc)
            for b in buckets: seen_bands.add(b)
    return kept,dupes

def main():
    docs=["the quick brown fox jumps"*3,"the quick brown fox jumps"*3+" extra",
          "a completely different document","the quick brown fox jumps"*3+" variant"]
    kept,dupes=dedup(docs)
    print(f"原始: {len(docs)}  去重后: {len(kept)}  重复: {len(dupes)}")
    manifest={"total":len(docs),"unique":len(kept),"duplicates":len(dupes),
              "sha256":hashlib.sha256(json.dumps(kept).encode()).hexdigest()[:16]}
    print(f"清单: {json.dumps(manifest,indent=2)}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
```
