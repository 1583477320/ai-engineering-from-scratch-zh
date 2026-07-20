# 大语料库下载器配方

## 流式解压

```python
import zstandard as zstd
dctx = zstd.ZstdDecompressor()
reader = dctx.stream_reader(open("corpus.zst", "rb"))
# 逐块读取，不将整个文件放入内存
```

## 断点续传

```python
# 检查本地状态
state = load_state(f"{local_path}.state")
if state["bytes_downloaded"] == remote_size:
    print("跳过：已完整下载")
else:
    # 从上次停止处恢复
    start = state["bytes_downloaded"]
    # ... 续传下载 ...
```

## MinHash + LSH 近似去重

```python
# 阈值 s=0.8 用 k=128, b=32, r=4
# 碰撞概率 per band: 0.8^4 = 0.4096
# 至少一个带碰撞: 1 - (1-0.4096)^32 ≈ 1.0
```

## 分片清单格式

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
