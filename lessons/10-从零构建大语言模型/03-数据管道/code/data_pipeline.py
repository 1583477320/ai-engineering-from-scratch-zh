# 数据管道：流式读取 + 分词 + 分块 + DataLoader

import hashlib
from collections import Counter


def stream_text(file_path, chunk_size=1024*1024):
    """流式读取文本文件——每次 1MB。"""
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def quality_filter(text, min_words=10, max_special_ratio=0.3):
    """基本质量过滤。"""
    words = text.split()
    if len(words) < min_words:
        return False
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special / max(len(text), 1) > max_special_ratio:
        return False
    unique_words = set(words)
    if len(words) > 0 and len(unique_words) / len(words) < 0.3:
        return False
    return True


def chunk_tokens(token_ids, chunk_size=2048):
    """将 token 序列分块为固定长度。"""
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        if len(chunk) > 10:
            chunks.append(chunk)
    return chunks


def deduplicate_texts(texts):
    """精确去重——基于文本哈希。"""
    seen = set()
    unique = []
    for text in texts:
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(text)
    return unique


def create_dataloader(chunks, batch_size=4, seq_len=2048):
    """创建 PyTorch DataLoader。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.tensor([c[:seq_len] for c in chunks], dtype=torch.long),
        torch.tensor([c[1:seq_len+1] for c in chunks], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # 示例
    texts = [
        "这是一段测试文本。" * 10,
        "另一段测试文本。" * 10,
        "这是一段测试文本。" * 10,  # 重复
    ]
    unique = deduplicate_texts(texts)
    print(f"去重前: {len(texts)}, 去重后: {len(unique)}")

    # 质量过滤
    for text in texts[:2]:
        ok = quality_filter(text)
        print(f"  '{text[:20]}...' -> {'通过' if ok else '过滤'}")
