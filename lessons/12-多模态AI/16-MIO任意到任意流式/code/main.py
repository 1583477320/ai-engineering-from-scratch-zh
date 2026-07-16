# MIO 四模态共享词表


def create_shared_vocabulary():
    """创建四种模态的共享词表。"""
    vocab = {}
    offset = 0
    for modality, size in [("text", 32000), ("image", 32000),
                            ("speech", 16000), ("music", 16000)]:
        vocab[modality] = {"start": offset, "end": offset + size, "size": size}
        offset += size
    vocab["total"] = offset
    return vocab


def route_modality(token_id, vocab):
    """根据 token ID 确定模态。"""
    for modality, info in vocab.items():
        if modality == "total":
            continue
        if info["start"] <= token_id < info["end"]:
            return modality
    return "unknown"


if __name__ == "__main__":
    print("MIO 共享词表演示\n")
    vocab = create_shared_vocabulary()
    for mod, info in vocab.items():
        if mod != "total":
            print(f"  {mod}: {info['start']}-{info['end']} ({info['size']} tokens)")
    print(f"  总词表: {vocab['total']} tokens")

    print("\n模态路由:")
    test_ids = [0, 32000, 64000, 80000, 100000]
    for tid in test_ids:
        mod = route_modality(tid, vocab)
        print(f"  token {tid} -> {mod}")
