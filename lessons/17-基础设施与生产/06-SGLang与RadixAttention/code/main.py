"""Radix 树前缀缓存模拟。"""


class RadixCache:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.tree = {}

    def lookup(self, tokens):
        matched = 0
        node = "root"
        for t in tokens:
            key = f"{node}:{t}"
            if key in self.tree:
                node = key
                matched += 1
            else:
                break
        return matched // self.block_size, len(tokens) - matched

    def insert(self, tokens, kv_blocks):
        node = "root"
        for t in tokens:
            key = f"{node}:{t}"
            if key not in self.tree:
                self.tree[key] = 0
            self.tree[key] = kv_blocks
            node = key

    def stats(self):
        total = sum(self.tree.values())
        return {"nodes": len(self.tree), "total_blocks": total}


if __name__ == "__main__":
    cache = RadixCache(block_size=16)
    # 模拟共享前缀
    prefix = list(range(200))  # 200 词元的系统提示
    cache.insert(prefix, kv_blocks=13)

    hits, misses = cache.lookup(prefix + list(range(200, 210)))
    print(f"前缀命中: {hits} 块  新分配: {misses} 词元")
    print(f"树统计: {cache.stats()}")
