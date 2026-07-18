"""缓存感知路由器模拟。"""


class CacheAwareRouter:
    def __init__(self):
        self.replica_cache = {}

    def update(self, replica_id, prefix_hash, has_cached):
        if replica_id not in self.replica_cache:
            self.replica_cache[replica_id] = set()
        if has_cached:
            self.replica_cache[replica_id].add(prefix_hash)
        else:
            self.replica_cache[replica_id].discard(prefix_hash)

    def route(self, prefix_hash):
        candidates = [r for r, h in self.replica_cache.items() if prefix_hash in h]
        return candidates[0] if candidates else list(self.replica_cache.keys())[0]


if __name__ == "__main__":
    router = CacheAwareRouter()
    router.update("replica-1", "hash_A", True)
    router.update("replica-1", "hash_B", True)
    router.update("replica-2", "hash_C", True)

    for h in ["hash_A", "hash_B", "hash_C", "hash_D"]:
        print(f"路由 {h} → {router.route(h)}")
