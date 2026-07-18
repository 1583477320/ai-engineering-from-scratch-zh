"""GPU 自动扩缩容模拟器——对比朴素 HPA、队列深度 HPA 和组调度。"""

from collections import deque


class AutoscalerSim:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.queue = deque()
        self.replicas = 3
        self.served = 0
        self.dropped = 0
        self.idle = 0

    def tick(self, incoming: int):
        for _ in range(incoming):
            self.queue.append(1)

        if self.strategy == "queue_depth":
            desired = max(1, len(self.queue) // 2)
            self.replicas = max(1, min(desired, 10))
        elif self.strategy == "naive_hpa":
            util = min(1.0, len(self.queue) / (self.replicas * 5))
            if util > 0.8:
                self.replicas = min(self.replicas + 1, 10)
            elif util < 0.3 and self.replicas > 1:
                self.replicas -= 1

        capacity = self.replicas * 2
        for _ in range(min(len(self.queue), capacity)):
            self.queue.popleft()
            self.served += 1

        self.dropped += max(0, len(self.queue) - 50)
        self.idle += max(0, self.replicas - capacity // 4)


def run():
    strategies = ["naive_hpa", "queue_depth", "gang"]
    sims = {s: AutoscalerSim(s) for s in strategies}

    for step in range(200):
        burst = 5 + int(20 * (0.5 + 0.5 * (step % 15 > 10)))
        for sim in sims.values():
            sim.tick(burst)

    print(f"{'策略':15s} {'服务':>6} {'丢弃':>6} {'空闲':>8}")
    print("-" * 38)
    for s in strategies:
        sim = sims[s]
        print(f"{s:15s} {sim.served:6d} {sim.dropped:6d} {sim.idle:8d}")


if __name__ == "__main__":
    run()
