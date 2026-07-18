"""成本跟踪器和终止开关。"""
import statistics


class CostTracker:
    def __init__(self):
        self.usage = {}

    def record(self, tenant, cost):
        self.usage.setdefault(tenant, []).append(cost)

    def check_kill_switch(self, tenant, z=4):
        costs = self.usage.get(tenant, [])
        if len(costs) < 10:
            return {"triggered": False}
        recent = costs[-5:]
        mu = statistics.mean(costs)
        sigma = max(statistics.stdev(costs), 0.001)
        zs = (sum(recent) / 5 - mu) / sigma
        return {"triggered": zs > z, "z_score": zs,
                "action": "暂停" if zs > z else "继续"}


if __name__ == "__main__":
    t = CostTracker()
    for i in range(100):
        cost = 0.01 * (1 + 0.1 * (i > 90))
        t.record("t1", cost)
    r = t.check_kill_switch("t1")
    print(f"z={r['z_score']:.2f}  {r['action']}")
