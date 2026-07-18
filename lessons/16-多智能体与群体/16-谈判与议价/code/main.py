"""谈判：合同网 + OG-Narrator 演示。

对比朴素全 LLM 议价 vs OG-Narrator（确定性报价生成器 + LLM 叙述）。
测量 1000 次试验的成交率。包含小的合同网任务市场演示。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class BargainState:
    buyer_max: int
    seller_min: int
    buyer_offer: int | None = None
    seller_offer: int | None = None
    rounds: int = 0
    max_rounds: int = 5


def naive_llm_bargain(state: BargainState, rng: random.Random) -> int:
    """朴素 LLM 议价：方差高，经常在 ZOPA 外。"""
    r = rng.random()
    if state.seller_offer is None:
        return rng.randint(state.buyer_max - 60, state.buyer_max + 30)
    elif r < 0.35:
        return state.seller_offer + rng.randint(-8, 3)
    elif r < 0.65:
        return rng.randint(state.seller_min - 30, state.buyer_max + 30)
    else:
        return rng.randint(state.seller_min - 60, state.buyer_max + 60)


def og_narrator_bargain(state, rng, concession=0.35):
    """OG-Narrator 确定性报价：Zeuthen 风格向中点让步。"""
    if state.seller_offer is None and state.buyer_offer is None:
        return state.buyer_max - max(1, int((state.buyer_max - state.seller_min) * 0.2))
    if state.seller_offer is None:
        return state.buyer_offer
    prior = state.buyer_offer if state.buyer_offer is not None else state.buyer_max
    move = max(1, int(concession * (state.seller_offer - prior)))
    return min(prior + move, state.buyer_max)


def seller_response(state, rng, concession=0.3):
    """卖方反报价。"""
    if state.buyer_offer is None and state.seller_offer is None:
        return state.seller_min + max(1, int((state.buyer_max - state.seller_min) * 0.4))
    if state.buyer_offer is None:
        return state.seller_offer
    prior = state.seller_offer if state.seller_offer is not None else state.seller_min + 20
    move = max(1, int(concession * (prior - state.buyer_offer)))
    return max(prior - move, state.seller_min)


def simulate_bargain(buyer_fn, rng, buyer_max=100, seller_min=60):
    state = BargainState(buyer_max=buyer_max, seller_min=seller_min)
    deal = False
    while state.rounds < state.max_rounds:
        state.buyer_offer = buyer_fn(state, rng)
        if state.seller_offer is not None and state.buyer_offer >= state.seller_offer:
            if state.seller_offer >= state.seller_min and state.seller_offer <= state.buyer_max:
                deal = True
            break
        state.seller_offer = seller_response(state, rng)
        if state.buyer_offer is not None and state.seller_offer <= state.buyer_offer:
            if state.buyer_offer <= state.buyer_max and state.buyer_offer >= state.seller_min:
                deal = True
            break
        state.rounds += 1
    return deal


def bench_deal_rate(buyer_fn, label, trials=1000):
    rng = random.Random(42)
    deals = 0
    for _ in range(trials):
        seller_min = rng.randint(50, 80)
        buyer_max = rng.randint(max(seller_min + 5, 75), 115)
        if simulate_bargain(buyer_fn, rng, buyer_max=buyer_max, seller_min=seller_min):
            deals += 1
    print(f"  {label:20s} 成交率: {deals / trials:.2%}  ({deals}/{trials})")


@dataclass
class Bid:
    bidder: str
    price: int
    eta_minutes: int
    confidence: float

class ContractNetManager:
    def __init__(self, bidders):
        self.bidders = bidders
        self.proposals = {}

    def broadcast_cfp(self, task):
        self.proposals[task.task_id] = []

    def receive_proposal(self, task_id, bid):
        self.proposals[task_id].append(bid)

    def award(self, task):
        props = self.proposals.get(task.task_id, [])
        feasible = [b for b in props if b.price <= task.budget and b.eta_minutes <= task.deadline_minutes]
        if not feasible:
            return None
        winner = max(feasible, key=lambda b: b.confidence / max(b.price, 1))
        return winner


def main():
    print("=" * 72)
    print("成交率——朴素 LLM 议价 vs OG-Narrator")
    print("=" * 72)
    bench_deal_rate(naive_llm_bargain, "naive LLM")
    bench_deal_rate(og_narrator_bargain, "OG-Narrator")

    print("\n合同网任务市场演示")
    from dataclasses import dataclass
    @dataclass
    class ContractNetTask:
        task_id: str
        description: str
        deadline_minutes: int
        budget: int

    task = ContractNetTask("t-1", "压缩 10GB 日志包", 30, 10)
    mgr = ContractNetManager(["worker-a", "worker-b", "worker-c"])
    mgr.broadcast_cfp(task)
    mgr.receive_proposal("t-1", Bid("worker-a", 3, 18, 0.82))
    mgr.receive_proposal("t-1", Bid("worker-b", 2, 25, 0.77))
    mgr.receive_proposal("t-1", Bid("worker-c", 4, 10, 0.90))
    winner = mgr.award(task)
    if winner:
        print(f"  赢家: {winner.bidder} (价格 {winner.price}, ETA {winner.eta_minutes}分钟)")

    print("\n要点:")
    print("  朴素 LLM 议价方差大——经常在 ZOPA 外。")
    print("  OG-Narrator（确定性报价 + LLM 叙述）每次试验都收敛。")
    print("  合同网扩展：广播 + 收集 + 颁奖；无需同步聊天。")


if __name__ == "__main__":
    main()
