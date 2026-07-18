"""生成智能体微型实现：stdlib 中的 Smallville。

五个智能体共享一个小世界。智能体 0 被植入派对目标。
随 tick 推进，邀请通过双边记忆观察传播，反思综合信念，
计划更新。到最终 tick 时，3+ 智能体在派对地点汇聚，无需任何中央编排者。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Memory:
    ts: int
    kind: str
    content: str
    importance: int

@dataclass
class Plan:
    tick: int
    where: str
    note: str

@dataclass
class Agent:
    name: str
    location: str
    stream: list[Memory] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)

    def observe(self, tick, content, importance=3):
        self.stream.append(Memory(tick, "observation", content, importance))

    def reflect(self, tick):
        """反思：从最近的高重要性记忆生成信念。"""
        recent_important = [m for m in self.stream
                            if m.importance >= 6 and tick - m.ts <= 5]
        for m in recent_important:
            if "invited" in m.content and "party at" in m.content:
                belief = "there is a party I was invited to"
                if belief not in self.beliefs:
                    self.beliefs.append(belief)
                    self.stream.append(Memory(tick, "reflection", belief, 8))

    def update_plan(self, tick):
        if "there is a party I was invited to" in self.beliefs:
            if not any(p.where == "HobbsCafe" for p in self.plans):
                self.plans.append(Plan(tick=5, where="HobbsCafe", note="attend the party"))

    def act(self, tick):
        for p in self.plans:
            if p.tick == tick:
                self.location = p.where
                return f"{self.name} moves to {p.where} ({p.note})"
        return f"{self.name} remains at {self.location}"


def retrieve_top_k(stream, query, tick, k=3):
    """检索：最近性 + 重要性 + 相关性的加权组合。"""
    def score(m):
        recency = math.exp(-0.3 * (tick - m.ts))
        importance = m.importance / 10.0
        relevance = 0.6 if any(w in m.content.lower()
                               for w in query.lower().split()) else 0.1
        return recency + importance + relevance
    return sorted(stream, key=score, reverse=True)[:k]


def run_simulation(n_agents=5, ticks=6):
    agents = [Agent(f"agent-{i}", location="home") for i in range(n_agents)]

    # 种子智能体 0 的派对目标
    agents[0].stream.append(Memory(0, "goal",
                                    "host a Valentine's party at HobbsCafe at tick 5", 10))
    agents[0].plans.append(Plan(tick=5, where="HobbsCafe", note="host the party"))
    agents[0].beliefs.append("there is a party I was invited to")

    print("=" * 72)
    print(f"生成智能体（微型）— {n_agents} 智能体, {ticks} ticks")
    print("=" * 72)

    for tick in range(ticks):
        print(f"\n--- tick {tick} ---")
        # 邀请传播
        if tick == 0:
            for i in (1, 2):
                agents[i].observe(tick, f"agent-0 invited me to a party at HobbsCafe at tick 5", 8)
                print(f"  agent-0 -> agent-{i}: invitation")
        if tick == 1:
            agents[3].observe(tick, "agent-1 invited me to a party at HobbsCafe at tick 5", 7)
            print(f"  agent-1 -> agent-3: second-degree invitation")
        if tick == 2:
            agents[4].observe(tick, "agent-2 invited me to a party at HobbsCafe at tick 5", 7)
            print(f"  agent-2 -> agent-4: second-degree invitation")

        for a in agents:
            a.reflect(tick)
            a.update_plan(tick)
            action = a.act(tick)
            if action.startswith(a.name + " moves"):
                print(f"  {action}")

    print("\n" + "=" * 72)
    print("最终位置:")
    for a in agents:
        print(f"  {a.name:10s} at {a.location}")

    at_party = sum(1 for a in agents if a.location == "HobbsCafe")
    print(f"\n{at_party}/{n_agents} agents converged at HobbsCafe.")
    print("没有编排者。一个种子。其余是记忆 + 反思 + 计划。")


def demo_retrieval():
    print("\n" + "=" * 72)
    print("检索演示 — 按最近性 + 重要性 + 相关性排序的 Top-k")
    print("=" * 72)
    stream = [
        Memory(0, "observation", "saw Isabella at the cafe", 4),
        Memory(1, "observation", "Isabella said she is planning a party", 7),
        Memory(2, "reflection", "I would enjoy a party at the cafe", 6),
        Memory(3, "observation", "Klaus mentioned he is writing a paper", 3),
    ]
    top = retrieve_top_k(stream, query="party cafe", tick=4, k=3)
    print("  query: 'party cafe' at tick 4")
    for m in top:
        print(f"  [t={m.ts}] {m.kind:11s} imp={m.importance} :: {m.content}")


def main():
    run_simulation()
    demo_retrieval()
    print("\n要点:")
    print("  一个种子 + 三个组件 = 无需编排者的协调到达。")
    print("  反思是负载承载的：去掉它停止信念形成。")
    print("  检索组合最近性、重要性、相关性——单个分数不够。")


if __name__ == "__main__":
    main()
