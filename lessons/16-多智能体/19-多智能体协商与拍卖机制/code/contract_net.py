# contract_net.py — 从零实现合同网协议
# 依赖：标准库（无第三方依赖）
# 对应课程：阶段 16 · 19（多智能体协商与拍卖机制）

"""
合同网协议（Contract Net Protocol）的教学实现。
包含管理者（Manager）和投标者（Bidder）两个角色，
支持任务公告、投标收集、评标和授标四个阶段。
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import random


# === 数据结构 ===

@dataclass
class Task:
    """一个待分配的任务。"""
    task_id: str
    description: str
    required_skill: str      # 需要的技能类型
    priority: int = 1        # 优先级，越高越紧急
    max_budget: float = 100  # 最大预算（算力单位）


@dataclass
class Agent:
    """一个可以投标的智能体。"""
    agent_id: str
    skills: dict             # 技能 -> 能力评分 (0~1)
    current_load: float = 0  # 当前负载 (0~1)，1 表示满载
    history_success: float = 0.8  # 历史成功率


@dataclass
class Bid:
    """一份投标书。"""
    agent_id: str
    skill_match: float       # 技能匹配度 (0~1)
    estimated_cost: float    # 预估成本
    estimated_time: float    # 预估耗时（秒）


# === 管理者 ===

class ContractNetManager:
    """合同网协议的管理者——发布任务、收集投标、选择中标者。"""

    def __init__(self, agents: list[Agent], bid_timeout: float = 5.0):
        """
        Args:
            agents: 可投标的智能体列表
            bid_timeout: 投标超时时间（秒），超时后停止收集
        """
        self.agents = agents
        self.bid_timeout = bid_timeout

    def announce(self, task: Task) -> list[Bid]:
        """向所有智能体广播任务公告，收集投标。"""
        bids = []
        start = time.time()

        for agent in self.agents:
            # 超时检查
            if time.time() - start > self.bid_timeout:
                break

            bid = self._evaluate_bid(agent, task)
            if bid is not None:
                bids.append(bid)

        # 如果没有投标，强制分配给负载最低的智能体（降级策略）
        if not bids:
            fallback = min(self.agents, key=lambda a: a.current_load)
            bids.append(Bid(
                agent_id=fallback.agent_id,
                skill_match=0.5,
                estimated_cost=task.max_budget * 0.5,
                estimated_time=2.0,
            ))

        return bids

    def _evaluate_bid(self, agent: Agent, task: Task) -> Optional[Bid]:
        """智能体评估自身是否应该投标。"""
        # 技能匹配度
        skill_match = agent.skills.get(task.required_skill, 0)
        if skill_match < 0.3:
            return None  # 能力不足，不投标

        # 预估成本：负载越高，成本越高
        estimated_cost = task.max_budget * (0.3 + 0.7 * agent.current_load)

        return Bid(
            agent_id=agent.agent_id,
            skill_match=skill_match,
            estimated_cost=estimated_cost,
            estimated_time=1.0 + agent.current_load * 2,
        )

    def evaluate_bids(self, bids: list[Bid], task: Task,
                      alpha: float = 0.5, beta: float = 0.3,
                      gamma: float = 0.2) -> Optional[Bid]:
        """评标：选择综合得分最高的投标者。

        Args:
            bids: 所有投标书
            task: 当前任务
            alpha: 匹配度权重
            beta: 负载均衡权重
            gamma: 历史成功率权重
        """
        if not bids:
            return None

        best_bid = None
        best_score = -1

        for bid in bids:
            agent = self._get_agent(bid.agent_id)
            score = (
                alpha * bid.skill_match
                + beta * (1 - agent.current_load)
                + gamma * agent.history_success
            )
            if score > best_score:
                best_score = score
                best_bid = bid

        return best_bid

    def _get_agent(self, agent_id: str) -> Agent:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        raise ValueError(f"未知智能体: {agent_id}")


# === 拍卖机制 ===

class EnglishAuction:
    """英式拍卖（升价拍卖）。"""

    def __init__(self, task: Task, starting_price: float):
        self.task = task
        self.price = starting_price
        self.winner = None

    def bid(self, agent_id: str, bid_price: float) -> bool:
        """出价。返回是否成功。"""
        if bid_price > self.price:
            self.price = bid_price
            self.winner = agent_id
            return True
        return False


class VickreyAuction:
    """维克里拍卖（二价拍卖）——密封投标，最高价者支付第二高价。"""

    def __init__(self, task: Task):
        self.task = task
        self.bids = []  # (agent_id, bid_price)

    def submit_bid(self, agent_id: str, bid_price: float):
        """提交密封投标。"""
        self.bids.append((agent_id, bid_price))

    def determine_winner(self) -> Optional[tuple[str, float]]:
        """确定获胜者和支付价格。"""
        if len(self.bids) < 2:
            return None

        # 按出价排序
        sorted_bids = sorted(self.bids, key=lambda x: x[1], reverse=True)
        winner_id = sorted_bids[0][0]
        # 赢家支付第二高价
        second_price = sorted_bids[1][1]
        return winner_id, second_price


# === 演示 ===

if __name__ == "__main__":
    # 创建 3 个智能体
    agents = [
        Agent("数据专家", {"数据分析": 0.9, "文本写作": 0.3}, current_load=0.2),
        Agent("写作专家", {"数据分析": 0.4, "文本写作": 0.95}, current_load=0.5),
        Agent("全能选手", {"数据分析": 0.7, "文本写作": 0.7}, current_load=0.1),
    ]

    manager = ContractNetManager(agents)

    # 发布任务
    task = Task(
        task_id="report-001",
        description="分析上季度销售数据，生成分析报告",
        required_skill="数据分析",
        priority=2,
        max_budget=80,
    )

    # 收集投标
    bids = manager.announce(task)
    print(f"收到 {len(bids)} 份投标")
    for bid in bids:
        print(f"  {bid.agent_id}: 匹配度={bid.skill_match:.1f}, "
              f"预估成本={bid.estimated_cost:.1f}")

    # 评标
    winner = manager.evaluate_bids(bids, task)
    if winner:
        print(f"\n中标者: {winner.agent_id}")

    # 维克里拍卖演示
    print("\n=== 维克里拍卖 ===")
    auction = VickreyAuction(task)
    auction.submit_bid("数据专家", 150)
    auction.submit_bid("写作专家", 120)
    auction.submit_bid("全能选手", 200)

    result = auction.determine_winner()
    if result:
        print(f"获胜者: {result[0]}, 支付价格: {result[1]}")
