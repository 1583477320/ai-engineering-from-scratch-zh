# consensus.py — 共识机制实现
# 依赖：标准库（无第三方依赖）
# 对应课程：阶段 16 · 23（多智能体共识与投票）

"""
多数决、加权投票、LLM 辩论式共识的实现。
"""

from collections import Counter
from dataclasses import dataclass, field


# === 多数决 ===

def majority_vote(votes: list[str]) -> tuple[str, float]:
    """多数决——得票最多的选项获胜。

    Returns:
        (获胜选项, 得票率)
    """
    counter = Counter(votes)
    winner = counter.most_common(1)[0][0]
    total = len(votes)
    ratio = counter[winner] / total
    return winner, ratio


# === 加权投票 ===

def weighted_vote(votes: list[tuple[str, float]]) -> tuple[str, float]:
    """加权投票——每个智能体带权重。

    Args:
        votes: (选项, 权重) 列表

    Returns:
        (获胜选项, 加权得分)
    """
    scores = {}
    for option, weight in votes:
        scores[option] = scores.get(option, 0) + weight
    winner = max(scores, key=scores.get)
    return winner, scores[winner]


# === LLM 辩论式共识 ===

class DebateConsensus:
    """LLM 辩论式共识——智能体通过多轮辩论达成一致。"""

    def __init__(self, agents: list[str], max_rounds: int = 3):
        self.agents = agents
        self.max_rounds = max_rounds
        self.history = []

    def run(self, question: str, get_opinion_fn) -> dict:
        """执行辩论流程。"""
        opinions = {}
        consensus = False

        for round_num in range(self.max_rounds):
            round_opinions = {}
            for agent_id in self.agents:
                context = self._format_history(round_num)
                opinion, confidence = get_opinion_fn(agent_id, question, context)
                round_opinions[agent_id] = {
                    "opinion": opinion,
                    "confidence": confidence,
                }
            self.history.append(round_opinions)

            # 检查是否达成共识
            first_opinion = list(round_opinions.values())[0]["opinion"]
            if all(
                v["opinion"] == first_opinion and v["confidence"] > 0.8
                for v in round_opinions.values()
            ):
                consensus = True
                break

        return {
            "consensus": consensus,
            "final_positions": round_opinions,
            "rounds": round_num + 1,
            "history": self.history,
        }

    def _format_history(self, round_num: int) -> str:
        """格式化前一轮辩论结果。"""
        if round_num == 0 or not self.history:
            return ""
        prev = self.history[-1]
        return "\n".join(
            f"{aid}: {op['opinion']} (置信度: {op['confidence']})"
            for aid, op in prev.items()
        )


# === 演示 ===

if __name__ == "__main__":
    # 多数决
    votes = ["看涨", "看跌", "看涨", "看涨", "看跌"]
    winner, ratio = majority_vote(votes)
    print(f"多数决: {winner} ({ratio:.0%})")

    # 加权投票
    weighted = [("看涨", 0.95), ("看涨", 0.80), ("看涨", 0.80),
                ("看跌", 0.92), ("看跌", 0.92)]
    w_winner, w_score = weighted_vote(weighted)
    print(f"加权投票: {w_winner} (得分: {w_score:.2f})")
