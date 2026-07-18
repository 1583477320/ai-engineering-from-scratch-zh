"""共识与 BFT——纯标准库。

实现三种聚合器（多数投票、CP-WBFT、DecentLLMs）和三种攻击模式
（拜占庭撒谎、谄媚从众、相关误差单一文化）。打印（攻击, 聚合器）
→ 最终答案表，突出正确决策。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass
class Vote:
    agent: str
    answer: str
    confidence: float

    def canonical(self) -> str:
        """粗略语义聚类：小写 + 去除空白/标点。"""
        return "".join(c for c in self.answer.lower().strip() if c.isalnum() or c == "." or c == "%")


def plurality(votes):
    """经典多数投票。"""
    counts, rep = {}, {}
    for v in votes:
        key = v.canonical()
        counts[key] = counts.get(key, 0) + 1
        rep.setdefault(key, v.answer)
    winner_key = max(counts, key=counts.get)
    return rep[winner_key], counts


def cp_wbft(votes, threshold=0.5):
    """置信度加权 BFT。"""
    weights, rep = {}, {}
    for v in votes:
        key = v.canonical()
        weights[key] = weights.get(key, 0.0) + v.confidence
        rep.setdefault(key, v.answer)
    total = sum(weights.values()) or 1.0
    winner_key = max(weights, key=weights.get)
    if weights[winner_key] / total < threshold:
        return None, weights
    return rep[winner_key], weights


def decentllms(votes):
    """几何中位数聚合——对异常值鲁棒。"""
    clusters = {}
    for v in votes:
        clusters.setdefault(v.canonical(), []).append(v)
    scores = {}
    for key, cluster in clusters.items():
        med = median([v.confidence for v in cluster])
        dist = sum(abs(v.confidence - med) for v in cluster)
        scores[key] = len(cluster) * max(0.0, 1.0 - dist)
    winner_key = max(scores, key=scores.get)
    return clusters[winner_key][0].answer, scores


def scenario(name, correct, votes):
    print(f"\n{'='*72}")
    print(f"场景: {name}")
    print(f"  正确答案: {correct!r}")
    print(f"{'='*72}")
    for v in votes:
        print(f"  {v.agent:12s} -> {v.answer!r:20s}  conf={v.confidence:.2f}")

    plural, counts = plurality(votes)
    cp, weights = cp_wbft(votes)
    dec, scores = decentllms(votes)

    def mark(a):
        return "[正确]" if a == correct else "[错误]" if a else "[低于阈值拒绝]"

    print(f"\n  多数投票    -> {str(plural):22s} {mark(plural)}")
    print(f"  CP-WBFT     -> {str(cp):22s} {mark(cp)}")
    print(f"  DecentLLMs  -> {str(dec):22s} {mark(dec)}")


def main():
    scenario("无攻击", "4.2%", [
        Vote("a", "4.2%", 0.85), Vote("b", "4.2%", 0.80),
        Vote("c", "4.2%", 0.75), Vote("d", "5%", 0.40), Vote("e", "4.2%", 0.70)])

    scenario("拜占庭撒谎", "4.2%", [
        Vote("a", "4.2%", 0.75), Vote("b", "4.2%", 0.70),
        Vote("c", "4.2%", 0.80), Vote("d", "42%", 0.95), Vote("e", "4.2%", 0.65)])

    scenario("谄媚从众", "4.2%", [
        Vote("a", "42%", 0.35), Vote("b", "42%", 0.30),
        Vote("c", "4.2%", 0.85), Vote("d", "4.2%", 0.80), Vote("e", "4.2%", 0.82)])

    scenario("相关误差单一文化", "4.2%", [
        Vote("a", "42%", 0.70), Vote("b", "42%", 0.68),
        Vote("c", "42%", 0.72), Vote("d", "4.2%", 0.85), Vote("e", "4.2%", 0.82)])

    print("\n要点:")
    print("  多数投票在单一文化攻击下失败。")
    print("  CP-WBFT 在从众攻击下部分缓解（从众者置信度低）。")
    print("  DecentLLMs 评分惩罚高方差簇——在单一文化不占多数时有帮助。")
    print("  当错误簇既大又高置信度时，没有聚合器能解决单一文化——需要多样性或验证。")


if __name__ == "__main__":
    main()
