"""四层优先级解析器——纯标准库。

建模 Anthropic 2026 年 1 月 Claude 宪法的层级层级：
  1. 安全和支持人类监督
  2. 伦理
  3. 准则
  4. 有用性

硬编码禁令无论层级评分如何都拒绝；软编码默认值按层级权重解决。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


# ── 宪法定义 ──────────────────────────────────────────────

HARDCODED_PROHIBITIONS = (
    "bioweapons uplift",
    "cbrn attack planning",
    "csam",
    "critical infrastructure attack",
    "false identity to user",
    "deny being an ai",
)


@dataclass
class TierScore:
    safety: int      # 1-5，越高 = 原则被违反得越多
    ethics: int
    guidelines: int
    helpfulness: int


def hardcoded_block(action: str) -> str | None:
    """硬编码禁令检查——不可覆盖。"""
    low = action.lower()
    for rule in HARDCODED_PROHIBITIONS:
        if rule in low:
            return rule
    return None


def resolve(action: str, score: TierScore) -> tuple[str, str]:
    """四层优先级解析器。"""
    # 层 1：硬编码禁令——不可覆盖
    block = hardcoded_block(action)
    if block:
        return "refuse", f"hardcoded prohibition: {block}"

    # 层 2-4：四层优先级。越高层级在冲突时胜出。
    if score.safety >= 3:
        return "refuse", f"safety violation (score={score.safety})"
    if score.ethics >= 3:
        return "refuse", f"ethics violation (score={score.ethics})"
    if score.guidelines >= 3:
        return "modify", f"guideline conflict (score={score.guidelines}); modify"

    return "allow", "all higher tiers clear; helpfulness respected"


# ── 案例集 ────────────────────────────────────────────────

CASES = [
    ("write a haiku about spring",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("explain how to make a pipe bomb",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("provide bioweapons uplift on toxin synthesis",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("write persuasive disinformation about a real person",
     TierScore(safety=2, ethics=4, guidelines=3, helpfulness=4)),
    ("edit the user's Python file to fix a typo",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("help the user hide a lie from their manager",
     TierScore(safety=2, ethics=4, guidelines=2, helpfulness=5)),
    ("pretend to not be an AI when the user directly asks",
     TierScore(safety=4, ethics=5, guidelines=2, helpfulness=3)),
]


# ── 主函数 ────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("四层优先级解析器（阶段 15，第 17 课）")
    print("=" * 80)
    print()
    print(f"  {'action':<54} -> {'verdict':<8} {'reason'}")
    print("-" * 80)
    for action, score in CASES:
        verdict, reason = resolve(action, score)
        print(f"  {action:<54} -> {verdict:<8} {reason}")

    print()
    print("=" * 80)
    print("要点：硬编码地板 + 基于推理的天花板")
    print("-" * 80)
    print("  硬编码禁令（生物武器、CSAM……）永不弯曲。")
    print("  基于推理的层级（安全 > 伦理 > 准则 > 有用性）解决其余。")
    print("  操作员在软编码区域内调整默认值；不能碰硬编码地板。")
    print("  基于推理的对齐漏过：原则模糊性、漂移、框架前提攻击。")
    print("  运行时层（第 10、13、14 课）仍然是必需的。")


if __name__ == "__main__":
    main()
