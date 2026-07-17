"""跨策略决策表差异工具——纯标准库。

读取三张小表，编码 OpenAI PF v2、Anthropic RSP v3.0、
DeepMind FSF v3 如何分类一组能力。输出并排对比。
表是三份源文件的教学性提炼；真正的策略阅读需要源文件。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policy:
    name: str
    table: dict[str, tuple[str, str]]  # 能力 → (分类, 触发动作)


# ── 说明性提炼；真实决策请参考源文件 ──────────────────────

OPENAI_PF_V2 = Policy(
    name="OpenAI Preparedness v2 (Apr 2025)",
    table={
        "long_range_autonomy": ("Research", "observed; potential mitigations"),
        "sandbagging": ("Research", "observed; potential mitigations"),
        "autonomous_replication": ("Research", "observed; potential mitigations"),
        "undermining_safeguards": ("Research", "observed; potential mitigations"),
        "rnd_automation": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "cyber_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "bio_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
    },
)

ANTHROPIC_RSP_V3 = Policy(
    name="Anthropic RSP v3.0 (Feb 2026)",
    table={
        "long_range_autonomy": ("named risk", "affirmative case at threshold"),
        "sandbagging": ("named via eval-context gap", "addressed in measurement methodology"),
        "autonomous_replication": ("not explicitly named", "covered under AI R&D-4"),
        "undermining_safeguards": ("hardcoded prohibition", "refuses training / deploy"),
        "rnd_automation": ("AI R&D-4 threshold", "affirmative case required"),
        "cyber_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
        "bio_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
    },
)

DEEPMIND_FSF_V3 = Policy(
    name="DeepMind FSF v3 (Sept 2025 + Apr 2026)",
    table={
        "long_range_autonomy": ("folded into ML R&D / Cyber domains", "CCL + Tracked Capability Level"),
        "sandbagging": ("deceptive alignment monitoring", "automated instrumental-reasoning monitor"),
        "autonomous_replication": ("folded into ML R&D domain", "CCL threshold"),
        "undermining_safeguards": ("deceptive alignment monitoring", "automated monitor + red-team"),
        "rnd_automation": ("ML R&D autonomy level 1", "Tracked Capability Level added Apr 2026"),
        "cyber_uplift": ("Cyber CCL", "security + deployment mitigations"),
        "bio_uplift": ("Bio CCL", "security + deployment mitigations"),
    },
)

POLICIES = [OPENAI_PF_V2, ANTHROPIC_RSP_V3, DEEPMIND_FSF_V3]


def diff(capability: str) -> None:
    """输出一个能力在三家策略中的分类和触发动作。"""
    print(f"\n能力: {capability}")
    print("-" * 80)
    for p in POLICIES:
        entry = p.table.get(capability, ("not in table", "—"))
        print(f"  {p.name}")
        print(f"    分类: {entry[0]}")
        print(f"    动作: {entry[1]}")


def main() -> None:
    print("=" * 80)
    print("跨策略差异工具（阶段 15，第 20 课）")
    print("=" * 80)

    for cap in ("long_range_autonomy", "sandbagging", "autonomous_replication",
                "undermining_safeguards", "rnd_automation"):
        diff(cap)

    print()
    print("=" * 80)
    print("要点：同一能力，三种不同分类")
    print("-" * 80)
    print("  长期自主性:")
    print("   - OpenAI: Research（不触发）")
    print("   - Anthropic: 命名风险（正面案例）")
    print("   - DeepMind: 领域折叠（CCL + 跟踪能力级别）")
    print()
    print("  破坏安全防护:")
    print("   - OpenAI: Research（不触发）")
    print("   - Anthropic: 硬编码禁令（拒绝）")
    print("   - DeepMind: 欺骗性对齐监控")
    print()
    print("  一起阅读三家是实践技能。")


if __name__ == "__main__":
    main()
