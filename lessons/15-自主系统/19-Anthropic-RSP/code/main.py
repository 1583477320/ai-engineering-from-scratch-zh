"""RSP v3.0 阈值评估器——纯标准库。

建模 Anthropic RSP v3.0 的 AI R&D-4 阈值决策形状。
给定候选模型的能力测量，决定阈值是否跨越以及正面案例必须覆盖什么。

教学用途——真正的 RSP 涉及跨更大证据基础的人类判断。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapabilityMeasurement:
    model_name: str
    rd_automation_share: float       # 可自动化的 AI 研究比例
    metr_horizon_hours: float        # METR 50% 视界（小时）
    aar_outperform_share: float      # 对齐研究超越人类的比例
    eval_context_gaming_rate: float  # 评估上下文博弈率

# RSP v3.0 说明性阈值
AI_RD_4_THRESHOLDS = {
    "rd_automation_share": 0.5,
    "metr_horizon_hours": 40.0,
    "aar_outperform_share": 0.4,
}


def threshold_crossed(m: CapabilityMeasurement) -> tuple[bool, list[str]]:
    """检查 AI R&D-4 阈值是否跨越（任意两个触发条件跨越）。"""
    reasons = []
    if m.rd_automation_share >= AI_RD_4_THRESHOLDS["rd_automation_share"]:
        reasons.append(f"rd_automation_share={m.rd_automation_share:.2f}")
    if m.metr_horizon_hours >= AI_RD_4_THRESHOLDS["metr_horizon_hours"]:
        reasons.append(f"metr_horizon_hours={m.metr_horizon_hours:.1f}")
    if m.aar_outperform_share >= AI_RD_4_THRESHOLDS["aar_outperform_share"]:
        reasons.append(f"aar_outperform_share={m.aar_outperform_share:.2f}")
    return len(reasons) >= 2, reasons


def affirmative_case_template(m: CapabilityMeasurement) -> list[str]:
    """正面案例必须覆盖的六个部分。"""
    sections = [
        "1. 能力清单：针对 RSP 阈值的具体测量",
        "2. 错位风险分析：模型可能表现出的模式",
        "3. 评估-部署差距：评估 vs 部署差异的残余风险",
        "4. 缓解设计：技术 + 操作 + 部署门控",
        "5. 残余风险承认：我们无法排除什么",
        "6. 审查：内部安全咨询小组签核 + 外部审查者",
    ]
    if m.eval_context_gaming_rate > 0.2:
        sections.append(f"7. 博弈调整后的能力估计（观察到的博弈率 {m.eval_context_gaming_rate:.0%}）")
    return sections


def evaluate(m: CapabilityMeasurement) -> None:
    crossed, reasons = threshold_crossed(m)
    print(f"\n模型: {m.model_name}")
    print("-" * 70)
    print(f"  rd_automation={m.rd_automation_share:.2f}  metr_horizon={m.metr_horizon_hours:.1f}h  "
          f"aar={m.aar_outperform_share:.2f}  gaming={m.eval_context_gaming_rate:.0%}")
    if crossed:
        print("  AI R&D-4 阈值: 已跨越")
        for r in reasons:
            print(f"    - {r}")
        print("  要求：正面案例涵盖：")
        for s in affirmative_case_template(m):
            print(f"    {s}")
    else:
        print("  AI R&D-4 阈值: 未跨越")
        if reasons:
            print("  单一触发（低于阈值）:")
            for r in reasons:
                print(f"    - {r}")


def main() -> None:
    print("=" * 70)
    print("RSP v3.0 AI R&D-4 阈值评估器（阶段 15，第 19 课）")
    print("=" * 70)

    # Claude Opus 4.6（v3.0 声明）：未跨越
    evaluate(CapabilityMeasurement(
        model_name="Claude Opus 4.6 (per Anthropic v3.0)",
        rd_automation_share=0.30, metr_horizon_hours=14.0,
        aar_outperform_share=0.35, eval_context_gaming_rate=0.12))

    # 合成下一代模型
    evaluate(CapabilityMeasurement(
        model_name="Synthetic next-gen (illustrative)",
        rd_automation_share=0.55, metr_horizon_hours=48.0,
        aar_outperform_share=0.45, eval_context_gaming_rate=0.28))

    print()
    print("=" * 70)
    print("要点：阅读策略是实践技能")
    print("-" * 70)
    print("  v3.0 的阈值是定性的，不是 v2 的定量。")
    print("  2023 暂停承诺已移除；正面案例形状替代了它。")
    print("  SaferAI 将 v3.0 从 2.2 降至 1.9（弱 RSP 类别）。")
    print("  评估上下文博弈（第 1 课）使能力数字从部署现实偏向乐观。")


if __name__ == "__main__":
    main()
