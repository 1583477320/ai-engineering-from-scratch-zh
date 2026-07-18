"""推理平台经济学——六家供应商成本对比器。"""

from dataclasses import dataclass


@dataclass
class InferenceWorkload:
    model_size: str
    tokens_per_day: int
    sustained_utilization: float
    peak_rps: int


VENDORS = {
    "Fireworks 按需": {"type": "per_token", "rate_per_m": 0.60},
    "Fireworks 批处理": {"type": "per_token", "rate_per_m": 0.30},
    "Together": {"type": "per_token", "rate_per_m": 0.55},
    "Baseten": {"type": "per_min", "rate_per_min": 0.80},
    "Modal": {"type": "per_sec", "rate_per_sec": 0.015},
    "Replicate": {"type": "per_pred", "rate_per_pred": 0.015},
}


def compare(w: InferenceWorkload):
    daily_tokens_m = w.tokens_per_day / 1_000_000
    daily_minutes = 24 * 60 * w.sustained_utilization

    print(f"日词元: {w.tokens_per_day:,}  利用率: {w.sustained_utilization:.0%}")
    print("-" * 55)

    for name, v in VENDORS.items():
        if v["type"] == "per_token":
            daily = daily_tokens_m * v["rate_per_m"]
        elif v["type"] == "per_min":
            daily = daily_minutes * v["rate_per_min"]
        elif v["type"] == "per_sec":
            daily = daily_minutes * 60 * v["rate_per_sec"]
        else:
            daily = w.peak_rps * 86400 * v["rate_per_pred"]

        eff = daily / daily_tokens_m if daily_tokens_m > 0 else 0
        print(f"{name:20s}  ${daily:.1f}/天  有效=${eff:.3f}/M")


if __name__ == "__main__":
    w = InferenceWorkload("70B", 50_000_000, 0.3, 50)
    compare(w)
