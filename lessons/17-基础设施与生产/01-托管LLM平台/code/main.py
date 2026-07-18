"""托管 LLM 平台对比器——三家云厂商的成本和延迟对比。"""

from dataclasses import dataclass


@dataclass
class Workload:
    model: str
    tokens_per_day: int
    sustained_pct: float
    peak_requests_per_sec: int
    latency_sla_ms: int


PLATFORMS = {
    "Bedrock 按需": {"ttft_ms": 75, "type": "per_token", "cost_per_mt": 1.50},
    "Bedrock PT":  {"ttft_ms": 55, "type": "per_hour", "cost_per_hour": 35.0},
    "Azure PTU":   {"ttft_ms": 50, "type": "per_hour", "cost_per_hour": 30.0},
    "Vertex 按需": {"ttft_ms": 65, "type": "per_token", "cost_per_mt": 1.60},
}


def compare(w: Workload):
    print(f"模型: {w.model}  日词元: {w.tokens_per_day:,}  SLA: {w.latency_sla_ms}ms")
    print("-" * 60)
    daily_tokens_m = w.tokens_per_day / 1_000_000

    for name, p in PLATFORMS.items():
        if p["type"] == "per_hour":
            daily_cost = p["cost_per_hour"] * 24
        else:
            daily_cost = daily_tokens_m * p["cost_per_mt"]

        meets_sla = p["ttft_ms"] <= w.latency_sla_ms
        print(f"{name:15s}  延迟={p['ttft_ms']}ms  "
              f"${daily_cost:.1f}/天  SLA={'✓' if meets_sla else '✗'}")


if __name__ == "__main__":
    w = Workload("Claude 3.5 Sonnet", 50_000_000, 0.6, 100, 100)
    compare(w)
