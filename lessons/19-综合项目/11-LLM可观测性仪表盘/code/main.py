"""LLM可观测性仪表盘——span导入+尾采样+评测脚手架。

核心架构原语是尾采样收集器加评测作为子span。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ms: int
    duration_ms: int
    attributes: dict
    status: str = "ok"

    def is_llm(self) -> bool:
        return "gen_ai.system" in self.attributes


@dataclass
class TailSampler:
    sample_rate: float = 0.10
    rng: random.Random = field(default_factory=lambda: random.Random(3))

    def decide(self, trace: list[Span]) -> bool:
        if any(s.status == "error" for s in trace):
            return True
        for s in trace:
            if s.name == "eval" and (s.attributes.get("toxicity", 0) > 0.5 or s.attributes.get("pii_leak", 0) > 0.8):
                return True
        return self.rng.random() < self.sample_rate


@dataclass
class SpanStore:
    spans: list[Span] = field(default_factory=list)
    by_user: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cost_by_user: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def insert_trace(self, trace: list[Span]) -> None:
        self.spans.extend(trace)
        for s in trace:
            if s.is_llm():
                u = s.attributes.get("user_id", "anon")
                m = s.attributes.get("gen_ai.request.model", "unknown")
                self.by_user[u] += 1
                self.by_model[m] += 1
                self.cost_by_user[u] += s.attributes.get("cost_usd", 0.0)


def eval_faithfulness(response: str, context: str) -> float:
    r = set(response.lower().split())
    c = set(context.lower().split())
    return len(r & c) / len(r) if r else 0.0


def eval_toxicity(response: str) -> float:
    bad = {"hate", "kill", "stupid", "garbage"}
    words = response.lower().split()
    hits = sum(1 for w in words if w in bad)
    return min(1.0, hits / max(1, len(words)) * 10)


def eval_pii_leak(response: str) -> float:
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", response):
        return 0.95
    if re.search(r"[\w.+-]+@[\w.-]+", response):
        return 0.6
    return 0.05


def prompt_fingerprint(prompt: str, n_bins: int = 8) -> int:
    return hashlib.sha256(prompt.encode()).digest()[0] % n_bins


def psi(a: list[int], b: list[int], n_bins: int = 8) -> float:
    ca = [0] * n_bins
    cb = [0] * n_bins
    for v in a:
        ca[v] += 1
    for v in b:
        cb[v] += 1
    total_a, total_b = max(sum(ca), 1), max(sum(cb), 1)
    score = 0.0
    for i in range(n_bins):
        pa = max(ca[i] / total_a, 0.0001)
        pb = max(cb[i] / total_b, 0.0001)
        score += (pa - pb) * math.log(pa / pb)
    return score


def synth_trace(trace_id: str, leak_pii: bool, rng: random.Random) -> list[Span]:
    model = rng.choice(["claude-sonnet-4-7", "gpt-5-4", "gemini-3-pro"])
    user = rng.choice(["u_01", "u_02", "u_03", "u_04"])
    root = Span(trace_id=trace_id, span_id=f"{trace_id}_0", parent_span_id=None,
                name="chat_turn", start_ms=int(time.time() * 1000),
                duration_ms=rng.randint(400, 2400), attributes={"app_id": "chatbot"})
    prompt = rng.choice(["what is the weather today", "summarize forecast", "give a tip"])
    resp = "your ssn is 123-45-6789" if leak_pii else "the weather is mild"
    ctx = "relevant weather context mild"
    llm = Span(trace_id=trace_id, span_id=f"{trace_id}_1", parent_span_id=root.span_id,
               name="llm_call", start_ms=root.start_ms + 50, duration_ms=root.duration_ms - 80,
               attributes={"gen_ai.system": model.split("-")[0], "gen_ai.request.model": model,
                           "gen_ai.usage.input_tokens": rng.randint(80, 800),
                           "gen_ai.usage.output_tokens": rng.randint(20, 300),
                           "user_id": user, "prompt": prompt, "response": resp,
                           "context": ctx, "cost_usd": round(rng.uniform(0.002, 0.05), 4)})
    return [root, llm]


def enrich_with_evals(trace: list[Span]) -> list[Span]:
    out = list(trace)
    for s in trace:
        if s.is_llm():
            resp = s.attributes.get("response", "")
            ctx = s.attributes.get("context", "")
            ev = Span(trace_id=s.trace_id, span_id=f"{s.span_id}_eval",
                      parent_span_id=s.span_id, name="eval",
                      start_ms=s.start_ms + s.duration_ms, duration_ms=120,
                      attributes={"faithfulness": eval_faithfulness(resp, ctx),
                                  "toxicity": eval_toxicity(resp), "pii_leak": eval_pii_leak(resp)})
            out.append(ev)
    return out


def alerter(store: SpanStore) -> list[str]:
    alerts = []
    pii = [s for s in store.spans if s.name == "eval" and s.attributes.get("pii_leak", 0) > 0.8]
    tox = [s for s in store.spans if s.name == "eval" and s.attributes.get("toxicity", 0) > 0.5]
    if pii:
        alerts.append(f"PII泄漏: {len(pii)}个事件")
    if tox:
        alerts.append(f"毒性激增: {len(tox)}个事件")
    return alerts


def main() -> None:
    rng = random.Random(5)
    sampler = TailSampler(sample_rate=0.20, rng=rng)
    store = SpanStore()
    b_fps, c_fps = [], []
    for i in range(200):
        trace = synth_trace(f"t{i:04d}", leak_pii=rng.random()<0.01, rng=rng)
        trace = enrich_with_evals(trace)
        if sampler.decide(trace):
            store.insert_trace(trace)
        fp = prompt_fingerprint(trace[1].attributes.get("prompt", ""))
        (c_fps if i > 150 else b_fps).append(fp)
    print(f"span数: {len(store.spans)}  按模型: {dict(store.by_model)}")
    for a in alerter(store):
        print(f"告警: {a}")
    p = psi(b_fps, c_fps)
    print(f"PSI: {p:.3f}" + (" 漂移!" if p > 0.2 else ""))


if __name__ == "__main__":
    main()
