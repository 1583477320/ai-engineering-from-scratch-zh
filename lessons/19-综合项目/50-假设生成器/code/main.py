"""假设生成器：温度调度采样、新颖性过滤、排序队列。

仅使用标准库。运行：python3 code/main.py
"""
from __future__ import annotations
import hashlib, json, math, re
from dataclasses import dataclass
from typing import Callable

HASH_DIM = 128

TAG_RE = re.compile(
    r"<hypothesis>\s*"
    r"<text>(?P<text>.*?)</text>\s*"
    r"<variables>(?P<variables>.*?)</variables>\s*"
    r"<metric>(?P<metric>.*?)</metric>\s*"
    r"(?:<baseline>(?P<baseline>.*?)</baseline>\s*)?"
    r"</hypothesis>",
    re.DOTALL,
)


@dataclass
class Hypothesis:
    id: int; text: str; variables: list[str]; metric: str
    baseline_ref: str | None; draft_pass: int; temperature: float
    novelty_score: float = 0.0; rank_score: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "variables": list(self.variables),
                "metric": self.metric, "baseline_ref": self.baseline_ref,
                "draft_pass": self.draft_pass, "temperature": round(self.temperature, 3),
                "novelty_score": round(self.novelty_score, 4),
                "rank_score": round(self.rank_score, 4)}


class ParserError(ValueError):
    pass


def hashed_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        vec[idx] += 1.0 if (h[4] & 1) == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def parse_response(raw: str) -> dict:
    m = TAG_RE.search(raw)
    if m is None:
        raise ParserError("未找到假设块")
    text = m.group("text").strip()
    if not text:
        raise ParserError("空的 text")
    metric = m.group("metric").strip()
    if not metric:
        raise ParserError("空的 metric")
    variables = [v.strip() for v in m.group("variables").split(",") if v.strip()]
    if not variables:
        raise ParserError("空的 variables")
    baseline = m.group("baseline")
    return {"text": text, "variables": variables, "metric": metric,
            "baseline_ref": baseline.strip() if baseline and baseline.strip() else None}


def temperature_bucket(t: float) -> int:
    if t < 0.35: return 0
    if t < 0.65: return 1
    if t < 0.95: return 2
    return 3


class MockLLM:
    def __init__(self, scripts: dict[tuple[str, int], list[str]]):
        self._scripts = dict(scripts)

    @staticmethod
    def prompt_signature(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]

    def sample(self, prompt: str, temperature: float, seed: int) -> str:
        key = (self.prompt_signature(prompt), temperature_bucket(temperature))
        bank = self._scripts.get(key)
        if not bank:
            return "<noise>无法解析的输出</noise>"
        return bank[seed % len(bank)]


@dataclass
class GeneratorConfig:
    n_passes: int = 6
    t_min: float = 0.2
    t_max: float = 1.2
    novelty_threshold: float = 0.25
    target_variable_count: int = 3
    w_novelty: float = 0.4
    w_specificity: float = 0.3
    w_testability: float = 0.3
    base_seed: int = 0

    def schedule(self) -> list[float]:
        if self.n_passes <= 0: return []
        if self.n_passes == 1: return [self.t_min]
        step = (self.t_max - self.t_min) / (self.n_passes - 1)
        return [self.t_min + i * step for i in range(self.n_passes)]


@dataclass
class GenerationLog:
    pass_index: int; temperature: float; seed: int
    accepted_id: int | None; reject_reason: str | None; raw_excerpt: str

    def to_dict(self) -> dict:
        return {"pass": self.pass_index, "temperature": round(self.temperature, 3),
                "seed": self.seed, "accepted_id": self.accepted_id,
                "reject_reason": self.reject_reason, "raw_excerpt": self.raw_excerpt[:80]}


class HypothesisGenerator:
    def __init__(self, llm: MockLLM, config: GeneratorConfig | None = None,
                 embedder: Callable[[str], list[float]] = hashed_embed):
        self._llm = llm; self._cfg = config or GeneratorConfig(); self._embed = embedder

    def _specificity_score(self, h: Hypothesis) -> float:
        return min(1.0, len(h.variables) / max(1, self._cfg.target_variable_count))

    def _testability_score(self, h: Hypothesis) -> float:
        return 1.0 if h.metric and h.baseline_ref else (0.5 if h.metric else 0.0)

    def _score(self, h: Hypothesis) -> float:
        return (self._cfg.w_novelty * h.novelty_score
                + self._cfg.w_specificity * self._specificity_score(h)
                + self._cfg.w_testability * self._testability_score(h))

    def _novelty(self, vec: list[float], survivors: list[list[float]]) -> float:
        return min([cosine_distance(vec, s) for s in survivors], default=1.0)

    def run(self, seed_prompt: str) -> tuple[list[Hypothesis], list[GenerationLog]]:
        survivors, vecs, logs = [], [], []
        next_id = 1
        for idx, temp in enumerate(self._cfg.schedule()):
            seed = self._cfg.base_seed + idx
            raw = self._llm.sample(seed_prompt, temp, seed)
            try:
                parsed = parse_response(raw)
            except ParserError as e:
                logs.append(GenerationLog(idx, temp, seed, None, f"解析错误:{e}", raw))
                continue
            vec = self._embed(parsed["text"])
            nov = self._novelty(vec, vecs)
            if nov < self._cfg.novelty_threshold:
                logs.append(GenerationLog(idx, temp, seed, None, "重复", raw))
                continue
            h = Hypothesis(next_id, parsed["text"], parsed["variables"], parsed["metric"],
                           parsed["baseline_ref"], idx, temp, novelty_score=nov)
            h.rank_score = self._score(h)
            survivors.append(h); vecs.append(vec)
            logs.append(GenerationLog(idx, temp, seed, next_id, None, raw))
            next_id += 1
        survivors.sort(key=lambda h: (-h.rank_score, h.id))
        return survivors, logs


def build_demo_scripts() -> dict:
    prompt = "研究小型 Transformer 中的注意力稀疏性"
    sig = MockLLM.prompt_signature(prompt)
    return {
        (sig, 0): ['<hypothesis><text>将注意力头数从 8 降至 4 会在 12M 参数模型上使验证损失增加不到 2%。</text><variables>head_count, validation_loss</variables><metric>validation_loss</metric><baseline>head_count_8</baseline></hypothesis>'],
        (sig, 1): ['<hypothesis><text>在 12M 参数规模下，k=16 的 Top-k 稀疏注意力在困惑度上匹配密集注意力。</text><variables>k, perplexity, parameter_count</variables><metric>perplexity</metric><baseline>dense_attention</baseline></hypothesis>'],
        (sig, 2): ['<hypothesis><text>通过学习门控路由注意力可减少 30% 的 FLOPs 而不损害下游准确率。</text><variables>gate_temperature, flops, accuracy</variables><metric>downstream_accuracy</metric><baseline>dense_attention</baseline></hypothesis>'],
        (sig, 3): ['<hypothesis><text>块大小为 32 的块稀疏注意力在消费级 GPU 上将训练时间降低 18%。</text><variables>block_size, training_seconds, hardware</variables><metric>training_seconds</metric><baseline>dense_attention</baseline></hypothesis>'],
    }


def main() -> int:
    llm = MockLLM(build_demo_scripts())
    gen = HypothesisGenerator(llm, GeneratorConfig(n_passes=4, t_min=0.2, t_max=1.1))
    queue, logs = gen.run("研究小型 Transformer 中的注意力稀疏性")
    print(json.dumps({"queue_size": len(queue),
                       "queue": [h.to_dict() for h in queue],
                       "logs": [l.to_dict() for l in logs]},
                      indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
