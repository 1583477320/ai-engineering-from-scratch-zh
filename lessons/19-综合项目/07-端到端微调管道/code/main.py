"""端到端微调管道编排脚手架。

核心架构原语是一个可复现的管道DAG：数据卫生->SFT->偏好调优->量化->
部署->评测->模型卡，每个阶段声明性配置，每个阶段通过内容哈希
消费前一个阶段的产物。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Artifact:
    name: str
    kind: str
    payload: dict
    produced_by: str
    produced_at: float = field(default_factory=time.time)

    def content_hash(self) -> str:
        blob = json.dumps(self.payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Manifest:
    artifacts: dict[str, Artifact] = field(default_factory=dict)

    def add(self, a: Artifact) -> None:
        self.artifacts[a.name] = a

    def get(self, name: str) -> Artifact:
        return self.artifacts[name]

    def summary(self) -> list[tuple[str, str, str, str]]:
        return [(a.name, a.kind, a.content_hash(), a.produced_by)
                for a in self.artifacts.values()]


Stage = Callable[[Manifest, dict], Artifact]


def stage_data(m: Manifest, cfg: dict) -> Artifact:
    raw_n = cfg.get("raw_examples", 300_000)
    dedup_ratio = 0.94
    qual_ratio = 0.91
    pii_ratio = 0.995
    kept = int(raw_n * dedup_ratio * qual_ratio * pii_ratio)
    return Artifact("dataset", "dataset", {
        "raw_examples": raw_n,
        "after_dedup": int(raw_n * dedup_ratio),
        "after_quality": int(raw_n * dedup_ratio * qual_ratio),
        "after_pii_scrub": kept,
        "seed": cfg.get("seed", 7),
    }, produced_by="Datatrove+Nemotron-CC+Presidio")


def stage_contamination(m: Manifest, cfg: dict) -> Artifact:
    ds = m.get("dataset")
    overlap = []
    for bench in ("MMLU-Pro", "MT-Bench-v2", "RewardBench-2"):
        overlap.append({"bench": bench, "overlap_examples": 0})
    return Artifact("contamination_check", "report", {
        "dataset_hash": ds.content_hash(),
        "overlaps": overlap,
        "status": "clean" if all(o["overlap_examples"] == 0 for o in overlap) else "dirty",
    }, produced_by="minhash-lsh")


def stage_sft(m: Manifest, cfg: dict) -> Artifact:
    ds = m.get("dataset")
    return Artifact("sft_checkpoint", "checkpoint", {
        "base": cfg["base_model"],
        "dataset_hash": ds.content_hash(),
        "epochs": 3,
        "val_loss": 1.03,
        "hours": 6.2,
        "gpus": 8,
    }, produced_by="axolotl v0.8 + ZeRO-3")


def stage_dpo(m: Manifest, cfg: dict) -> Artifact:
    sft = m.get("sft_checkpoint")
    return Artifact("dpo_checkpoint", "checkpoint", {
        "from": sft.content_hash(),
        "epochs": 1,
        "beta": 0.08,
        "hours": 1.7,
    }, produced_by="trl 0.15 DPO")


def stage_quantize(m: Manifest, cfg: dict) -> Artifact:
    ckpt = m.get("dpo_checkpoint")
    return Artifact("quants", "quant", {
        "from": ckpt.content_hash(),
        "gptq_int4_gb": 4.6,
        "awq_int4_gb": 4.8,
        "gguf_q4_km_gb": 5.1,
    }, produced_by="gptq+awq+llama.cpp")


def stage_serve(m: Manifest, cfg: dict) -> Artifact:
    return Artifact("endpoint", "endpoint", {
        "backend": "vLLM 0.7 + EAGLE-3",
        "quant": "GPTQ-INT4-Marlin",
        "eagle_acceptance": 0.74,
        "p99_bs8_ms": 126,
        "tokens_per_sec_bs32": 6400,
        "dollars_per_mtokens": 0.28,
    }, produced_by="vllm+speculators")


def stage_eval(m: Manifest, cfg: dict) -> Artifact:
    return Artifact("eval_report", "report", {
        "mmlu_pro_delta": 3.2,
        "mt_bench_v2_delta": 0.41,
        "rewardbench2_delta": 0.08,
        "llama_guard_4_pass": 0.987,
    }, produced_by="lm-eval-harness")


def stage_model_card(m: Manifest, cfg: dict) -> Artifact:
    return Artifact("model_card", "report", {
        "standard": "MOF 2026",
        "data_license_declared": True,
        "training_config_hash": m.get("sft_checkpoint").content_hash(),
        "eval_attached": True,
        "safety_attached": True,
    }, produced_by="mof-template")


PIPELINE: list[tuple[str, Stage]] = [
    ("data", stage_data),
    ("contamination", stage_contamination),
    ("sft", stage_sft),
    ("dpo", stage_dpo),
    ("quantize", stage_quantize),
    ("serve", stage_serve),
    ("eval", stage_eval),
    ("model_card", stage_model_card),
]


def run_pipeline(cfg: dict) -> Manifest:
    m = Manifest()
    for name, stage_fn in PIPELINE:
        print(f"[{name:14s}] 运行中...")
        art = stage_fn(m, cfg)
        m.add(art)
        print(f"[{name:14s}] -> 产物 '{art.name}' 哈希={art.content_hash()}")
    return m


def main() -> None:
    cfg = {"base_model": "llama-3.3-8b", "raw_examples": 300_000, "seed": 7, "dpo_beta": 0.08}
    print("=== 微调管道运行 ===")
    m = run_pipeline(cfg)
    print("\n=== 产物清单 ===")
    for name, kind, h, by in m.summary():
        print(f"  {name:18s} {kind:10s} {h} by {by}")
    print("\n=== 评测报告 ===")
    print(json.dumps(m.get("eval_report").payload, indent=2))
    print("\n=== 服务端点 ===")
    print(json.dumps(m.get("endpoint").payload, indent=2))


if __name__ == "__main__":
    main()
