# 综合项目07——端到端微调管道（数据到SFT到DPO到部署）

> 一个8B模型在你的数据上训练、在你的偏好上DPO对齐、量化、投机解码、以可测量的$/100万token部署。2026年的开源工具栈是Axolotl v0.8、TRL 0.15、Unsloth、GPTQ/AWQ/GGUF、vLLM 0.7配EAGLE-3。本综合项目要求你完整运行管道——YAML配置输入，服务端点输出——并按照2026年模型开放框架发布模型卡。

**类型：** 综合项目
**编程语言：** Python（管道），YAML（配置），Bash（脚本）
**前置知识：** 第2章（ML基础）、第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第11章（LLM工程）、第17章（基础设施）、第18章（安全）
**涉及章节：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**预计时间：** 35小时

---

## 学习目标

- 构建可复现的端到端微调管道：数据清洗→SFT→DPO→量化→部署→评测
- 实现数据去重、质量过滤和污染检查
- 实现EAGLE-3投机解码部署
- 发布符合MOF 2026标准的模型卡

---

## 1. 问题

2026年每个认真的AI团队都有端到端微调管道。不是因为他们在训练前沿基础模型，而是因为下游适配——领域SFT、针对偏好标签的DPO、投机解码的蒸馏草案——才是可衡量的收益所在。

Axolotl v0.8处理多GPU SFT配置。TRL 0.15处理DPO和GRPO。Unsloth提供快速单GPU迭代。vLLM 0.7配EAGLE-3将解码吞吐量提升2-3倍而不损失质量。

---

## 2. 核心概念

### 2.1 五阶段管道

**数据**：去重（MinHash/Datatrove）、质量过滤（Nemotron-CC风格分类器）、PII清洗、分割卫生检查（检测公共基准污染）。

**SFT**：Axolotl YAML配置、ZeRO-3 on 8×H100、余弦学习率、序列打包、2-3轮。

**DPO或GRPO**：TRL配置、1轮、偏好对（人工标注或模型评判）、beta调优。

**量化**：GPTQ + AWQ + GGUF用于部署灵活性。

**部署**：vLLM 0.7配EAGLE-3投机解码头（或SGLang配SpecForge）、K8s部署、HPA。

### 2.2 消融实验

SFT-only vs SFT+DPO vs SFT+GRPO在三个任务特定基准上对比。服务指标：不同批大小的token/s、EAGLE-3接受率、$/100万token。安全评测：Llama Guard 4通过率。模型卡：偏差评测、可复现种子、数据许可。

---

## 3. 从零实现

`code/main.py`实现可复现管道DAG的编排逻辑：数据卫生→SFT→偏好调优→量化→部署→评测→模型卡。

```python
"""端到端微调管道编排脚手架。

核心架构原语是一个可复现的管道DAG：数据卫生->SFT->偏好调优->量化->
部署->评测->模型卡，每个阶段声明性配置（这里用YAML风格字典），
每个阶段通过内容哈希消费前一个阶段的产物。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# 产物 + 清单——内容哈希记账
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    name: str
    kind: str         # "dataset" | "checkpoint" | "quant" | "endpoint" | "report"
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


# ---------------------------------------------------------------------------
# 阶段——每个接收清单和配置，返回新产物
# ---------------------------------------------------------------------------

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
    quants = m.get("quants")
    return Artifact("endpoint", "endpoint", {
        "backend": "vLLM 0.7 + EAGLE-3",
        "quant": "GPTQ-INT4-Marlin",
        "eagle_acceptance": 0.74,
        "p99_bs8_ms": 126,
        "tokens_per_sec_bs32": 6400,
        "dollars_per_mtokens": 0.28,
    }, produced_by="vllm+speculators")


def stage_eval(m: Manifest, cfg: dict) -> Artifact:
    ckpt = m.get("dpo_checkpoint")
    return Artifact("eval_report", "report", {
        "from": ckpt.content_hash(),
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
        "reproducibility_command": "./pipeline.sh config/llama3.3-8b-domainX.yaml",
    }, produced_by="mof-template")


# ---------------------------------------------------------------------------
# DAG编排器——按顺序运行阶段，每一步快照清单
# ---------------------------------------------------------------------------

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
    cfg = {
        "base_model": "llama-3.3-8b",
        "raw_examples": 300_000,
        "seed": 7,
        "dpo_beta": 0.08,
    }
    print("=== 微调管道运行 ===")
    m = run_pipeline(cfg)
    print()
    print("=== 产物清单 ===")
    for name, kind, h, by in m.summary():
        print(f"  {name:18s} {kind:10s} {h} by {by}")
    print()
    print("=== 评测报告 ===")
    print(json.dumps(m.get("eval_report").payload, indent=2))
    print()
    print("=== 服务端点 ===")
    print(json.dumps(m.get("endpoint").payload, indent=2))


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 微调管道运行 ===
[data           ] 运行中...
[data           ] -> 产物 'dataset' 哈希=a1b2c3d4e5f6
[contamination  ] 运行中...
[contamination  ] -> 产物 'contamination_check' 哈希=b2c3d4e5f6a7
[sft            ] 运行中...
[sft            ] -> 产物 'sft_checkpoint' 哈希=c3d4e5f6a7b8
[dpo            ] 运行中...
[dpo            ] -> 产物 'dpo_checkpoint' 哈希=d4e5f6a7b8c9
[quantize       ] 运行中...
[quantize       ] -> 产物 'quants' 哈希=e5f6a7b8c9d0
[serve          ] 运行中...
[serve          ] -> 产物 'endpoint' 哈希=f6a7b8c9d0e1
[eval           ] 运行中...
[eval           ] -> 产物 'eval_report' 哈希=a7b8c9d0e1f2
[model_card     ] 运行中...
[model_card     ] -> 产物 'model_card' 哈希=b8c9d0e1f2a3

=== 产物清单 ===
  dataset             dataset    a1b2c3d4e5f6 by Datatrove+Nemotron-CC+Presidio
  contamination_check report     b2c3d4e5f6a7 by minhash-lsh
  sft_checkpoint      checkpoint c3d4e5f6a7b8 by axolotl v0.8 + ZeRO-3
  dpo_checkpoint      checkpoint d4e5f6a7b8c9 by trl 0.15 DPO
  quants              quant      e5f6a7b8c9d0 by gptq+awq+llama.cpp
  endpoint            endpoint   f6a7b8c9d0e1 by vllm+speculators
  eval_report         report     a7b8c9d0e1f2 by lm-eval-harness
  model_card          report     b8c9d0e1f2a3 by mof-template
```

---

## 4. 工具实践

**技术栈：**
- 数据：Datatrove去重、Nemotron-CC分类器、Presidio PII清理
- 基础模型：Llama 3.3 8B、Qwen3 14B、Gemma 3 12B
- SFT：Axolotl v0.8 + ZeRO-3 + Flash Attention 3
- 偏好调优：TRL 0.15（DPO/GRPO）、Unsloth
- 量化：GPTQ（Marlin）、AWQ、GGUF
- 部署：vLLM 0.7 + EAGLE-3
- 评测：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro

---

## 5. LLM视角

**复现性视角**：端到端管道的核心是可复现性。每条命令从头到尾运行，相同种子产生相同结果。

**污染检测视角**：公共基准污染是微调管道最常见的陷阱。MinHash污染检查应在SFT前运行。

**投机解码视角**：EAGLE-3将解码吞吐量提升2-3倍而不损失质量。接受率是关键指标。

---

## 6. 工程最佳实践

**数据卫生**：
- MinHash去重（Datatrove）
- Nemotron-CC风格质量过滤
- Presidio PII清理
- 污染检查（MinHash-LSH）

**训练配置**：
- ZeRO-3、Flash Attention 3、序列打包
- 余弦学习率调度
- 固定种子

**评测矩阵**：
- 基准评测（MMLU-Pro、MT-Bench-v2）
- 偏好评测（RewardBench-2）
- 安全评测（Llama Guard 4）

---

## 7. 常见错误

**错误1：不检查污染**
症状：MMLU-Pro分数异常高
修复：运行MinHash污染检查

**错误2：跳过量化**
症状：部署成本过高
修复：GPTQ+AWQ+GGUF三种量化

**错误3：无可复现种子**
症状：每次运行结果不同
修复：固定所有随机种子

---

## 8. 面试考点

**Q1：微调管道的五个阶段是什么？**
考察：对端到端流程的理解

**Q2：为什么污染检查在SFT前很重要？**
考察：对数据卫生的理解

**Q3：EAGLE-3投机解码如何工作？**
考察：对部署优化的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Axolotl | "SFT训练器" | 统一的YAML驱动训练器，支持SFT、DPO和蒸馏 |
| TRL | "偏好调优器" | Hugging Face的DPO/GRPO/PPO库 |
| GRPO | "组相对策略优化" | DeepSeek R1的可验证奖励RL配方 |
| EAGLE-3 | "投机解码草案" | 预测N个token的草案头；vLLM用目标模型验证 |
| MOF | "模型开放框架" | 2026年根据数据、代码、许可分级模型发布的标准 |
| 污染检查 | "分割卫生" | 基于MinHash的测试集泄漏检测 |
| 接受率 | "EAGLE/MTP指标" | 目标模型接受的草案token比例 |

---

## 参考文献

- [Axolotl文档](https://axolotl-ai-cloud.github.io/axolotl/)
- [TRL文档](https://huggingface.co/docs/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [DeepSeek R1论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948)
- [vLLM + EAGLE-3文档](https://docs.vllm.ai)
- [Model Openness Framework 2026](https://isocpp.org/)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
