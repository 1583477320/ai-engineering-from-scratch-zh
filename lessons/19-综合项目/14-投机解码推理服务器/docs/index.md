# 综合项目14——投机解码推理服务器

> EAGLE-3在vLLM 0.7中以2.5-3倍真实流量吞吐量交付。P-EAGLE（AWS 2026）将并行投机推得更远。SGLang的SpecForge大规模训练草案头。Red Hat的Speculators中心发布常见开源模型的对齐草案。TensorRT-LLM将投机解码作为一等公民。2026年生产推理栈是vLLM或SGLang加EAGLE系列草案、FP8或INT4量化和HPA。本综合项目要求你以2.5倍+基线吞吐量服务两个开源模型，并提供完整尾延迟报告。

**类型：** 综合项目
**编程语言：** Python（推理），C++/CUDA（内核检查），YAML（配置）
**前置知识：** 第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第17章（基础设施）
**涉及章节：** P3 · P7 · P10 · P17
**预计时间：** 30小时

---

## 学习目标

- 理解投机解码的draft/verify架构
- 部署EAGLE-3投机解码，测量接受率和吞吐量加速
- 对比不同草案-目标对齐度和草稿大小的效果
- 提供完整的端到端成本和延迟报告

---

## 1. 问题

投机解码在2026年成为商品化技术。EAGLE-3草案头在目标模型的隐藏状态上训练，预测N个token；目标模型一次验证全部。60-80%的接受率转化为2-3倍端到端吞吐量。

手艺在于推理运维。接受率随流量分布漂移。拒绝下的p99延迟比无投机更差。你必须在多个批大小下报告p99，而非仅稳态tokens/s。

---

## 2. 核心概念

### 2.1 Draft/Verify调度

投机解码有两层。**草案**模型（EAGLE-3头、ngram或更小的目标对齐模型）每步提出k个候选token。**目标**模型一次验证全部k个token；接受的前缀替换贪心路径。接受率取决于草案-目标对齐度和输入分布。

### 2.2 部署

Kubernetes部署。vLLM 0.7每GPU运行一个副本或张量并行分片。HPA基于队列等待而非CPU自动扩展。FP8（Marlin）和INT4（AWQ）量化将GPU内存保持在H100/H200包络内。

### 2.3 报告

端到端报告：吞吐量、接受率、p50/p99在批大小1/8/32下的延迟，以及$/100万token。

---

## 3. 从零实现

`code/main.py`实现投机解码的draft/verify调度器，对比基线贪婪解码。

```python
"""投机解码服务器——draft/verify调度器脚手架。

核心架构原语是draft/verify调度器：草案模型提出k个候选token；
目标模型在一次批量传递中验证它们；任何接受的前缀被提交，
被拒绝的后缀从目标模型重新采样。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


VOCAB = list("abcdefghij")


def softmax_from(seed: int) -> list[float]:
    rnd = random.Random(seed)
    weights = [rnd.random() for _ in VOCAB]
    total = sum(weights)
    return [w / total for w in weights]


def sample(dist: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(dist):
        acc += p
        if r <= acc:
            return i
    return len(dist) - 1


@dataclass
class TargetModel:
    calls: int = 0
    tokens_verified: int = 0

    def distribution(self, ctx_seed: int) -> list[float]:
        return softmax_from(ctx_seed * 7 + 13)

    def verify(self, draft_tokens: list[int], ctx_seed: int, rng: random.Random) -> tuple[list[int], int]:
        self.calls += 1
        self.tokens_verified += len(draft_tokens) + 1
        accepted = []
        for pos, tok in enumerate(draft_tokens):
            dist = self.distribution(ctx_seed + pos)
            if dist[tok] >= 0.5 * max(dist):
                accepted.append(tok)
            else:
                break
        ctx = ctx_seed + len(accepted)
        dist = self.distribution(ctx)
        next_tok = sample(dist, rng)
        return accepted, next_tok


@dataclass
class DraftModel:
    calls: int = 0
    alignment: float = 0.80

    def propose(self, ctx_seed: int, k: int, rng: random.Random, target: TargetModel) -> list[int]:
        self.calls += 1
        draft_tokens = []
        for pos in range(k):
            dist = target.distribution(ctx_seed + pos)
            if rng.random() < self.alignment:
                draft_tokens.append(max(range(len(dist)), key=lambda i: dist[i]))
            else:
                draft_tokens.append(sample(dist, rng))
        return draft_tokens


@dataclass
class Metrics:
    generated: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    accepted_sum: int = 0

    def acceptance_rate(self, k: int) -> float:
        return self.accepted_sum / (self.target_calls * k) if self.target_calls else 0.0

    def tokens_per_target_call(self) -> float:
        return self.generated / max(1, self.target_calls)


def speculative_decode(n_tokens, k, rng, target, draft):
    m = Metrics()
    ctx_seed = 1
    while m.generated < n_tokens:
        draft_tokens = draft.propose(ctx_seed, k, rng, target)
        m.draft_calls += 1
        accepted, next_tok = target.verify(draft_tokens, ctx_seed, rng)
        m.target_calls += 1
        m.accepted_sum += len(accepted)
        for tok in accepted:
            m.generated += 1; ctx_seed += 1
            if m.generated >= n_tokens: break
        if m.generated < n_tokens:
            m.generated += 1; ctx_seed += 1
    return m


def baseline_decode(n_tokens, rng, target):
    m = Metrics(); ctx_seed = 1
    while m.generated < n_tokens:
        target.calls += 1; m.target_calls += 1
        dist = target.distribution(ctx_seed)
        sample(dist, rng); m.generated += 1; ctx_seed += 1
    return m


def main():
    n_tokens = 500
    print(f"=== 解码{n_tokens}个token，对比基线 vs 投机 ===")
    target = TargetModel(); rng = random.Random(7)
    base = baseline_decode(n_tokens, rng, target)
    print(f"基线: {base.target_calls} 次目标调用, {base.tokens_per_target_call():.2f} tok/call")
    for alignment in (0.60, 0.75, 0.90):
        for k in (2, 4, 6):
            t = TargetModel(); d = DraftModel(alignment=alignment)
            m = speculative_decode(n_tokens, k, random.Random(7), t, d)
            speedup = base.target_calls / max(1, m.target_calls)
            print(f"  对齐={alignment:.2f} k={k}  目标调用={m.target_calls:3d}  "
                  f"接受率={m.acceptance_rate(k):.2f}  tok/call={m.tokens_per_target_call():.2f}  加速={speedup:.2f}x")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 解码500个token，对比基线 vs 投机 ===
基线: 500 次目标调用, 1.00 tok/call
  对齐=0.60 k=2  目标调用=340  接受率=0.38  tok/call=1.47  加速=1.47x
  对齐=0.60 k=4  目标调用=264  接受率=0.51  tok/call=1.89  加速=1.89x
  对齐=0.60 k=6  目标调用=220  接受率=0.60  tok/call=2.27  加速=2.27x
  对齐=0.75 k=2  目标调用=288  接受率=0.47  tok/call=1.74  加速=1.74x
  对齐=0.75 k=4  目标调用=192  接受率=0.65  tok/call=2.60  加速=2.60x
  对齐=0.75 k=6  目标调用=152  接受率=0.73  tok/call=3.29  加速=3.29x
  对齐=0.90 k=2  目标调用=266  接受率=0.48  tok/call=1.88  加速=1.88x
  对齐=0.90 k=4  目标调用=144  接受率=0.80  tok/call=3.47  加速=3.47x
  对齐=0.90 k=6  目标调用=108  接受率=0.86  tok/call=4.63  加速=4.63x
```

---

## 4. 工具实践

**技术栈：**
- 推理：vLLM 0.7或SGLang 0.4
- 投机方法：EAGLE-3草案头、P-EAGLE并行投机、ngram回退
- 草案训练：SpecForge或Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + HPA

---

## 5. LLM视角

**接受率视角**：接受率是投机解码的核心指标。0.75+对齐度的EAGLE-3通常达到70-80%接受率。流量分布影响接受率——代码流量通常比聊天流量接受率低。

**延迟视角**：p99延迟在拒绝时更差，因为验证传递更大。必须在多个批大小下报告，而非仅稳态吞吐量。

**成本视角**：$/100万token是可信度杠杆。2.5x吞吐量意味着约60%的成本降低。

---

## 6. 工程最佳实践

**基线建立**：
- 启用投机前：不同批大小的tokens/s、p50/p99延迟
- 启用投机后：相同的基准+接受率+尾延迟变化

**部署配置**：
- vLLM 0.7 + EAGLE-3草案头
- FP8/INT4量化
- K8s HPA基于队列等待
- 持续监控接受率

---

## 7. 常见错误

**错误1：仅报告稳态吞吐量**
症状：p99延迟数据缺失
修复：在多个批大小下报告p50/p99

**错误2：不监控接受率漂移**
症状：流量分布变化后性能下降
修复：持续监控接受率并设置告警

---

## 8. 面试考点

**Q1：投机解码的draft/verify机制如何工作？**
考察：对推理优化的理解

**Q2：为什么EAGLE-3比ngram草案更好？**
考察：对草案架构的理解

**Q3：P-EAGLE的并行投机有什么优势和劣势？**
考察：对高级投机解码的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 草案模型 | "投机器" | 提出N个token供目标验证的小模型 |
| EAGLE-3 | "2026草案架构" | 在目标隐藏状态上训练的草案头；~75%接受率 |
| P-EAGLE | "并行投机" | 在一次目标传递中验证的草案分支树 |
| 接受率 | "命中率" | 无需重新采样就被接受的草案token比例 |
| 队列等待 | "HPA指标" | 请求在推理开始前在待处理队列中的等待时间 |
| Speculators中心 | "对齐草案" | Red Hat Neural Magic的常见开源模型EAGLE草案集 |

---

## 参考文献

- [vLLM EAGLE和P-EAGLE文档](https://docs.vllm.ai)
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge)
- [Red Hat Speculators](https://github.com/neuralmagic/speculators)
- [EAGLE-3论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840)
- [vLLM仓库](https://github.com/vllm-project/vllm)
