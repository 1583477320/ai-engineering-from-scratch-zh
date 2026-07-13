# 推测解码

> 用一个小模型快速"猜"多个词元，再用大模型并行验证——推测解码将大模型的生成速度提升 2-3 倍，而输出质量与完全用大模型生成完全相同。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 07 · 05（完整 Transformer）| **预计时间：** ~45 分钟 | **所处阶段：** Tier 2 | **关联课程：** 阶段 10 · 14（KV 缓存优化）

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释推测解码的核心流程——草稿模型生成 K 个候选词元，目标模型一次并行验证
- [ ] 说明拒绝采样为什么能保证输出分布与完全用大模型生成等价
- [ ] 实现一个完整的推测解码流程（包括拒绝采样和分布等价性验证）
- [ ] 分析草稿模型与目标模型的预测一致性如何影响加速比
- [ ] 在工业部署中选择合适的推测解码工具和草稿模型配置

---

## 1. 问题

大语言模型（LLM）的推理是串行的——生成每个词元都需要一次完整的前向传播。对于一个 70B 参数的模型，在 A100 GPU 上生成 512 个词元可能需要数秒到十几秒。这个延迟直接影响用户体验：聊天机器人需要等待、代码补全需要卡顿、实时翻译需要排队。

但你有没有想过：既然模型在生成第 5 个词元时，已经"知道"了前 4 个词元的上下文，那它能不能一次性"猜"出接下来 5 个词元，然后自己验证哪些猜对了？

这就是推测解码的核心思想。用一个小模型（草稿模型）快速"猜"多个词元，再用大模型（目标模型）一次前向传播并行验证——猜对的直接用，猜错的从大模型分布重新采样。生成速度提升 2-3 倍，而输出质量与完全用大模型生成完全相同。

传统加速方案（量化、蒸馏）要么牺牲精度，要么需要重新训练。推测解码不需要改变模型权重，不需要重新训练，不需要特殊硬件——它只需要一个足够"像"大模型的小模型。

---

## 2. 概念

### 2.1 推测解码的三步流程

推测解码的工作流程可以用一个简单的类比理解：你让一个实习生（小模型）先写一份草稿，然后你（大模型）快速审核——大部分时候实习生写得不错，直接通过；偶尔写错了，你修改一下。

```
步骤 1：草稿模型生成 K 个候选词元（串行，但草稿模型很小，速度快）
  t1 = draft_model(prompt)
  t2 = draft_model(prompt + t1)
  t3 = draft_model(prompt + t1 + t2)
  t4 = draft_model(prompt + t1 + t2 + t3)

步骤 2：目标模型一次前向验证所有候选（并行，只需一次前向传播）
  logits = target_model(prompt + [t1, t2, t3, t4])

步骤 3：逐位置拒绝采样
  位置 1: 比较 P_target(t1) 与 P_draft(t1)
           → 接受率 = min(1, P_target/P_draft) → 接受或重采样
  位置 2: 比较 P_target(t2) 与 P_draft(t2)
           → 同上
  ...
  一旦某位置拒绝：丢弃后续所有草稿词元，从目标模型分布重新采样
  全部接受：额外从目标模型生成第 K+1 个词元
```

### 2.2 为什么输出质量等价

推测解码的输出分布与完全用大模型生成的分布**完全相同**。这是通过拒绝采样（Rejection Sampling）实现的。

直觉理解：假设目标模型在位置 i 对词元 t 的概率是 P_target(t)，草稿模型的概率是 P_draft(t)。我们以概率 min(1, P_target(t)/P_draft(t)) 接受草稿词元。如果拒绝，就从残差分布 max(P_target - P_draft, 0) 中重新采样。

数学上可以证明，经过这个过程后，最终采样到词元 t 的概率恰好等于 P_target(t)。这就是推测解码的核心数学保证——它不是近似，而是精确等价。

### 2.3 每步节省的计算量

```
标准自回归解码生成 K 个词元:
  需要 K 次大模型前向传播

推测解码生成 K 个词元:
  需要 K 次小模型前向传播 + 1 次大模型前向传播
  小模型通常是大模型的 1/10-1/50 参数量
  → 大模型前向次数从 K 次降到 1 次
  → 总计算量降低 2-5 倍（取决于小模型大小和猜对率）
```

关键指标是**拒绝率**（Rejection Rate）——草稿词元被拒绝的比例。拒绝率越低，加速效果越好。拒绝率取决于草稿模型与目标模型的预测一致性。

---

## 3. 从零实现

### 第 1 步：模拟语言模型

我们用条件概率表代替真实神经网络，聚焦于推测解码的核心逻辑。

```python
import numpy as np
from typing import List, Tuple

VOCAB_SIZE = 10
TOKEN_LABELS = ["the", "a", "is", "not", "and",
                "in", "to", "of", "it", "you"]


class SimpleLM:
    """简化的语言模型：基于转移概率表生成词元。"""

    def __init__(self, sharpness=1.0, seed=42):
        rng = np.random.default_rng(seed)
        alpha = np.ones(VOCAB_SIZE) * sharpness
        self.transition = rng.dirichlet(alpha, size=VOCAB_SIZE)
        self.start_probs = rng.dirichlet(alpha)

    def predict_next(self, context: List[int]) -> np.ndarray:
        """根据上下文预测下一个词元的概率分布。"""
        if not context:
            return self.start_probs
        return self.transition[context[-1]]

    def generate_token(self, context: List[int]) -> int:
        """采样一个词元。"""
        return int(np.random.choice(
            VOCAB_SIZE, p=self.predict_next(context)))
```

`sharpness` 参数控制分布的尖锐程度——值越小，分布越尖锐，模拟更大、更确定的模型。草稿模型用较大的 `sharpness`（分布较平），目标模型用较小的 `sharpness`（分布更尖锐）。

### 第 2 步：实现拒绝采样

拒绝采样是推测解码的核心——它保证输出分布与直接用目标模型生成完全一致。

```python
def rejection_sample(draft_prob, target_prob,
                     draft_token) -> Tuple[bool, int]:
    """拒绝采样：决定是否接受草稿词元。

    接受概率 = min(1, target_prob(draft_token) / draft_prob(draft_token))
    拒绝后从残差分布 max(target - draft, 0) 重采样。
    """
    p_draft = draft_prob[draft_token]
    p_target = target_prob[draft_token]

    # 接受条件：以 min(1, P_target/P_draft) 的概率接受
    if np.random.random() < min(1.0, p_target / max(p_draft, 1e-12)):
        return True, draft_token

    # 拒绝后从残差分布重采样
    residual = np.maximum(target_prob - draft_prob, 0)
    residual /= residual.sum() + 1e-12
    return False, int(np.random.choice(VOCAB_SIZE, p=residual))
```

关键细节：`max(p_draft, 1e-12)` 防止除零错误；残差分布 `max(P_target - P_draft, 0)` 保证采样概率非负。

### 第 3 步：实现推测解码

将草稿生成、并行验证、拒绝采样组合成完整的推测解码流程。

```python
def speculative_decode(draft, target, prefix,
                       max_new, K=4):
    """推测解码：草稿模型猜 K 个词元，大模型一次并行验证。"""
    generated = list(prefix)
    target_fwd = 0

    while len(generated) - len(prefix) < max_new:
        # 草稿模型逐词元生成 K 个候选
        draft_tokens = []
        for _ in range(K):
            draft_tokens.append(
                draft.generate_token(generated + draft_tokens))

        # 目标模型一次前向验证全部候选
        n = len(draft_tokens)
        target_probs = np.zeros((n, VOCAB_SIZE))
        for i in range(n):
            target_probs[i] = target.predict_next(
                generated + draft_tokens[:i])
        target_fwd += 1

        # 逐位置拒绝采样
        accepted_count = 0
        for i in range(K):
            draft_prob = draft.predict_next(
                generated + draft_tokens[:i])
            accepted, token = rejection_sample(
                draft_prob, target_probs[i], draft_tokens[i])
            generated.append(token)
            if accepted:
                accepted_count += 1
            else:
                break  # 一旦拒绝，丢弃后续所有草稿词元

        # 全部接受时，额外从目标模型生成第 K+1 个词元
        if accepted_count == K:
            extra = target.generate_token(generated)
            generated.append(extra)
            target_fwd += 1

    return generated[len(prefix):max_new + len(prefix)], target_fwd
```

### 第 4 步：运行演示

```python
draft = SimpleLM(sharpness=1.0, seed=42)
target = SimpleLM(sharpness=0.3, seed=99)
prefix = [0, 1]
K, max_new = 5, 20

# 标准解码（基线）
generated = list(prefix)
for _ in range(max_new):
    generated.append(target.generate_token(generated))

# 推测解码
spec_tokens, spec_fwd = speculative_decode(
    draft, target, list(prefix), max_new, K)

print(f"标准解码: {max_new} 次大模型前向")
print(f"推测解码: {spec_fwd} 次大模型前向")
print(f"加速比: {max_new / spec_fwd:.1f}x")
```

完整代码见 `code/main.py`，包含分布等价性的蒙特卡洛验证。

---

## 4. 工业工具

### 4.1 llama.cpp

llama.cpp 是本地部署推测解码最成熟的工具。通过 `--draft` 参数指定草稿模型：

```bash
# 使用推测解码生成文本
./main -m llama-70b.gguf \
       --draft llama-7b.gguf \
       --draft-p 5 \
       -p "The meaning of life is" \
       -n 128

# 对比无推测解码的生成速度
./main -m llama-70b.gguf \
       -p "The meaning of life is" \
       -n 128
```

`--draft-p 5` 表示草稿长度 K=5。llama.cpp 会自动处理拒绝采样和分布等价性保证。

### 4.2 vLLM

vLLM 支持推测解码的在线服务部署：

```python
from vllm import LLM, SamplingParams

# 启用推测解码
llm = LLM(
    model="meta-llama/Llama-3-70B",
    speculative_model="meta-llama/Llama-3-8B",
    num_speculative_tokens=5,
    speculative_max_model_len=4096,
)

output = llm.generate("The meaning of life is", SamplingParams(max_tokens=128))
```

### 4.3 HuggingFace Transformers

HuggingFace 提供了实验性的推测解码支持：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B")
draft_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B")

inputs = tokenizer("The meaning of life is", return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=128,
    speculative_draft_model=draft_model,
    num_speculative_tokens=5,
)
```

### 4.4 性能对比

| 工具 | 部署方式 | 典型加速比 | 适用场景 |
|---|---|---|---|
| llama.cpp | 本地 CPU/GPU | 1.5-2.5x | 个人部署、边缘设备 |
| vLLM | 在线服务 | 2-3x | 生产环境 API 服务 |
| HuggingFace | 实验/研究 | 1.5-2x | 原型验证、学术研究 |
| TensorRT-LLM | NVIDIA GPU 优化 | 2-4x | 高性能生产环境 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

推测解码在工业界已有广泛应用。vLLM（最流行的 LLM 推理引擎之一）内置了推测解码支持，许多云服务商的 API 服务也默认启用推测解码来降低延迟。OpenAI、Anthropic 等公司的推理服务中，推测解码是常用的优化手段之一——用户感知到的"响应速度"背后，很可能就有推测解码在工作。

### 5.2 LLM 时代什么变了？

在小模型时代（BERT、GPT-2），推测解码没有意义——模型本身就很快。但当模型规模增长到 70B、400B 甚至更大时，单次前向传播的延迟成为瓶颈。推测解码让小模型"预判"大模型的行为，是一种**不改变模型权重、不牺牲精度**的推理加速方案。这在大语言模型时代尤为重要：用户对延迟的容忍度有限，但对输出质量的要求越来越高。

### 5.3 什么没变？

推测解码的数学基础——拒绝采样——自 1950 年代以来就没有变化。变化的是应用场景：从统计物理的蒙特卡洛模拟，到大语言模型的推理加速。理解拒绝采样的数学原理，让你在面对任何"先猜后验"的优化场景时，都能判断其正确性和有效性。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 或 Claude 时，推测解码可能正在幕后工作。如果模型响应速度明显快于"按模型大小应该有的速度"，很可能是因为背后有一个小模型在帮助生成候选词元。理解推测解码，你就理解了为什么有时候大语言模型的响应时间波动较大——这可能与草稿模型的猜对率有关：简单问题猜对率高、响应快，复杂问题猜对率低、响应慢。

---

## 6. 工程最佳实践

### 6.1 草稿模型选择策略

| 场景 | 推荐草稿模型 | 选择理由 |
|---|---|---|
| 通用文本生成 | 同系列小模型（如 Llama-3-8B 配 Llama-3-70B） | 同系列模型训练数据一致，预测分布最接近 |
| 代码补全 | 专用代码小模型（如 CodeLlama-7B） | 代码领域的预测模式更集中 |
| 无额外模型 | 自验证（用目标模型的浅层） | 不需要额外显存加载草稿模型 |
| 数学推理 | 同系列小模型 + 温度调低 | 数学推理的分布更尖锐，猜对率更高 |

### 6.2 草稿长度 K 的调优

- **K 太小（1-2）**：加速有限，每轮只节省 1-2 次前向传播
- **K 适中（3-5）**：大多数场景的最优平衡点
- **K 太大（8+）**：如果拒绝率高，后半段草稿词元全部浪费
- **动态调整**：根据近期拒绝率动态调整 K——拒绝率低时增大 K，拒绝率高时减小 K

### 6.3 中文场景特别建议

- 中文文本的词元密度高（每个中文字通常对应 1-2 个词元），草稿模型的猜对率可能与英文不同
- 选择草稿模型时，确保它在中文数据上有足够的训练量——否则中文场景的拒绝率会显著升高
- 对于中文+代码混合场景，优先使用同系列模型配对（如 Qwen-7B 配 Qwen-72B），因为它们共享训练语料

### 6.4 踩坑经验

- **草稿模型太大**：草稿生成时间超过验证时间，加速比反而 < 1x（比不用还慢）
- **KV 缓存不共享**：两个模型各自维护 KV 缓存，显存可能翻倍——部署时注意显存预算
- **忽略额外前向**：全部接受时需要额外一次目标模型前向生成下一个词元，遗漏会导致输出分布偏差
- **温度不匹配**：草稿模型和目标模型的采样温度不一致，会降低猜对率
- **批量推理不适用**：推测解码是串行优化，在 batch size > 1 时收益大幅下降

---

## 7. 常见错误

### 错误 1：草稿模型选择不当，加速比 < 1x

**现象：** 启用推测解码后，生成速度反而变慢。

**原因：** 草稿模型太大（参数量接近目标模型），草稿生成的时间超过了节省的验证时间。或者草稿模型与目标模型预测差异太大，拒绝率超过 60%，大部分草稿词元被丢弃。

**修复：**
```python
# ❌ 草稿模型太大（50B 配 70B）
draft_model = load_model("llama-50b")
target_model = load_model("llama-70b")

# ✓ 草稿模型足够小（8B 配 70B，参数比约 1:9）
draft_model = load_model("llama-8b")
target_model = load_model("llama-70b")

# ✓ 监控拒绝率，动态调整
rejection_rate = rejected_count / total_draft_tokens
if rejection_rate > 0.6:
    print("警告：拒绝率过高，考虑减小 K 或更换草稿模型")
```

### 错误 2：忘记全部接受时的额外前向传播

**现象：** 推测解码的输出分布与标准解码不一致，蒙特卡洛验证显示显著差异。

**原因：** 当所有 K 个草稿词元都被接受时，需要额外一次目标模型前向传播生成第 K+1 个词元。遗漏这一步会导致输出分布偏差。

**修复：**
```python
# ❌ 遗漏额外前向
if accepted_count == K:
    pass  # 不做任何事

# ✓ 全部接受时，额外生成一个词元
if accepted_count == K:
    extra = target.generate_token(generated)
    generated.append(extra)
    target_fwd += 1
```

### 错误 3：拒绝采样中残差分布计算错误

**现象：** 输出分布与标准解码存在系统性偏差，某些词元的频率持续偏高或偏低。

**原因：** 残差分布没有正确归一化，或者没有将负值截断为 0。

**修复：**
```python
# ❌ 没有截断负值，没有归一化
residual = target_prob - draft_prob
token = np.random.choice(VOCAB_SIZE, p=residual)  # 可能有负概率

# ✓ 正确：截断 + 归一化
residual = np.maximum(target_prob - draft_prob, 0)
residual /= residual.sum() + 1e-12  # 防止全零
token = np.random.choice(VOCAB_SIZE, p=residual)
```

### 错误 4：草稿模型和目标模型的词表不一致

**现象：** 运行时报错，或者生成出无意义的词元。

**原因：** 草稿模型和目标模型使用了不同的分词器，导致词元 ID 对应不同的词。推测解码要求两个模型共享相同的词表和分词器。

**修复：**
```python
# ❌ 两个模型用不同分词器
draft_tokenizer = AutoTokenizer.from_pretrained("model-a")
target_tokenizer = AutoTokenizer.from_pretrained("model-b")

# ✓ 使用同一个分词器
tokenizer = AutoTokenizer.from_pretrained("target-model")
# 草稿模型也使用 target 的分词器编码输入
```

### 错误 5：在批量推理中误用推测解码

**现象：** 批量推理时吞吐量反而下降。

**原因：** 推测解码是串行优化——它减少的是大模型的前向传播次数，但增加了小模型的串行计算。在 batch size > 1 时，GPU 的并行计算能力已经被充分利用，推测解码的收益大幅下降。

**修复：**
```python
# ❌ 批量推理时启用推测解码
outputs = model.generate(
    batch_inputs,
    speculative_draft_model=draft_model,  # 批量场景收益低
)

# ✓ 批量推理用标准解码，单条请求用推测解码
if batch_size == 1:
    outputs = model.generate(input, speculative_draft_model=draft_model)
else:
    outputs = model.generate(batch_inputs)
```

---

## 8. 面试考点

### Q1：请用 3 句话解释推测解码的核心思想。（难度：⭐）

**参考答案：**

推测解码用一个小模型（草稿模型）快速生成 K 个候选词元，再用大模型（目标模型）一次前向传播并行验证。猜对的词元直接接受，猜错的从目标模型分布重新采样。数学上可以证明，最终输出分布与完全用大模型生成完全相同。

### Q2：为什么推测解码能保证输出质量不下降？请解释拒绝采样的数学原理。（难度：⭐⭐）

**参考答案：**

拒绝采样的核心是：以概率 min(1, P_target(t)/P_draft(t)) 接受草稿词元 t。如果拒绝，则从残差分布 max(P_target - P_draft, 0) 中重新采样。可以证明，经过这个过程后，采样到词元 t 的总概率恰好等于 P_target(t)——接受路径贡献 P_draft(t) * min(1, P_target/P_draft) = min(P_draft, P_target)，拒绝路径贡献 P_draft(t) * (1 - min(1, P_target/P_draft)) * [残差归一化后概率]，两者之和恰好等于 P_target(t)。

### Q3：推测解码的加速比取决于哪些因素？如何优化？（难度：⭐⭐）

**参考答案：**

加速比主要取决于三个因素：（1）草稿长度 K——K 越大，每轮能验证的词元越多，但拒绝率可能升高；（2）拒绝率——草稿模型与目标模型预测越一致，拒绝率越低，加速比越高；（3）草稿模型大小——草稿模型越小，生成候选的速度越快，但可能与目标模型差异更大。优化策略：选择同系列小模型作为草稿模型（预测分布最接近），动态调整 K（根据近期拒绝率），在显存允许的前提下使用尽可能小的草稿模型。

### Q4：什么时候推测解码不适用？请举出 3 个场景。（难度：⭐⭐）

**参考答案：**

（1）**批量推理**：batch size > 1 时，GPU 并行能力已被充分利用，推测解码的串行优化收益大幅下降；（2）**草稿模型与目标模型差异过大**：拒绝率超过 60% 时，大部分草稿词元被丢弃，加速比可能 < 1x；（3）**短序列生成**：如果只需要生成 1-3 个词元，推测解码的初始化开销（加载草稿模型、首次验证）可能超过收益；（4）**数学/逻辑推理任务**：这类任务的输出分布非常尖锐，草稿模型很难猜对，拒绝率高。

### Q5：如何用蒙特卡洛方法验证推测解码的输出分布等价性？（难度：⭐⭐⭐）

**参考答案：**

具体步骤：（1）用标准解码从相同前缀生成 N 条序列（N 建议 ≥ 2000）；（2）用推测解码生成 N 条序列；（3）统计每种 3-gram（或完整序列）在两组中的频率；（4）计算两组频率的最大绝对差异；（5）如果最大差异 < 0.05（在 N=2000 时），认为分布等价。理论上，随着 N 趋向无穷，差异趋向 0。注意：这个验证是统计性的，不能证明数学等价，但实践中足够可靠。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 草稿模型 | "用一个小模型代替大模型" | 小型 LLM（如 0.5B-8B），快速生成候选词元序列，参数量通常为验证模型的 1/10-1/50 |
| 验证模型 | "大模型审核一下" | 大型 LLM（如 70B），一次前向传播并行验证所有草稿词元，是最终输出的分布来源 |
| 拒绝采样 | "猜错就扔掉重来" | 以概率 min(1, P_target/P_draft) 接受草稿词元，拒绝后从残差分布 max(P_target - P_draft, 0) 重采样 |
| 自验证 | "同一个模型自己验证自己" | 草稿模型和验证模型是同一个模型的不同配置（如浅层 vs 完整层），不需要额外的草稿模型 |
| 草稿长度 | "一次猜几个词" | 每轮草稿模型生成的候选词元数量 K，通常取 3-5，太大导致浪费，太小加速有限 |
| 拒绝率 | "猜对率有多高" | 草稿词元被目标模型拒绝的比例，拒绝率越低加速效果越好，取决于两个模型的预测一致性 |
| 残差分布 | "猜错后的补救方案" | 拒绝采样后重采样的目标分布，定义为 max(P_target - P_draft, 0) 归一化后的概率分布 |
| 分布等价性 | "输出完全一样吗" | 推测解码的数学保证——最终词元采样分布与完全用目标模型生成完全相同，通过拒绝采样实现 |

---

## 📚 小结

推测解码用小模型快速"猜"多个词元，再用大模型并行验证——正确接受，错误重采。核心洞察是：拒绝采样保证输出分布与完全用大模型生成完全相同（不是近似，是精确等价）。在 8B→70B 的模型配对上，生成速度通常提升 2-3 倍。自验证让推测解码在没有单独草稿模型时也能工作。

工业部署中，llama.cpp、vLLM、TensorRT-LLM 都支持推测解码。选择草稿模型的关键是预测分布一致性——同系列小模型通常是最优选择。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释拒绝采样中"残差分布"的作用。如果草稿模型和目标模型完全相同，残差分布会是什么样子？写 200 字以内的说明。

2. **【实现】** 修改 `code/main.py` 中的 `speculative_decode` 函数，添加一个拒绝率监控——统计每轮的拒绝率，并在拒绝率超过 60% 时自动减小草稿长度 K。

3. **【实验】** 调整 `SimpleLM` 的 `sharpness` 参数，观察草稿模型与目标模型的预测一致性如何影响加速比。记录 5 组不同 `sharpness` 组合下的加速比和拒绝率。

4. **【思考】** 推测解码要求草稿模型和目标模型共享词表。这在实际部署中可能带来什么限制？如果两个模型的词表不同（如 Llama 和 GPT），有什么解决方案？

5. **【设计】** 设计一个动态草稿长度调整策略：根据前 N 轮的平均拒绝率，自动调整 K 值。用伪代码描述你的策略，并分析它在"简单文本"和"复杂文本"两种场景下的行为差异。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 推测解码模拟实现 | `code/main.py` | 从零实现的推测解码流程，包含拒绝采样和分布等价性验证 |
| 推测解码速查指南 | `outputs/speculative-decoding-guide.md` | 核心机制、配置要点、部署工具速查表 |

---

## 📖 参考资料

1. [论文] Leviathan et al. "Fast Inference from Transformers via Speculative Decoding". ICML, 2023. https://arxiv.org/abs/2211.17192
2. [论文] Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling". 2023. https://arxiv.org/abs/2302.01318
3. [论文] Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads". ICML, 2024. https://arxiv.org/abs/2401.10774
4. [官方文档] llama.cpp Speculative Decoding: https://github.com/ggerganov/llama.cpp/blob/master/docs/speculative_decoding.md
5. [GitHub] vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
