# LoRA 微调

> 全量微调 7B 模型需要 28GB 显存的 Adam 状态。LoRA 只训练 2% 的参数——用单张消费级 GPU 微调 LLM。它是 2026 年个人开发者和中小团队定制 LLM 的唯一可行路径。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 08（DPO）、阶段 10 · 11（量化）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 LoRA 层——理解低秩分解的数学原理
- [ ] 配置 LoRA 训练参数——rank、alpha、target_modules 的选择
- [ ] 在 HuggingFace 上使用 PEFT 库微调 LLM
- [ ] 合并 LoRA 权重到基础模型并导出

---

## 1. 问题

全量微调一个 7B 模型需要 28GB 的 Adam 优化器状态 + 14GB 梯度 + 14GB 激活值 ≈ 56GB 显存——只有 A100 80GB 能放下。LoRA 的答案：冻结原始权重，只训练两个小矩阵 A 和 B——参数量从 7B 降到 ~10M（<2%）。

```
原始层: y = Wx         → 固定，不训练
LoRA 层: y = Wx + αBAx  → 训练 A, B，固定 W
```

---

## 2. 概念

### 2.1 LoRA 数学

对于权重矩阵 W ∈ ℝ^(d×d)：
```
W_new = W + α · B · A
其中: A ∈ ℝ^(d×r), B ∈ ℝ^(r×d), r << d
可训练参数: 2dr（远小于 d²）
```

### 2.2 LoRA 参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| rank (r) | 8-64 | 表达能力与过拟合的权衡 |
| alpha (α) | 等于 rank 或 2×rank | 缩放因子 |
| target_modules | q_proj, v_proj | 注入到注意力的 Q/V 层 |
| dropout | 0.05 | 防止过拟合 |

### 2.3 QLoRA——量化 + LoRA

```
基础模型（INT4 量化，冻结）
    ↓
注入 LoRA（FP16，可训练）
    ↓
训练：只需要 ~12GB 显存
```

---

## 3. 从零实现

### Step 1：LoRA 层实现

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original, rank=16, alpha=16):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)  # 冻结

        d_in, d_out = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.original(x) + self.scaling * (x @ self.lora_A @ self.lora_B.T)
```

### Step 2：PEFT 库使用

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 约 0.1% 可训练
```

### Step 3：合并 LoRA 权重

```python
model = model.merge_and_unload()  # 合并 LoRA 到原始权重
model.save_pretrained("./lora-merged")  # 保存合并后模型
```

---

## 4. 工具

### 4.1 PEFT + TRL

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, train_dataset=dataset,
    args=SFTConfig(output_dir="./output", num_train_epochs=3, learning_rate=2e-4))
trainer.train()
```

### 4.2 Unsloth（2x 加速）

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/llama-3-8b")
model = FastLanguageModel.get_peft_model(model, r=16)
```

---

## 6. 工程最佳实践

### 6.1 LoRA vs 全量微调

| 场景 | 选择 | 原因 |
|------|------|------|
| 快速定制 | LoRA r=16 | 几小时/单 GPU |
| 严格对齐 | 全量微调 | 最高质量但需集群 |
| 边缘部署 | QLoRA r=16 | INT4 量化 + LoRA = 12GB |

### 6.2 踩坑经验

- **rank 太小**：欠拟合——任务复杂度超过 LoRA 表达能力
- **rank 太大**：过拟合——失去 LoRA 轻量化优势
- **学习率太高**：灾难性遗忘——LoRA 应用 2e-4，全量微调用 2e-5
- **忘记设置 target_modules**：LoRA 未注入任何层——需要指定目标层

---

## 7. 常见错误

### 错误 1：LoRA 配置写错 target_modules

**现象：** 微调后模型输出完全没变——LoRA 没有生效。

**原因：** target_modules 写的层名不匹配实际层名（如 `q_proj` vs `q_proj.weight`）。

**修复：** 先打印模型结构确认层名：`for name, module in model.named_modules(): print(name)`

### 错误 2：合并 LoRA 后权重精度变化

**现象：** 合并后的模型生成质量下降。

**原因：** LoRA 权重是 FP32，原始权重是 FP16——合并后精度不一致。

**修复：** 合并前将所有权重转为相同精度，合并后再转为目标精度。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| LoRA 实现 | `code/main.py` | 低秩分解演示 + 参数量对比 |

---

## 8. 面试考点

### Q1：LoRA 如何在只训练 2% 参数的情况下保持模型质量？（难度：⭐⭐）

**参考答案：**
LoRA 利用了微调时权重更新的低秩特性——实践中发现微调时的权重变化矩阵 ΔW 通常是低秩的。因此只需训练低秩分解的 A 和 B 矩阵，用极少参数近似完整的 ΔW。这类似于 SVD 的思想——大部分信息集中在前几个奇异值中。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| LoRA | "低秩适配" | 冻结基础模型权重，只训练低秩增量矩阵 A×B |
| QLoRA | "量化+LoRA" | 基础模型 INT4 量化 + LoRA 适配，显存需求降至 ~12GB |
| target_modules | "注入位置" | 指定哪些层应用 LoRA——通常是注意力的 Q/V 层 |
| 合并 (merge) | "烘焙权重" | 将 LoRA 增量合并到原始权重，减少推理开销 |

---

## 📚 小结

LoRA 用低秩分解将微调参数从 7B 降到 ~10M（<2%）。QLoRA 在 INT4 量化模型上用 LoRA 微调——单 GPU 可行。PEFT 库自动注入 LoRA 层。训练后合并权重导出。LoRA 是 2026 年个人和中小团队定制 LLM 的最佳路径。

---

## ✏️ 练习

1. **【实现】** 用 PEFT 在 Llama 3.1 8B 上对中文文本做 LoRA 微调——r=16，训练 1000 步
2. **【实验】** 对比 r=4, r=16, r=64 的训练效果和收敛速度

---

## 📖 参考资料

1. [论文] Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models". ICLR, 2022. https://arxiv.org/abs/2106.09685
2. [GitHub] PEFT: https://github.com/huggingface/peft
3. [GitHub] Unsloth: https://github.com/unslothai/unsloth
