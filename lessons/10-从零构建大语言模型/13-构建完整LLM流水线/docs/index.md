# 构建完整 LLM 流水线

> 从预处理到部署——一个完整的 LLM 企业级落地需要数据管道、训练、评估、量化和推理优化的无缝集成。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-12 | **时间：** ~120 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 01-12 的所有内容——这是一个综合实战课

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计端到端 LLM 流水线——数据→训练→评估→量化→部署
- [ ] 选择开源 LLM 生态——HuggingFace transformers + vLLM + DeepSpeed
- [ ] 用 LoRA 在领域数据上微调基础模型
- [ ] 将训练好的模型部署为可服务的 API

---

## 1. 问题

前 12 课你学了：分词器、数据管道、预训练、分布式训练、SFT、RLHF/DPO、评估、量化、推理优化。但这些是**独立的积木**。

这一课是**蓝图**——将所有积木拼成一个可以落地的流水线。

---

## 2. 概念

### 2.1 完整流水线概览

```
数据收集 → 清洗/去重 → 分词 → 预训练
    ↓
评估 → SFT（指令微调）
    ↓
评估 → RLHF/DPO（对齐）
    ↓
评估 → 量化（INT8/INT4）
    ↓
部署（vLLM / TensorRT-LLM）
    ↓
监控 + 迭代
```

### 2.2 2026 年的最佳实践

| 阶段 | 推荐方案 | 原因 |
|------|---------|------|
| 基础模型 | Llama 3.1 8B / Qwen2.5 | 开源、社区支持好 |
| SFT | LLaMA-Factory + LoRA | 一站式工具 |
| 对齐 | DPO > RLHF | 更简单、更快 |
| 量化 | AWQ INT4 | 质量和速度最佳 |
| 部署 | vLLM | 最易用的 GPU 推理 |

### 2.3 LoRA 微调——最实用的定制方案

```
基础模型（冻结）
    ↓
注入 LoRA 适配器（可训练）
    ↓
在领域数据上微调（只训练 LoRA 参数）
    ↓
合并 LoRA 到基础模型 → 导出
```

LoRA 的优势：参数量小（<2%）、训练快、可复用、多 LoRA 切换。

---

## 3. 实现

### Step 1：选择基础模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 推荐：Llama 3.1 8B（平衡质量和效率）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

### Step 2：LoRA 微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 注入的层
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print(f"可训练参数: {model.print_trainable_parameters()}")
# 可训练参数: 0.1% (约 2M)
```

### Step 3：训练

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=domain_dataset,
    args=SFTConfig(
        output_dir="./lora-output",
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
    ),
)
trainer.train()
```

### Step 4：部署

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./lora-output",  # LoRA 微调后的模型
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

outputs = llm.generate(["你好，请介绍一下自己"], SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
```

---

## 4. 工具

### 4.1 一站式工具对比

| 工具 | 功能 | 特点 |
|------|------|------|
| LLaMA-Factory | SFT + DPO + RLHF | 零代码训练 |
| Axolotl | 灵活配置 | YAML 配置 |
| OpenChat | 对话微调 | 中文友好 |
| Unsloth | 快速微调 | 2x 加速 |

### 4.2 开源模型选型

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|---------|
| Llama 3.1 8B | 8B | 通用、社区大 | 通用 |
| Qwen2.5 7B | 7B | 中文强 | 中文场景 |
| Mistral 7B | 7B | 效率高 | 边缘设备 |
| Phi-3 Mini | 3.8B | 小巧高效 | 轻量任务 |

---

## 5. LLM 视角

### 5.1 2026 年的 LLM 生态

2026 年，构建定制 LLM 的成本已经降到个人开发者可承受：
- **免费**：用 LoRA 在消费级 GPU 上微调
- **快速**：LLaMA-Factory 一键训练
- **高效**：AWQ INT4 让 8B 模型在笔记本上运行

### 5.2 从训练到生产的鸿沟

训练好的模型 ≠ 可以服务的产品。还需要：
- **量化**：减少显存占用
- **推理优化**：vLLM / TensorRT-LLM
- **监控**：日志、性能指标、用户反馈
- **安全**：输入过滤、输出审核、拒答机制

---

## 6. 踩坑经验

- **LoRA 配置不当**：r 太小→欠拟合；r 太大→过拟合。推荐 r=16-32
- **学习率设置错误**：LoRA 的学习率应比全微调大 10-100 倍（如 2e-4）
- **忘记评估就部署**：至少跑 AlpacaEval + MT-Bench 再上线
- **量化后质量下降**：检查是否用了正确的量化方案（AWQ > GGUF > NF4）

---

## 7. 面试考点

### Q1：如果要为一个中文客服场景定制 LLM，你的完整流水线是什么？（难度：⭐⭐⭐）

**参考答案：**
(1) **基础模型**：Qwen2.5 7B（中文最强开源模型）；(2) **SFT 数据**：收集 5K 中文客服对话，格式化为聊天模板；(3) **微调**：LoRA（r=16）在 1-2 个 A100 上训练 3 轮；(4) **对齐**：用 DPO 在 1K 偏好对上优化；(5) **评估**：AlpacaEval + MT-Bench + 人工客服评估；(6) **量化**：AWQ INT4；(7) **部署**：vLLM 服务。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| LoRA | "只训练 1% 参数" | 低秩适配——冻结基础模型，只训练小的可训练参数 |
| LLaMA-Factory | "一键微调" | 零代码的 LLM 训练框架 |
| LoRA 合并 | "烘焙 LoRA" | 将 LoRA 权重合并到基础模型——减少推理开销 |

---

## 📚 小结

完整 LLM 流水线 = 数据管道 → 预训练 → SFT → 对齐 → 评估 → 量化 → 部署。2026 年的最佳实践：用 Llama 3.1 / Qwen2.5 作为基础模型，LoRA 微调，DPO 对齐，AWQ INT4 量化，vLLM 部署。整个流程在消费级 GPU 上即可完成。

---

## ✏️ 练习

1. **【实验】** 用 LLaMA-Factory 在中文数据上微调 Llama 3.1 8B——跑通完整流水线。
2. **【思考】** 如果需要支持 10 万用户并发，你的部署方案是什么？需要多少 GPU？

---

## 📖 参考资料

1. [GitHub] LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
2. [GitHub] Unsloth: https://github.com/unslothai/unsloth
3. [文档] Hugging Face PEFT: https://huggingface.co/docs/peft

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
