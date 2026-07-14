# 指令微调——SFT

> 预训练模型预测下一个词。仅此而已。它不遵循指令、不回答问题、不拒绝有害请求。SFT 是 token 预测器和有用助手之间的桥梁。你用过的每一个模型——Claude、GPT、Llama Chat——都经过了这一步。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练 MiniGPT）| **时间：** ~90 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 04（预训练）— 从基础模型到 SFT | 阶段 10 · 07（RLHF）— SFT 之后的对齐步骤

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 SFT——将基础语言模型转换为遵循指令的助手
- [ ] 使用聊天模板格式化训练数据——系统/用户/助手角色 + 在非助手 token 上掩码损失
- [ ] 解释为什么需要 SFT——基础模型续写而非回答问题
- [ ] 对比 SFT 前后的模型表现

---

## 1. 问题

你在第 04 课训练了一个模型。它能预测下一个词。给它"Transformer 架构"，它可能会续写"彻底改变了自然语言处理"。这对下一个词预测器来说令人印象深刻。

但当你问它"什么是 Transformer？"时，它会续写：

```
什么是 Transformer？这是一个很好的问题。根据我的知识，Transformer 是...
但更有趣的是，如果你输入更多问题...
```

它在**续写问题**，而不是**回答问题**。

**预训练模型学习的是"续写"，不是"回答"。** 它不知道什么是"系统消息"、什么是"用户问题"、什么是"期望的回答"。SFT（监督微调）用高质量的指令-回答对教模型这些概念。

---

## 2. 概念

### 2.1 SFT 的本质

SFT = 在指令-回答对上用 next-token prediction 微调模型。关键区别：
- **预训练**：在大量通用文本上训练——学语言能力
- **SFT**：在少量高质量指令对上训练——学遵循指令

### 2.2 聊天模板格式化

SFT 的训练数据必须使用**聊天模板**——将人类可读的对话转换为模型看到的 token 序列：

```
用户消息 → 系统指令（可选）→ 用户问题 → 助手回答 → 结束
```

以 Llama 3 格式为例：

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
你是一个有帮助的助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
什么是 Transformer？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Transformer 是一种基于注意力机制的神经网络架构...
```

### 2.3 标签掩码

**关键：只在助手的回答部分计算损失。** 系统和用户部分的 token 用 -100 掩码——模型不需要"学习"如何说"用户说"。

```python
def create_sft_labels(input_ids, assistant_start_positions, pad_id=-100):
    labels = input_ids.clone()
    # 掩码非助手部分
    for i, start in enumerate(assistant_start_positions):
        labels[i, :start] = pad_id  # 系统+用户部分设为-100
    return labels
```

### 2.4 SFT 数据的质量 vs 数量

| 数据集规模 | 质量 | 效果 |
|-----------|------|------|
| 1K 高质量 | 高 | 足以让模型学会格式和基本指令 |
| 10K 高质量 | 高 | 显著提升遵循指令能力 |
| 100K 混合质量 | 中 | 可能引入噪声——模型学到坏习惯 |
| 1M 混合质量 | 低 | 过拟合训练分布，泛化差 |

**质量 >> 数量。** 1K 高质量标注通常比 100K 低质量标注效果更好。

### 2.5 常见 SFT 数据集

| 数据集 | 规模 | 来源 | 用途 |
|--------|------|------|------|
| ShareGPT | 70K | ChatGPT 对话 | 对话格式学习 |
| Alpaca | 52K | Self-Instruct | 指令遵循 |
| Dolly | 15K | Databricks 员工 | 开源对齐 |
| HH-RLHF | 170K | Anthropic | 有帮助/无害 |

---

## 3. 从零实现

### Step 1：格式化训练数据

```python
def format_sft_sample(system_prompt, user_message, assistant_response):
    """使用 Llama 3 格式格式化 SFT 样本。"""
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
    return prompt
```

### Step 2：创建训练标签（掩码非助手部分）

```python
def create_sft_labels(input_ids, system_len, user_len, assistant_len):
    """只在助手回答部分计算损失。"""
    labels = torch.full_like(input_ids, -100)
    start = system_len + user_len
    end = start + assistant_len
    labels[start:end] = input_ids[start:end]
    return labels
```

### Step 3：SFT 训练循环

```python
def train_sft(model, dataset, epochs=3, lr=2e-5):
    """SFT 训练——只在助手 token 上计算损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # 非助手位置 = -100

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss  # 自动忽略 -100 位置

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss / len(dataset):.4f}")
```

### Step 4：评估——对比 SFT 前后

```python
def evaluate_sft(model, tokenizer, prompts):
    """对比基础模型和 SFT 模型的回答质量。"""
    for prompt in prompts:
        # 基础模型：续写
        base_response = generate(base_model, tokenizer, prompt, max_tokens=100)
        # SFT 模型：回答
        sft_response = generate(sft_model, tokenizer, prompt, max_tokens=100)
        print(f"Prompt: {prompt}")
        print(f"  基础模型: {base_response[:100]}...")
        print(f"  SFT 模型: {sft_response[:100]}...")
```

---

## 4. 工具

### 4.1 Hugging Face TRL 的 SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=SFTConfig(
        output_dir="./sft-output",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_seq_length=2048,
        logging_steps=10,
    ),
)
trainer.train()
```

### 4.2 工具对比

| 工具 | 用途 | 特点 |
|------|------|------|
| TRL SFTTrainer | SFT 训练 | Hugging Face 官方，功能完整 |
| LLaMA-Factory | 多阶段微调 | 一站式工具，支持 SFT/DPO/RLHF |
| Axolotl | 灵活配置 | YAML 配置，支持多种架构 |

---

## 5. LLM 视角

### 5.1 SFT 在 LLM 训练中的位置

```
预训练（海量数据，15T tokens）
    ↓
SFT（少量高质量，10-100K 对话）
    ↓
RLHF/DPO（人类偏好优化）
    ↓
最终产品
```

SFT 是从"语言模型"到"助手"的桥梁。没有 SFT，模型只会续写；有了 SFT，模型学会了"回答"。

### 5.2 为什么 SFT 数据不需要很多

预训练已经让模型学会了"语言"。SFT 只需要教它一个新格式——"用户问→助手答"。这个格式在 1K-10K 个高质量示例中就能学会。太多数据反而有害——因为大量低质量数据会稀释格式学习信号。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你与 Claude 对话时，你看到的"你是一个有帮助的助手"就是 SFT 训练的结果。Claude 之所以能理解"用户说"和"助手答"的格式，是因为它在 SFT 阶段被训练成这样的。

---

## 6. 工程最佳实践

### 6.1 SFT 超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 5e-5 | 比预训练小 10-100 倍 |
| epochs | 1-5 | 数据量少时多跑几轮 |
| max_seq_length | 2048-4096 | 太长浪费，太短截断 |
| batch_size | 4-8 | 受限于 GPU 显存 |
| weight_decay | 0.01-0.1 | 防止过拟合 |

### 6.2 中文场景特别建议

- Alpaca 中文版（Chinese-Alpaca）有 52K 中文指令对
- 中文 SFT 数据需要确保标点符号和格式正确
- 注意中文特殊 token（如省略号、全角标点）

### 6.3 踩坑经验

- **忘记掩码非助手 token**：模型会学到"模仿用户的提问方式"——输出格式混乱
- **学习率太高**：灾难性遗忘——模型忘了预训练学到的语言能力
- **数据质量差**：SFT 数据中的错误会被模型放大——宁缺毋滥

---

## 7. 常见错误

### 错误 1：忘记掩码非助手 token

**现象：** SFT 后模型输出混乱——有时续写用户问题而非回答。

**原因：** 损失函数在整个序列上计算——模型学会了"续写用户"而非"回答用户"。

### 错误 2：学习率设置过高

**现象：** SFT 1-2 轮后 loss 急剧下降但生成质量变差——灾难性遗忘。

**修复：** 使用 1e-5 ~ 5e-5 的学习率——比预训练低 10-100 倍。

### 错误 3：SFT 数据格式不一致

**现象：** 模型在某些提示词下回答好，在其他提示词下输出垃圾。

**原因：** 训练数据混合了多种格式（不同聊天模板、不同角色定义）。模型学到了不一致的模式。

---

## 8. 面试考点

### Q1：SFT 和预训练有什么区别？为什么不能直接在预训练模型上做 RLHF？（难度：⭐⭐）

**参考答案：**
预训练在海量通用文本上用 next-token prediction 训练——学的是"续写"。SFT 在少量高质量指令对上微调——学的是"回答"。直接在预训练模型上做 RLHF 有两个问题：(1) 模型不知道"助手"应该输出什么格式——RLHF 的奖励信号太稀疏；(2) 预训练模型的动作空间太大——它可能输出任意续写而非有用回答。SFT 先给模型一个"好的初始化"——知道回答的格式和质量基线。

### Q2：SFT 的标签掩码为什么重要？（难度：⭐⭐）

**参考答案：**
如果不掩码非助手 token，模型会学到"模仿用户的输入"。它会把用户的问题当作续写目标——输出看起来像是在"继续说用户的问题"。掩码后，模型只学习"给定上下文，生成助手的回答"。技术上，`labels` 中非助手位置设为 -100，CrossEntropyLoss 会自动忽略这些位置。

### Q3：SFT 需要多少数据？为什么 1K 够了？（难度：⭐⭐⭐）

**参考答案：**
预训练已经让模型学会了"语言"——语法、语义、世界知识。SFT 只需要教一个新的"行为模式"——"用户问→助手答"。这个模式很简单：格式固定、内容多样但结构一致。1K-10K 个高质量示例足以让模型学会这个模式。太多数据反而有害：(1) 低质量数据会引入噪声；(2) 模型可能过拟合到特定训练分布；(3) 增加训练成本。Alpaca（52K）是上限——更少数据通常也足够。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| SFT | "微调" | 在指令-回答对上用 next-token prediction 微调预训练模型 |
| 聊天模板 | "对话格式" | 将角色（系统/用户/助手）转换为模型 token 序列的精确格式 |
| 标签掩码 | "只算助手的损失" | 非助手 token 的标签设为 -100——CrossEntropy 自动忽略 |
| 灾难性遗忘 | "SFT 后忘了一切" | 学习率太高导致预训练知识丢失 |
| 指令跟随 | "模型听话" | SFT 后模型能理解并执行用户的指令 |

---

## 📚 小结

SFT 是预训练模型到有用助手的桥梁——用高质量指令-回答对微调。聊天模板格式化训练数据。标签掩码确保只在助手回答部分计算损失。SFT 数据重质量轻数量——1K 高质量对通常足够。学习率比预训练低 10-100 倍，防止灾难性遗忘。SFT 是 RLHF/DPO 的前置步骤——模型必须先学会"回答"，才能进一步对齐到"好回答"。

---

## ✏️ 练习

1. **【实现】** 用 Alpaca 数据集微调一个小型 GPT（124M）。对比 SFT 前后的回答质量。
2. **【实验】** 对比不同 SFT 数据量（100, 1000, 10000）对验证 loss 的影响——画出数据量 vs 质量曲线。
3. **【思考】** 如果 SFT 数据中有一些低质量回答，模型会学到什么？如何检测和过滤低质量 SFT 数据？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| SFT 训练脚本 | `code/sft_train.py` | 聊天模板格式化 + 标签掩码 + 训练循环 |

---

## 📖 参考资料

1. [论文] Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT). NeurIPS, 2022. https://arxiv.org/abs/2203.02155
2. [论文] Taori et al. "Stanford Alpaca: An Instruction-following LLaMA Model". 2023.
3. [GitHub] HuggingFace TRL SFTTrainer: https://huggingface.co/docs/trl/main/en/sft_trainer
4. [GitHub] LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
