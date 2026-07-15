# LLM 评估

> 你怎么知道你的模型变好了？困惑度不够。你需要人类偏好的代理指标——AlpacaEval、MT-Bench、Chatbot Arena——以及最终的裁判：人类评估。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、06（SFT）| **时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 04（预训练）— 困惑度评估 | 阶段 10 · 06（SFT）— 指令跟随评估

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分内在评估和外在评估——困惑度 vs 人类偏好
- [ ] 解释 AlpacaEval 和 MT-Bench 的工作原理——LLM-as-Judge
- [ ] 说明 Chatbot Arena 的 ELO 评分——基于用户偏好的 LLM 排名

---

## 1. 问题

训练完了——怎么知道模型变好了？困惑度衡量"预测下一个词的能力"，但用户关心的是"你更想用哪个模型"。FID 和 CLIP Score 用于图像生成评估——LLM 也需要类似的量化指标。

三个主流基准：
- **AlpacaEval**：单轮问答，LLM-as-Judge 评分
- **MT-Bench**：多轮对话，GPT-4 作为评判者
- **Chatbot Arena**：ELO 排名的"LLM 世界杯"——最佳的主观质量评估

---

## 2. 概念

### 2.1 内在评估 vs 外在评估

| 类型 | 指标 | 衡量什么 | 局限 |
|------|------|---------|------|
| 内在评估 | 困惑度 (PPL) | 模型对文本的"惊讶程度" | 不等于"有用"或"好" |
| 外在评估 | 人类偏好、LLM-as-Judge | 模型输出的质量 | 成本高、有偏差 |

**困惑度越低 = 模型预测下一个词越准确 = 生成越连贯。** 但"连贯"≠"有帮助"——一个模型可以非常连贯地胡说八道。

### 2.2 AlpacaEval

AlpacaEval 2.0：从 805 个指令-回答对中，让 GPT-4 Turbo 比较模型输出与参考输出。胜率越高越好。

| AlpacaEval 2.0 胜率 | 评估 |
|---------------------|------|
| > 50% | 优于 GPT-4 基线 |
| 60-70% | 强模型 |
| > 70% | 顶级模型 |

### 2.3 MT-Bench

MT-Bench 使用 80 个多轮对话问题，覆盖 8 个类别（写作、推理、编码、数学、STEM、人文、角色扮演、其他）。GPT-4 对每轮回答打分（1-10）。

### 2.4 Chatbot Arena

Chatbot Arena 是"LLM 世界杯"——用户与两个匿名模型对话，选择更好的那个。使用 ELO 评分系统——类似国际象棋排名。

2026 年的 Chatbot Arena 排名（示例）：
| 排名 | 模型 | ELO |
|------|------|-----|
| 1 | GPT-4o | 1290 |
| 2 | Claude 3.5 Sonnet | 1275 |
| 3 | Gemini 1.5 Pro | 1260 |
| ... | ... | ... |

### 2.5 LLM-as-Judge 的偏差

| 偏差 | 描述 | 修复 |
|------|------|------|
| 位置偏差 | 倾向选择第一个/最后一个回答 | 随机化顺序 |
| 长度偏差 | 倾向选择更长的回答 | 指令中明确"长度不影响评分" |
| 自我偏好 | GPT-4 偏好 GPT-4 的回答 | 使用多个不同模型做评判 |

---

## 3. 工具

### 3.1 AlpacaEval

```bash
pip install alpaca-eval
alpaca-eval evaluate --model_name my-model
```

### 3.2 MT-Bench

```bash
pip install fastchat
python -m fastchat.llm_judge.chatgpt_gpt4_handler
```

### 3.3 Chatbot Arena

```python
# 在本地运行
# https://github.com/lm-sys/FastChat
python -m fastchat.serve.cli --model-path my-model
```

---

## 4. LLM 视角

### 4.1 为什么"困惑度"不够

GPT-3 的困惑度在某些基准上很好，但它的回答可能无帮助、不安全或充满幻觉。困惑度只能衡量"模型学到的语言模式有多准确"，不能衡量"模型的行为是否符合人类偏好"。

### 4.2 LLM-as-Judge 的趋势

2026 年，LLM-as-Judge 已经成为主流评估方法——比人类标注更快、更便宜、更一致。但需要多个评判者做交叉验证以减少偏差。

---

## 5. 工程最佳实践

### 5.1 评估组合

不要只看一个指标——至少使用 2-3 个互补的指标：

| 场景 | 推荐指标 |
|------|---------|
| 预训练监控 | 困惑度 |
| SFT 效果 | AlpacaEval + MT-Bench |
| 对齐效果 | Chatbot Arena + 人类评估 |
| 代码能力 | HumanEval + MBPP |
| 数学能力 | GSM8K + MATH |

---

## 6. 常见错误

### 错误 1：只看困惑度

**现象：** 困惑度低但模型回答无帮助。

**修复：** 同时使用 AlpacaEval + MT-Bench 做外在评估。

### 错误 2：LLM-as-Judge 的位置偏差

**现象：** GPT-4 稳定地选择第一个回答。

**修复：** 每个比较运行两次，随机化回答顺序。

---

## 7. 面试考点

### Q1：为什么困惑度不是衡量 LLM 质量的充分指标？（难度：⭐⭐）

**参考答案：**
困惑度衡量的是"模型预测下一个词有多准确"——但"准确"不等于"有帮助"。一个模型可以非常连贯地生成错误信息（低困惑度，高质量文本但事实错误）。还需要外在指标：AlpacaEval（指令遵循）、MT-Bench（多轮对话）、人类评估（偏好）。

### Q2：Chatbot Arena 的 ELO 评分是如何计算的？（难度：⭐⭐⭐）

**参考答案：**
ELO 评分系统来源于国际象棋排名。每个模型有一个分数（初始 1000）。当两个模型"对战"（用户选择更好的那个）时，分数根据预期胜负和实际胜负更新。预期胜负由当前分数差决定——分数高的模型预期赢的概率更大。如果"弱"模型赢了，它获得更多分数。这确保了排名的公平性。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 困惑度 (PPL) | "模型多准确" | exp(交叉熵损失)——衡量模型对文本的预测能力 |
| AlpacaEval | "LLM 自己打分" | GPT-4 比较模型输出与基线输出的胜率 |
| MT-Bench | "多轮对话测试" | 80 个多轮对话，GPT-4 打分（1-10） |
| Chatbot Arena | "LLM 世界杯" | 用户匿名选择更好模型，ELO 排名 |
| LLM-as-Judge | "用 AI 评估 AI" | 用大模型评估小模型输出——快速但有偏差 |

---

## 📚 小结

困惑度是预训练的基本指标，但不等于"有用"。AlpacaEval（单轮）、MT-Bench（多轮）、Chatbot Arena（用户偏好）是 2026 年的三大外在评估基准。LLM-as-Judge 是主流但需要校准——位置偏差和长度偏差是主要问题。最佳实践：使用 2-3 个互补指标。

---

## ✏️ 练习

1. **【实验】** 用你的 MiniGPT 计算 AlpacaEval 胜率——对比 SFT 前后的得分变化。
2. **【实验】** 设计一个简单的 LLM-as-Judge——用 GPT-4 评估 10 对回答，对比与人类评估的一致性。

---

## 📖 参考资料

1. [GitHub] AlpacaEval: https://github.com/tatsu-lab/alpaca_eval
2. [GitHub] MT-Bench: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
3. [GitHub] Chatbot Arena: https://lmsys.org/blog/2023-10-30-arena/
4. [论文] Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". NeurIPS, 2023. https://arxiv.org/abs/2306.05685

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
