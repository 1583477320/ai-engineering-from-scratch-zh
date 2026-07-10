# 术语表 (Terminology)

> **规则：** 所有 Lesson 必须严格使用本表定义的翻译。不允许混用英文原文和中文翻译，也不允许同一概念使用多个翻译。

---

## 使用说明

- **第一列**为英文原文（检索用）
- **第二列**为推荐中文翻译（文档中必须使用）
- **第三列**为说明/备注
- **❌ 标记**的翻译为本项目明确禁止使用的译法

---

## 1. 基础概念 (Foundation)

| English | 中文翻译 | 说明 |
|---|---|---|
| Artificial Intelligence (AI) | 人工智能 | 不缩写为"AI"（正文中首次出现标注英文，后续直接用中文） |
| Machine Learning (ML) | 机器学习 | |
| Deep Learning (DL) | 深度学习 | |
| Natural Language Processing (NLP) | 自然语言处理 | |
| Computer Vision (CV) | 计算机视觉 | |
| Reinforcement Learning (RL) | 强化学习 | |
| Generative AI | 生成式人工智能 | 不使用 ❌ "生成式 AI" |

## 2. 模型相关 (Model)

| English | 中文翻译 | 说明 |
|---|---|---|
| Model | 模型 | |
| Large Language Model (LLM) | 大语言模型 | 不使用 ❌ "LLM"、"大模型" |
| Small Language Model (SLM) | 小语言模型 | |
| Vision Language Model (VLM) | 视觉语言模型 | |
| Multimodal Model | 多模态模型 | |
| Foundation Model | 基础模型 | 不使用 ❌ "基座模型" |
| Pre-trained Model | 预训练模型 | |
| Fine-tuned Model | 微调模型 | |
| Model Architecture | 模型架构 | |
| Model Weights | 模型权重 | |
| Parameters | 参数 | 如 7B 参数 → 70 亿参数 |
| Checkpoint | 检查点 | |
| Inference | 推理 | 注意与 Reasoning 区分 |
| Reasoning | 推理（逻辑推理） | 首次出现标注"(Reasoning，指逻辑推理)" |
| Training | 训练 | |
| Pre-training | 预训练 | |
| Fine-tuning | 微调 | |
| Instruction Tuning | 指令微调 | |
| Alignment | 对齐 | |
| RLHF | RLHF（基于人类反馈的强化学习）| 首次出现标注中文全称，后续可用 RLHF |
| DPO | DPO（直接偏好优化） | 同上 |
| Distillation | 蒸馏 | |
| Quantization | 量化 | |
| Pruning | 剪枝 | |

## 3. 神经网络 (Neural Networks)

| English | 中文翻译 | 说明 |
|---|---|---|
| Neural Network | 神经网络 | |
| Neuron | 神经元 | |
| Layer | 层 | |
| Hidden Layer | 隐藏层 | |
| Activation Function | 激活函数 | |
| ReLU | ReLU | 不翻译 |
| GELU | GELU | 不翻译 |
| Softmax | Softmax | 不翻译 |
| Sigmoid | Sigmoid | 不翻译 |
| Backpropagation | 反向传播 | |
| Gradient | 梯度 | |
| Gradient Descent | 梯度下降 | |
| Loss Function | 损失函数 | |
| Cross-Entropy Loss | 交叉熵损失 | |
| Optimizer | 优化器 | |
| Adam / AdamW | Adam / AdamW | 不翻译 |
| SGD | SGD（随机梯度下降）| |
| Learning Rate | 学习率 | |
| Batch | 批次 | |
| Epoch | 轮次 | 不使用 ❌ "epoch"、"周期" |
| Overfitting | 过拟合 | |
| Underfitting | 欠拟合 | |
| Regularization | 正则化 | |
| Dropout | Dropout | 不翻译 |
| Normalization | 归一化 | |
| Layer Normalization | 层归一化 | |
| Batch Normalization | 批归一化 | |
| Residual Connection | 残差连接 | |

## 4. Transformer 架构

| English | 中文翻译 | 说明 |
|---|---|---|
| Transformer | Transformer | 不翻译 |
| Attention | 注意力 | |
| Self-Attention | 自注意力 | |
| Multi-Head Attention | 多头注意力 | |
| Cross-Attention | 交叉注意力 | |
| Scaled Dot-Product Attention | 缩放点积注意力 | |
| Query (Q) | 查询 (Q) | |
| Key (K) | 键 (K) | |
| Value (V) | 值 (V) | |
| Positional Encoding | 位置编码 | |
| Rotary Position Embedding (RoPE) | RoPE（旋转位置嵌入）| |
| Feed-Forward Network (FFN) | 前馈网络 | |
| Encoder | 编码器 | |
| Decoder | 解码器 | |
| Encoder-Decoder | 编码器-解码器 | |
| Autoregressive | 自回归 | |
| Causal Mask | 因果掩码 | |
| KV Cache | KV 缓存 | |
| Context Window | 上下文窗口 | |
| Context Length | 上下文长度 | |

## 5. 词元与嵌入 (Tokens & Embeddings)

| English | 中文翻译 | 说明 |
|---|---|---|
| Token | 词元 | **最重要！** 不使用 ❌ "Token"、"标记" |
| Tokenizer | 分词器 | 不使用 ❌ "Token 化器" |
| Tokenization | 分词 | 不使用 ❌ "Token 化" |
| BPE (Byte Pair Encoding) | BPE（字节对编码）| |
| SentencePiece | SentencePiece | 不翻译 |
| Vocabulary | 词表 | |
| Embedding | 嵌入 | |
| Word Embedding | 词嵌入 | |
| Token Embedding | 词元嵌入 | |
| Positional Embedding | 位置嵌入 | |
| Embedding Dimension | 嵌入维度 | |
| Vector | 向量 | |
| Hidden State | 隐藏状态 | |

## 6. 提示与生成 (Prompt & Generation)

| English | 中文翻译 | 说明 |
|---|---|---|
| Prompt | 提示词 | 不使用 ❌ "Prompt"、"提示" |
| Prompt Engineering | 提示词工程 | |
| System Prompt | 系统提示词 | |
| User Prompt | 用户提示词 | |
| Few-shot Prompting | 少样本提示 | |
| Zero-shot Prompting | 零样本提示 | |
| Chain-of-Thought (CoT) | 思维链 (CoT) | |
| Generation | 生成 | |
| Decoding | 解码 | |
| Greedy Decoding | 贪心解码 | |
| Beam Search | 束搜索 | |
| Temperature | 温度 | |
| Top-k | Top-k | 不翻译 |
| Top-p | Top-p | 不翻译 |
| Sampling | 采样 | |
| Hallucination | 幻觉 | |

## 7. AI 工程 (AI Engineering)

| English | 中文翻译 | 说明 |
|---|---|---|
| Agent | 智能体 | 不使用 ❌ "Agent"、"代理" |
| Multi-Agent | 多智能体 | |
| Tool Calling | 工具调用 | |
| Function Calling | 函数调用 | |
| RAG (Retrieval-Augmented Generation) | RAG（检索增强生成）| |
| Vector Database | 向量数据库 | |
| Embedding Model | 嵌入模型 | |
| Reranker | 重排序器 | |
| Chunking | 分块 | |
| LangChain | LangChain | 不翻译 |
| LangGraph | LangGraph | 不翻译 |
| LlamaIndex | LlamaIndex | 不翻译 |
| Orchestration | 编排 | |
| Pipeline | 流水线 | |
| Latency | 延迟 | |
| Throughput | 吞吐量 | |
| Deployment | 部署 | |
| Serving | 在线服务 | |
| vLLM | vLLM | 不翻译 |
| TensorRT-LLM | TensorRT-LLM | 不翻译 |
| Ollama | Ollama | 不翻译 |

## 8. 评估 (Evaluation)

| English | 中文翻译 | 说明 |
|---|---|---|
| Evaluation / Eval | 评估 | |
| Benchmark | 基准测试 | |
| Metric | 指标 | |
| Accuracy | 准确率 | |
| Precision | 精确率 | |
| Recall | 召回率 | |
| F1 Score | F1 分数 | |
| Perplexity | 困惑度 | |
| BLEU | BLEU | 不翻译 |
| ROUGE | ROUGE | 不翻译 |
| Human Evaluation | 人工评估 | |
| Ground Truth | 真实答案 | |
| MMLU | MMLU | 不翻译 |
| HumanEval | HumanEval | 不翻译 |

## 9. 数据 (Data)

| English | 中文翻译 | 说明 |
|---|---|---|
| Dataset | 数据集 | |
| Training Set | 训练集 | |
| Validation Set | 验证集 | |
| Test Set | 测试集 | |
| Sample | 样本 | |
| Label | 标签 | |
| Annotation | 标注 | |
| Preprocessing | 预处理 | |
| Data Augmentation | 数据增强 | |
| Corpus | 语料库 | |
| Data Leakage | 数据泄露 | |

## 10. 硬件与性能 (Hardware & Performance)

| English | 中文翻译 | 说明 |
|---|---|---|
| GPU | GPU | 不翻译 |
| CPU | CPU | 不翻译 |
| TPU | TPU | 不翻译 |
| NPU | NPU | 不翻译 |
| VRAM | 显存 | |
| FLOPs | FLOPs（浮点运算次数）| |
| Memory Bandwidth | 内存带宽 | |
| Batch Size | 批次大小 | |
| Mixed Precision | 混合精度 | |
| FP16 / FP32 / BF16 | FP16 / FP32 / BF16 | 不翻译 |
| Distributed Training | 分布式训练 | |
| Data Parallelism | 数据并行 | |
| Model Parallelism | 模型并行 | |
| Pipeline Parallelism | 流水线并行 | |
| Tensor Parallelism | 张量并行 | |

---

## 原则总结

1. **函数名、库名、框架名不翻译** — `Softmax`、`Dropout`、`LangChain`、`vLLM` 保留原文
2. **缩写首次出现标注全称** — 如 "RLHF（基于人类反馈的强化学习）"，后续直接用 RLHF
3. **优先中文** — 有自然中文术语的，用中文；英文只有在约定俗成或中文无法简洁表达时才保留
4. **一致性高于一切** — 即使你个人偏好某个翻译，也必须使用本表定义

---

## 常见错误速查

| ❌ 错误用法 | ✓ 正确用法 |
|---|---|
| Token | 词元 |
| 大模型 / LLM | 大语言模型 |
| Prompt | 提示词 |
| Agent / 代理 | 智能体 |
| 基座模型 | 基础模型 |
| epoch | 轮次 |
| 推理 (指 Inference) | 推理 / 推理（逻辑推理）— 注意区分 |
| Fine-tune | 微调 |
| Embedding (用作名词) | 嵌入 |
| Attention | 注意力 |
| 大模型时代 | 大语言模型时代 |
