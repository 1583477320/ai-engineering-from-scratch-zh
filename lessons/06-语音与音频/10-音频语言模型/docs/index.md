# 音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 音频语言模型同时推理语音、环境声和音乐。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹配 GPT-4o Audio。开放与封闭之间的差距基本消失——除多音频任务外，所有模型都接近随机。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 06 · 04（ASR）、阶段 12 · 03（视觉语言模型）、阶段 7 · 10（音频 Transformer）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别音频语言模型的三组件架构——音频编码器、投影层、LLM 解码器
- [ ] 区分 Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio 的技术差异和开放状态
- [ ] 理解音频 LLM 的训练三阶段——投影层预训练 → 指令微调 → 语音输入输出

---

## 1. 问题

5 秒音频：狗叫、有人喊"停下！"，然后沉默。有用的问题跨越多个轴：

- **转录：** "说了什么？"——ASR 的领域
- **语义推理：** "这个人有危险吗？"——需要联合理解狗叫+喊叫+沉默
- **音乐推理：** "哪些乐器演奏旋律？"
- **长音频检索：** "教授在 90 分钟的演讲中哪个位置解释了梯度下降？"

单个模型用一个提示回答所有这些问题——这就是**音频语言模型（LALM/ALM）**。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，不只是转录。

---

## 2. 概念

### 2.1 三组件架构模板

每个 2026 年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper encoder / BEATs / CLAP / WavLM / 各模型自定义编码器
2. **投影层。** 线性或 MLP，将音频编码器特征映射到 LLM 的 token 嵌入空间
3. **LLM。** Llama / Qwen / Gemma 为基础的解码器。接受交织的文本 + 音频 token；生成文本

训练分为三阶段：

- **阶段 1：** 冻结编码器 + LLM；仅在 ASR/字幕数据上训练投影层
- **阶段 2：** 完整 / LoRA 微调，指令跟随音频任务（QA、推理、音乐理解）
- **阶段 3（可选）：** 语音输入 + 语音输出——添加语音解码器。Qwen2.5-Omni 和 AF3-Chat 做了这一步

### 2.2 2026 模型图谱

| 模型 | 骨干 | 音频编码器 | 输出模态 | 许可 |
|---|---|---|---|---|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商用 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| GPT-4o (closed) | GPT-4o | 专有 | 文本 + 语音 | API |

### 2.3 中文音频语言模型

Qwen2.5-Omni 是 2026 年中文音频理解的最佳开源选择。它支持中文语音输入+中文文本输出，训练数据中中文占比充足。对于中文法律/医疗等垂直领域的语音理解，fine-tune Qwen2.5-Omni 是当前的最低成本方案。

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| LALM | 音频语言模型——用 LLM 推理语音、环境声、音乐，输出自由形式文本 |
| 投影层 | 线性/MLP 将音频编码器特征映射到 LLM token 空间的桥接模块 |
| MMAU-Pro | 多模态音频理解基准——衡量音频 QA、事件检测等任务 |

---

## ✏️ 练习

1. 【理解】画出 Qwen2.5-Omni 的三组件架构图（音频编码器→投影层→LLM）。
2. 【实验】用 `transformers` 加载 Whisper encoder，取一个音频样本，打印中间层激活的形状。

---

## 📖 参考资料

1. [论文] Yang et al. "Qwen2.5-Omni Technical Report". 2025.
2. [论文] Chu et al. "Audio Flamingo 3". 2025.
3. [基准] MMAU-Pro. https://github.com/google-research/google-research/tree/master/mmau

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
