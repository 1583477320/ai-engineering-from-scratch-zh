# 开源模型架构分析

> LLaMA 3、Qwen2.5、Mistral、Phi-3——2026年的开源LLM各有巧妙的设计选择。分析它们的架构，理解'为什么这么设计'。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段10·01-12
**时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 说明LLaMA 3的架构选择（GQA + GLU + RoPE）——为什么每个选择对推理速度有利
- [ ] 解释Qwen2.5的GQA配置——8KV头 vs 8/16/32查询头
- [ ] 理解Mistral的滑动窗口注意力 + MoE设计

---

## 📚 小结

2026年开源LLM的统一架构：Pre-LN + GQA + SwiGLU + RoPE + RMSNorm。差异在注意力变体（Mistral的滑动窗口、DeepSeek-V3的MoE）和训练数据（LLaMA 3的15T token vs Qwen2.5的18T token）。

---

## 📖 参考资料

1. [模型卡] LLaMA 3. https://ai.meta.com/blog/meta-llama-3/
2. [模型卡] Qwen2.5. https://huggingface.co/Qwen/Qwen2.5-7B
3. [论文] Jiang et al. 'Mistral 7B'. 2023.