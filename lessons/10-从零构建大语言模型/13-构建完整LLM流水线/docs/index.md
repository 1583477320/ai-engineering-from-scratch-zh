# 构建完整LLM流水线

> 从预处理到部署——一个完整的LLM企业级落地需要数据管道、训练、评估、量化和推理优化的无缝集成。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段10·01-12
**时间：** ~120 分钟

---

## 🎯 学习目标

- [ ] 设计端到端LLM流水线——数据→训练→评估→量化→部署
- [ ] 选择开源LLM生态——HuggingFace transformers + vLLM + DeepSpeed
- [ ] 用LoRA在领域数据上微调基础模型

---

## 📚 小结

完整LLM流水线=数据管道→预训练→SFT→RLHF/DPO→评估→量化→部署。2026年的最佳实践：用LLaMA 3或Qwen2.5作为基础模型，用LoRA在领域数据上微调，用量化（GPTQ/AWQ）压缩模型，用vLLM部署。

---

## 📖 参考资料

1. [博客] 'What We Learned from a Year of Building LLMs' — 2026生产经验