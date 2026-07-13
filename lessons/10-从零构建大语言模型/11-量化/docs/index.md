# 量化——压缩 LLM

> 量化将模型权重从 FP16（16位）压缩到 INT4（4位）。参数量不变，但内存占用降到原来的1/4——在消费级GPU上运行7B模型。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 10 · 04（预训练）
**时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 理解量化原理——从FP16到INT4的精度折衷
- [ ] 区分权重量化和激活量化——以及各自的优缺点
- [ ] 说明GGML/GPTQ/AWQ三种量化方案的差异

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| FP16量化 | 从FP16到INT8/INT4——精度损失但内存4x节省 |
| GPTQ | 基于Hessian矩阵的权重量化——逐层校准 |
| GGML | llama.cpp的量化方案——CPU友好 |
| AWQ | 基于激活值的感知量化——质量损失最小 |

---

## 📚 小结

量化=从FP16到INT4/INT8——内存降到1/2到1/4。GPTQ质量最高但需要校准数据。GGML CPU友好。AWQ在质量/速度上最佳平衡。7B FP16需要14GB VRAM；7B INT4只需要4GB——可在消费级GPU上推理。

---

## 📖 参考资料

1. [论文] Frantar et al. 'GPTQ: Accurate Post-Training Quantization'. 2022.
2. [论文] Lin et al. 'AWQ: Activation-aware Weight Quantization'. 2023.