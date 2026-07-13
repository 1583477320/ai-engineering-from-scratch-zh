# 推理优化

> LLM推理是自回归的——每生成一个token需要一次前向传播。KV缓存+FlashAttention+推测解码是加速的三大支柱。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 10 · 04（预训练）、阶段 07 · 12（KV缓存）
**时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 理解推理延迟的三个来源——预填充（prefill）+ KV生成 + 解码（decoding）
- [ ] 解释vLLM的PagedAttention——如何解决KV缓存碎片化
- [ ] 说明推测解码如何用小模型加速大模型

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 预填充 | 处理输入token，构建初始KV缓存——计算密集但并行 |
| PagedAttention | 将KV缓存分页管理——类似操作系统的虚拟内存——消除碎片化 |
| 连续批处理 | 在推理过程中动态添加/移除请求——最大化GPU利用率 |
| vLLM | 高吞吐LLM推理引擎——PagedAttention + 连续批处理 |

---

## 📚 小结

推理优化=KV缓存(减少重复计算)+FlashAttention(降低内存)+PagedAttention(消除碎片)+连续批处理(满GPU利用率)。vLLM/TRT-LLM在2026年是生产推理的选择。推测解码用草稿模型+主模型并行——额外2-3x加速。

---

## 📖 参考资料

1. [论文] Kwon et al. 'Efficient Memory Management for Large Language Model Serving with PagedAttention'. SOSP, 2023.
2. [项目] vLLM. https://github.com/vllm-project/vllm