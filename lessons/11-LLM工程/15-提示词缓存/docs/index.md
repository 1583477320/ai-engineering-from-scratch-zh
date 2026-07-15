# 提示词缓存

> Prompt Caching 是 Anthropic/OpenAI 在 2024-2025 年推出的关键优化——自动缓存长系统消息的预填充 KV 缓存，成本降低 90%，延迟降低 85%。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 11 · 11（缓存与成本优化）| **时间：** ~30 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 Prompt Caching 的工作原理——KV 缓存前缀重用
- [ ] 设计高效的缓存友好的提示词结构
- [ ] 分析缓存命中率对延迟和成本的影响

---

## 1. 问题

你的 LLM 应用的系统消息有 5000 token——包含角色定义、工具描述、安全规则。每次 API 调用都重新处理这 5000 token 的预填充。Prompt Caching 让你只付费一次，后续调用重用缓存。

---

## 2. 概念

### 2.1 工作原理

```
调用 1: [系统消息 5000 token] + [查询1 100 token]
         预填充 5000（计算密集）+ 生成 100
         → 缓存 KV[5000 token]

调用 2: [系统消息 5000 token] + [查询2 100 token]
         预填充 0（缓存命中）+ 生成 100
         → 成本: 90% 折扣
```

### 2.2 支持平台

| 平台 | Prompt Caching | 折扣 | 最低前缀 |
|------|---------------|------|---------|
| Anthropic | 自动 | 90% | 1024 token |
| OpenAI | 自动 | 50% | 1024 token |
| Google | 上下文缓存 | 75% | 32K token |

### 2.3 缓存友好的提示词设计

```
[系统消息: 2000 token] ← 长且稳定（角色、规则、工具）
   ↓
[对话历史: 动态]
   ↓
[当前查询: 短]
```

将重要信息放系统消息——它会自动缓存。

---

## 3. 从零实现

```python
def estimate_cache_savings(calls_per_day, system_tokens, avg_query_tokens):
    """估算 Prompt Cache 节省。"""
    # 无缓存
    total_tokens_no_cache = calls_per_day * (system_tokens + avg_query_tokens)
    # 有缓存（前缀只计一次）
    total_tokens_with_cache = system_tokens + calls_per_day * avg_query_tokens
    savings = 1 - total_tokens_with_cache / total_tokens_no_cache
    return savings
```

---

## 6. 工程最佳实践

- **系统消息放重要规则**：系统消息自动缓存——稳定的长前缀效果最好
- **避免频繁修改系统消息**：前缀变化会导致缓存失效
- **长对话中缓存价值递减**：对话历史增长后前缀占比减小

---

## 7. 面试考点

### Q1：Prompt Caching 为什么能加速？（难度：⭐⭐）

**参考答案：**
LLM 推理分两步：(1) 预填充——处理所有输入 token 的 KV 缓存（O(n²)计算）；(2) 解码——逐 token 生成（O(n)每步）。Prompt Caching 缓存了预填充的 KV 结果。如果系统消息 5000 token，第二次调用可以跳过这 5000 token 的 O(n²) 计算——直接从缓存的 KV 开始，只做剩余 token 的计算。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Prompt Caching | "提示词缓存" | API 层面缓存共享前缀的 KV 缓存——自动降低延迟和成本 |

---

## 📚 小结

Prompt Caching 自动缓存长前缀的 KV 缓存——Anthropic 折扣 90%，OpenAI 折扣 50%。设计上将稳定的长内容放系统消息可以最大化缓存效益。

---

## ✏️ 练习

1. **【实验】** 计算系统消息 5000 token、日调用 10 万次时的年成本节省
2. **【思考】** Prompt Caching 对 RAG 应用有什么影响？检索内容变化时缓存会怎样？

---

## 📖 参考资料

1. [文档] Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
2. [文档] OpenAI Prompt Caching: https://platform.openai.com/docs/guides/prompt-caching
