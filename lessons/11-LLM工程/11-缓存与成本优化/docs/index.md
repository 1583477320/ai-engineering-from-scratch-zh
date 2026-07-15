# 缓存与成本优化

> LLM API 按 token 收费。Prompt Cache 和缓存策略能让你节省 50-90% 的成本。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 12（推理优化）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 Prompt Caching 原理——减少重复预填充的 API 优化
- [ ] 实现响应缓存——基于语义相似度
- [ ] 分析 LLM 应用的成本结构并设计优化方案
- [ ] 理解 Batch API 和流式输出如何降低成本

---

## 1. 问题

你的 LLM 应用每天处理 10 万次 API 调用。大部分调用的系统消息完全相同——但每次调用都重新处理。Prompt Caching 可以为这部分系统消息的预填充计算节省 90% 的成本。

---

## 2. 概念

### 2.1 LLM 成本构成

| 部分 | 成本 | 优化方向 |
|------|------|---------|
| 输入 token | ~1/3 价格 | Prompt Cache, 压缩上下文 |
| 输出 token | 全价 | 减少不必要输出 |
| 预填充 | 高（计算密集） | Prompt Cache |

### 2.2 Prompt Caching

两次调用共享相同的前缀（如系统消息），第二次调用可重用预填充结果：

```
调用 1: [系统消息(2000 token)] + [查询1] → 预填充2000 + 生成100
调用 2: [系统消息(2000 token)] + [查询2] → 预填充0（缓存命中） + 生成100
```

系统消息长 2000 token 时，第二次调用节省 90% 输入成本。

### 2.3 成本优化策略

| 策略 | 节省 | 实现难度 |
|------|------|---------|
| Prompt Cache | 50-90% 输入成本 | 低 |
| 响应缓存 | 30-70% 总成本 | 中 |
| Batch API | 50% 输出成本 | 低 |
| 流式输出 | 早期停止节省 | 中 |
| 小模型路由 | 70-90% 总成本 | 高 |

---

## 3. 从零实现

### Step 1：响应缓存

```python
import hashlib
import json

class ResponseCache:
    def __init__(self):
        self.cache = {}

    def _hash(self, prompt, model):
        return hashlib.md5((prompt + model).encode()).hexdigest()

    def get(self, prompt, model):
        return self.cache.get(self._hash(prompt, model))

    def set(self, prompt, model, response):
        self.cache[self._hash(prompt, model)] = response
```

### Step 2：成本计算器

```python
def calculate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """计算 API 调用成本。"""
    prices = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6),
              "claude-sonnet": (3.0, 15.0)}
    in_price, out_price = prices.get(model, (2.5, 10.0))
    cost = (input_tokens / 1_000_000 * in_price) + (output_tokens / 1_000_000 * out_price)
    return cost

def estimate_monthly_cost(daily_calls, avg_input=500, avg_output=200, cache_hit_rate=0.0):
    """估算月度成本。"""
    monthly_calls = daily_calls * 30
    effective_input = avg_input * (1 - cache_hit_rate) + avg_input * 0.1 * cache_hit_rate
    total_cost = monthly_calls * calculate_cost(effective_input, avg_output)
    return total_cost
```

---

## 4. 工具

### 4.1 OpenAI Prompt Caching

OpenAI API 自动支持 Prompt Caching——无需代码修改，超过 1024 token 的系统消息前缀自动缓存，成本降低 50%。

### 4.2 Batch API

OpenAI Batch API 以 50% 折扣批量处理——适合不需要实时响应的任务。

---

## 6. 工程最佳实践

### 6.1 中文场景

- 中文响应的 token 数通常比英文少 30-50%——成本更低
- 中文响应可以更简洁——用提示词控制长度

### 6.2 踩坑经验

- **缓存失效**：用户对话历史变化导致系统消息前缀变化 → 重新缓存
- **过度缓存**：缓存太旧的数据导致回复过时 → 设置合理的 TTL

---

## 7. 面试考点

### Q1：Prompt Caching 如何工作？（难度：⭐⭐）

**参考答案：**
Prompt Caching 重用共享前缀的预填充结果。如果两次 API 调用的系统消息完全相同，第二次调用可以跳过这部分 token 的处理，成本降低 50-90%。实现原理：API 在第一次调用后缓存 KV 缓存。第二次调用检测到相同前缀后，直接加载缓存的 KV 而不重新计算。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Prompt Caching | "缓存提示词" | API 层面的缓存——重用共享前缀的预填充计算 |
| Batch API | "批量处理" | API 以 50% 折扣批量处理请求——适合非实时任务 |
| 流式输出 | "边生成边返回" | 逐 token 返回——支持早期停止和实时展示 |

---

## 📚 小结

LLM 成本优化的核心：Prompt Cache 节省输入成本、响应缓存避免重复生成、Batch API 节省输出成本、小模型路由降低总成本。OpenAI 的 Prompt Caching 是零配置的生产优化。

---

## ✏️ 练习

1. **【实验】** 模拟 10 万次日调用的成本——对比有/无 Prompt Cache 的月费差异
2. **【实现】** 实现基于嵌入相似度的响应缓存系统

---

## 📖 参考资料

1. [文档] OpenAI Prompt Caching: https://platform.openai.com/docs/guides/prompt-caching
2. [文档] OpenAI Batch API: https://platform.openai.com/docs/guides/batch-processing
3. [博客] LLM Cost Optimization: https://www.anthropic.com/research/prompt-caching
