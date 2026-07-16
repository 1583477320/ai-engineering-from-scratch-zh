# LLM 路由层——LiteLLM、OpenRouter、Portkey

> 供应商锁定代价高昂。不同的工具调用工作负载适合不同的模型。路由网关提供一个 API 表面、重试、故障转移、成本追踪和护栏。2026 年三种架构主导：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用）、17（网关）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分自托管、托管和生产级路由选项
- [ ] 设计 LLM 路由层——故障转移、成本追踪、护栏
- [ ] 对比 LiteLLM、OpenRouter、Portkey 的优劣
- [ ] 实现简单的路由网关

---

## 1. 问题

不同的工具调用工作负载适合不同的模型——简单查询用小模型（便宜、快），复杂推理用大模型（贵、慢）。但为每个请求选择最佳模型很复杂。

LLM 路由层解决这个问题——统一 API、智能路由、故障转移、成本控制。

---

## 2. 概念

### 2.1 路由决策因素

| 因素 | 简单查询 | 复杂推理 |
|------|---------|---------|
| 模型选择 | 小模型（GPT-4o-mini） | 大模型（GPT-4o） |
| 温度 | 低（0-0.3） | 中（0.5-0.7） |
| 最大 token | 少（256-512） | 多（2048+） |
| 成本敏感度 | 高 | 中 |

### 2.2 三种路由架构

| 架构 | 代表 | 特点 |
|------|------|------|
| 自托管 | LiteLLM | 开源、完全可控 |
| 托管 SaaS | OpenRouter | 免维护、多模型 |
| 生产级 | Portkey | 企业功能、监控 |

### 2.3 路由网关功能

- **智能路由**：根据请求复杂度选择模型
- **故障转移**：主模型失败自动切换到备用
- **成本追踪**：实时监控每个模型的成本
- **护栏**：输入/输出过滤、内容审核
- **缓存**：相似查询缓存结果

---

## 3. 从零实现

### Step 1：简单路由网关

```python
class LLMRouter:
    """简化版 LLM 路由网关。"""
    def __init__(self):
        self.models = {}
        self.fallback_model = None
        self.cost_tracker = {}

    def register(self, name, model_fn, cost_per_1k_tokens=0.01):
        self.models[name] = {"fn": model_fn, "cost": cost_per_1k_tokens}

    def set_fallback(self, name):
        self.fallback_model = name

    def route(self, prompt, strategy="simple"):
        """路由请求到最佳模型。"""
        if strategy == "simple":
            model_name = list(self.models.keys())[0]
        elif strategy == "cost":
            model_name = min(self.models, key=lambda n: self.models[n]["cost"])
        elif strategy == "quality":
            model_name = max(self.models, key=lambda n: self.models[n]["cost"])

        try:
            result = self.models[model_name]["fn"](prompt)
            return {"response": result, "model": model_name, "cost": self.models[model_name]["cost"]}
        except Exception:
            if self.fallback_model:
                result = self.models[self.fallback_model]["fn"](prompt)
                return {"response": result, "model": self.fallback_model, "fallback": True}
            return {"error": "所有模型不可用"}

    def get_cost_summary(self):
        return self.cost_tracker
```

### Step 2：路由策略

```python
def classify_query_complexity(prompt):
    """根据查询复杂度选择模型。"""
    if len(prompt) < 100 and any(k in prompt for k in ["你好", "天气", "时间"]):
        return "simple"  # 小模型
    elif len(prompt) > 500 or any(k in prompt for k in ["分析", "推理", "证明"]):
        return "complex"  # 大模型
    else:
        return "medium"  # 中等模型
```

---

## 4. 工具

### 4.1 LiteLLM

```python
from litellm import completion

# 统一 API——自动路由到最优模型
response = completion(
    model="gpt-4o",  # 或 claude-sonnet-5, gemini-1.5-pro
    messages=[{"role": "user", "content": "你好"}],
)
```

### 4.2 OpenRouter

```python
import requests
response = requests.post("https://openrouter.ai/api/v1/chat/completions", json={
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "你好"}],
})
```

### 4.3 Portkey

```python
from portkey_ai import Portkey

client = Portkey(api_key="your-key", provider="openai")
response = client.completions.create(model="gpt-4o", prompt="你好")
```

### 4.4 工具对比

| 工具 | 类型 | 特点 |
|------|------|------|
| LiteLLM | 开源自托管 | 完全可控，支持 100+ 模型 |
| OpenRouter | 托管 SaaS | 免维护，多模型市场 |
| Portkey | 生产级 | 企业功能，监控、成本追踪 |
| Martian | 智能路由 | 自动选择最优模型 |

---

## 5. 工程最佳实践

### 5.1 路由策略

| 场景 | 推荐 | 原因 |
|------|------|------|
| 成本敏感 | 智能路由 | 简单用小模型，复杂用大模型 |
| 延迟敏感 | 质量路由 | 始终用最快模型 |
| 故障容忍 | 故障转移 | 主模型不可用时自动切换 |

### 5.2 踩坑经验

- **路由开销**：路由决策本身有延迟——简单查询不要过度路由
- **多供应商 API 差异**：不同供应商的函数调用格式不同——需要适配层
- **成本不透明**：不同供应商的定价复杂——需要详细的成本追踪

---

## 6. 常见错误

### 错误 1：所有请求用同一个模型

**现象：** 简单查询和复杂推理用同一个模型——成本浪费或质量不足。

**修复：** 实现智能路由——根据查询复杂度选择模型。

### 错误 2：没有故障转移

**现象：** 主模型不可用时用户看到错误。

**修复：** 实现自动故障转移——主模型失败时切换到备用。

---

## 7. 面试考点

### Q1：LLM 路由层解决了什么问题？（难度：⭐⭐）

**参考答案：**
(1) **供应商锁定**：切换提供商时不需要重写所有代码；(2) **成本优化**：智能路由——简单查询用便宜模型，复杂推理用大模型；(3) **故障容忍**：自动故障转移——主模型不可用时切换备用；(4) **统一监控**：所有请求经过网关——成本、延迟、错误率集中监控。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| LLM 路由层 | "模型选择器" | 根据请求特征选择最佳 LLM——统一 API、故障转移、成本追踪 |
| 智能路由 | "自动选模型" | 根据查询复杂度、成本、延迟等因素自动选择模型 |
| 故障转移 | "备用模型" | 主模型不可用时自动切换到备用模型 |
| 成本追踪 | "看花了多少钱" | 实时监控每个模型的输入/输出 token 数和成本 |

---

## 📚 小结

LLM 路由层解决供应商锁定——统一 API、智能路由、故障转移、成本追踪。三种架构：LiteLLM（开源）、OpenRouter（托管）、Portkey（生产级）。智能路由根据查询复杂度选择模型——简单用小模型，复杂用大模型。

---

## ✏️ 练习

1. **【设计】** 为一个客服应用设计 LLM 路由策略——什么场景用 GPT-4o，什么场景用 GPT-4o-mini
2. **【实现】** 实现简单的故障转移网关——主模型失败时自动切换到备用

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 路由网关 | `code/main.py` | 智能路由 + 故障转移 + 成本追踪 |

---

## 📖 参考资料

1. [GitHub] LiteLLM: https://github.com/BerriAI/litellm
2. [GitHub] OpenRouter: https://openrouter.ai
3. [GitHub] Portkey: https://github.com/Portkey-AI/portkey-cookbook

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
