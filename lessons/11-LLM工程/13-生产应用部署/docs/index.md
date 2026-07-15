# 生产应用部署

> 研究论文的"部署"是 `pip install`。生产部署是监控、日志、限流、回滚、A/B 测试、成本控制的总和。从 demo 到产品的鸿沟比你想象的更深。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 11 · 01-12 | **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计 LLM 应用的生产架构——API 网关、缓存、负载均衡
- [ ] 实现监控和可观测性——延迟、成本、质量指标
- [ ] 理解 A/B 测试在 LLM 评估中的应用
- [ ] 设计优雅降级和故障回滚策略

---

## 1. 问题

你的 LLM 应用在 demo 中运行良好。现在有 1000 个并发用户。GPT-4o API 超时率 5%。部分用户触发了有害内容生成。某个 prompt 变体的响应质量下降了 20%——但没人知道为什么。

生产部署不是 `pip install`。它是一个工程系统。

---

## 2. 概念

### 2.1 生产架构

```
用户请求 → API 网关（限流/认证）
    ↓
缓存层（命中缓存→直接返回）
    ↓
LLM 路由（简单任务→小模型，复杂任务→大模型）
    ↓
LLM 推理（OpenAI/Anthropic/自部署）
    ↓
输出审核（安全过滤）
    ↓
响应缓存 + 返回
```

### 2.2 可观测性

| 指标 | 说明 | 报警阈值 |
|------|------|---------|
| 延迟 (P50/P95) | 首 token 延迟 + 总延迟 | P95 > 5s |
| 成本/调用 | 输入+输出 token 成本 | > 预算阈值 |
| 错误率 | API 超时/拒绝/格式错误 | > 1% |
| 安全事件 | 提示注入/有害内容 | > 0 |
| 质量分数 | LLM-as-Judge 分数 | < 基线-10% |

### 2.3 优雅降级

```
主要模型（GPT-4o）不可用
  → 回退到次选模型（Claude Sonnet）
    → 回退到本地小模型
      → 返回缓存的默认响应
```

### 2.4 A/B 测试

```
50% 用户 → Prompt 版本 A
50% 用户 → Prompt 版本 B
比较：转化率、满意度、延迟、成本
```

---

## 3. 从零实现

### Step 1：生产架构模拟

```python
class LLMGateway:
    def __init__(self, models, cache, rate_limit=100):
        self.models = models  # 按优先级排序的模型列表
        self.cache = cache
        self.rate_limit = rate_limit
        self.call_count = 0

    def handle_request(self, prompt, system_prompt):
        # 1. 限流
        if self.call_count >= self.rate_limit:
            return {"error": "速率限制", "retry_after": 60}

        # 2. 缓存检查
        cached = self.cache.get(prompt)
        if cached:
            return {"response": cached, "source": "cache"}

        # 3. 尝试主模型
        for model in self.models:
            try:
                response = self._call_model(model, prompt, system_prompt)
                self.cache.set(prompt, response)
                self.call_count += 1
                return {"response": response, "source": model}
            except Exception:
                continue

        # 4. 回退
        return {"response": "抱歉，服务暂时不可用。请稍后再试。", "source": "fallback"}
```

### Step 2：监控仪表盘

```python
class LLMMetrics:
    def __init__(self):
        self.latencies = []
        self.errors = []
        self.costs = []

    def record(self, latency, cost, error=None):
        self.latencies.append(latency)
        self.costs.append(cost)
        if error:
            self.errors.append(error)

    def summary(self):
        if not self.latencies:
            return "无数据"
        return {
            "avg_latency": sum(self.latencies) / len(self.latencies),
            "p95_latency": sorted(self.latencies)[int(len(self.latencies) * 0.95)],
            "error_rate": len(self.errors) / max(len(self.latencies), 1),
            "total_cost": sum(self.costs),
        }
```

---

## 6. 工程最佳实践

### 6.1 部署清单

- [ ] API 密钥安全管理（环境变量/Vault）
- [ ] 速率限制和配额控制
- [ ] 错误重试策略（指数退避）
- [ ] 日志记录和监控
- [ ] 优雅降级和回滚
- [ ] A/B 测试框架
- [ ] 成本追踪和预算告警

### 6.2 中文场景

- 中文 LLM 部署考虑 CDN 延迟——亚太区域选择
- 中文输出审核需要中文敏感词库

### 6.3 踩坑经验

- **未做限流**：突发流量导致 API 超时和成本飙升
- **未监控成本**：月底发现超出预算 10 倍
- **未做回滚**：新 prompt 上线后质量下降但没有旧版本可以回退

---

## 7. 常见错误

### 错误 1：未做灰度发布

**现象：** 新 prompt 一次性全量上线——50% 用户受到影响。

**原因：** 没有灰度发布机制——无法在小流量上验证新版本。

**修复：** 使用 A/B 测试框架——5% 流量灰度→25%→100%。设置回滚触发条件（如错误率上升或质量下降）。

### 错误 2：依赖单一 LLM 提供商

**现象：** OpenAI API 宕机时整个应用不可用。

**原因：** 没有备用模型或回退策略。

**修复：** 实现多提供商路由——OpenAI 不可用时自动切换到 Anthropic。设置健康检查和自动故障转移。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 生产架构实现 | `code/main.py` | API 网关 + 监控指标 + 优雅降级 |

---

---

## 8. 面试考点

### Q1：LLM 生产应用和传统 Web 应用有什么特殊之处？（难度：⭐⭐）

**参考答案：**
(1) **成本不可预测**：输出长度不确定，每个请求的成本波动大；(2) **延迟不稳定**：LLM 推理是自回归的，输出长度影响延迟；(3) **质量不可控**：相同输入不同输出，需要 A/B 测试和持续监控；(4) **安全边界模糊**：提示注入和有害内容需要多层防护；(5) **模型版本管理**：模型更新（如 API 提供商升级）可能导致现有 prompt 失效。

### Q2：如何设计一个高可用的 LLM 应用？（难度：⭐⭐⭐）

**参考答案：**
三层防护：(1) 多提供商冗余——OpenAI 失效自动切换到 Anthropic；(2) 模型降级——大模型失效切换到小模型；(3) 缓存回退——所有 LLM 都不可用时返回缓存的默认响应。加上健康检查、自动重启、灰度发布。

---

## 📚 小结

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| API 网关 | "前端门卫" | 限流、认证、路由、日志的统一入口 |
| 优雅降级 | "故障时别崩溃" | 主模型不可用时自动切换到次选模型或缓存 |
| A/B 测试 | "同时跑两个版本" | 随机分流用户到不同 prompt/模型版本，比较效果 |
| Prompt 版本管理 | "管理提示词" | 对提示词做版本控制——记录每次修改的效果变化 |

---

## 📚 小结

生产部署 = API 网关 + 缓存 + 路由 + 监控 + 降级 + A/B 测试。监控四维度：延迟、成本、错误率、质量。优雅降级确保高可用。A/B 测试确保变更安全。生产 LLM 系统需要比 demo 更多的工程投资。

---

## ✏️ 练习

1. **【设计】** 为一个客服 LLM 应用设计生产架构图——包括网关、缓存、路由、监控、降级
2. **【实现】** 实现一个简单的 LLM 监控指标收集器——追踪延迟、成本、错误

---

## 📖 参考资料

1. [文档] OpenAI Rate Limits: https://platform.openai.com/docs/guides/rate-limits
2. [博客] Anthropic Deployment Best Practices: https://docs.anthropic.com/en/docs/build-with-claude
3. [文档] LangSmith: https://docs.smith.langchain.com/
