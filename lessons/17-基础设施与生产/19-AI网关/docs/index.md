# AI 网关——LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关坐在你的应用和模型提供商之间。核心功能是提供商路由、故障切换、重试、速率限制、密钥引用、可观测性、防护栏。2026 年市场分化：LiteLLM 是 MIT 开源，100+ 提供商，但在约 2000 RPS 时崩溃（8GB 内存，级联故障）。Portkey 是控制平面定位（防护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源。Kong AI Gateway 建在 Kong Gateway 之上——基准测试中比 Portkey 快 228%，比 LiteLLM 快 859%。数据驻留决定了自托管决策。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 列举网关的六大核心功能（路由、故障切换、重试、速率限制、密钥引用、可观测性、防护栏）
- [ ] 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例
- [ ] 引用 Kong 基准（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释为什么对 >500 RPS 很重要
- [ ] 根据数据驻留和运维预算选择自托管 vs 托管

---

## 1. 问题

你的产品调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供商有不同的 SDK、错误模型、速率限制和认证方式。你想要故障切换（OpenAI 429 时尝试 Anthropic）、单一凭据存储、统一可观测性和每租户速率限制。

在应用层重新发明这个会将每个服务耦合到每个提供商。网关层将其整合为一个进程、一个 API（通常是 OpenAI 兼容的），向外扇出到提供商。

---

## 2. 概念

### 2.1 六大核心功能

1. **提供商路由** — OpenAI、Anthropic、Gemini、自托管等在一个 API 后面
2. **故障切换** — 429、5xx 或质量失败时重试其他提供商
3. **重试** — 指数退避，有界尝试
4. **速率限制** — 每租户、每密钥、每模型
5. **密钥引用** — 运行时从保险库拉取凭据（永远不在应用中）
6. **可观测性** — OTel + GenAI 属性 + 成本归因
7. **防护栏** — PII 脱敏、越狱检测、允许话题过滤

### 2.2 LiteLLM——MIT 开源，Python

- 100+ 提供商，OpenAI 兼容，路由器配置，故障切换，基础可观测性
- 在 Kong 的基准测试中约 2000 RPS 时崩溃；8GB 内存占用，持续负载下级联故障
- 最佳匹配：Python 应用，<500 RPS，开发/测试网关，实验路由

### 2.3 Portkey——控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。防护栏、PII 脱敏、越狱检测、审计追踪
- 每请求 20-40ms 延迟开销
- 生产层 $49/月（含保留和 SLA）
- 最佳匹配：需要防护栏+可观测性捆绑的受监管行业

### 2.4 Kong AI Gateway——规模化方案

- 建在 Kong Gateway 之上（成熟的 API 网关产品，lua+OpenResty）
- Kong 自己的基准：12 CPU 等效上比 Portkey 快 228%，比 LiteLLM 快 859%
- 定价：$100/模型/月，Plus 层最多 5 个模型
- 最佳匹配：已在使用 Kong；>1000 RPS；愿意购买许可证

### 2.5 延迟预算

- LiteLLM：典型 5-15ms 开销
- Portkey：20-40ms 开销
- Kong：3-8ms 开销
- Cloudflare/Vercel：1-3ms 开销（边缘优势）

网关延迟直接叠加到 TTFT。TTFT P99 < 100ms SLA 时选 Kong 或 Cloudflare。TTFT P99 < 500ms 时任何都行。

### 2.6 自托管 vs 托管

数据驻留是决定因素。医疗和金融默认自托管（LiteLLM、Portkey OSS 或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合方案：受监管租户自托管，其他托管。

---

## 3. 从零实现

### 第 1 步：带故障切换的网关模拟

```python
import random


class MockGateway:
    """简化版网关——带故障切换和重试。"""

    def __init__(self, providers):
        self.providers = providers

    def route(self, prompt, provider_idx=0):
        """路由到提供商，失败时故障切换。"""
        for i in range(provider_idx, len(self.providers)):
            provider = self.providers[i]
            # 模拟 5% 错误率
            if random.random() < 0.05:
                continue
            return {"provider": provider, "status": "success", "cost": 0.01}
        return {"provider": None, "status": "all_failed", "cost": 0}


# 演示
gw = MockGateway(["openai", "anthropic", "self-hosted"])
results = [gw.route("测试") for _ in range(100)]
success = sum(1 for r in results if r["status"] == "success")
print(f"成功率: {success}/100")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 网关选型对照

| 工具 | 许可 | 规模上限 | 最佳场景 |
|---|---|---|---|
| LiteLLM | MIT | ~500 RPS | Python 开发/测试 |
| Portkey | Apache 2.0 | 中等 | 受监管行业+防护栏 |
| Kong AI | 商业 | >1000 RPS | 已在 Kong 生态 |
| Bifrost | 商业 | 中等 | 重试+故障切换 |
| Cloudflare | 托管 | 边缘 | JS 应用+零运维 |

---

## 5. 工程最佳实践

### 5.1 网关延迟直接叠加到 TTFT

如果 SLA 是 TTFT P99 < 100ms，网关延迟（20-40ms）直接吃掉 20-40% 的预算。选择低延迟网关（Kong 3-8ms、Cloudflare 1-3ms）。

### 5.2 密钥永远不在应用中

网关应该从保险库拉取凭据——应用代码永远不接触 API 密钥。

### 5.3 中文场景特别建议

- **国内网关方案。** LiteLLM 和 Portkey 都可以在国内部署。Kong 在国内有阿里云等合作伙伴
- **国内 LLM 的网关需求。** 国内 LLM API（通义千问、文心一言、讯飞星火）的错误模型和速率限制各不相同——网关可以统一管理
- **数据驻留。** 国内监管要求数据存储在中国境内。自托管网关（LiteLLM/Portkey OSS）可以部署在境内服务器上

---

## 6. 常见错误

### 错误 1：用 LiteLLM 处理 >1000 RPS

**现象：** 流量超过 1000 RPS 后网关响应时间飙升，出现级联故障。

**原因：** LiteLLM 在约 2000 RPS 时崩溃——8GB 内存限制，Python GIL 限制了并发。

**修复：** 流量超过 500 RPS 时迁移到 Kong 或 Portkey 托管版。

### 错误 2：密钥硬编码在应用中

**现象：** API 密钥泄露在日志或代码仓库中。

**原因：** 凭据直接写在代码里，没有通过网关或保险库管理。

**修复：** 使用网关的密钥引用功能——从 Vault 拉取凭据，应用代码永远不接触密钥。

---

## 7. 面试考点

### Q1：网关的七大核心功能是什么？（难度：⭐⭐）

**参考答案：**
提供商路由（统一 API 后面多个提供商）、故障切换（失败时重试其他提供商）、重试（指数退避）、速率限制（每租户/每密钥/每模型）、密钥引用（从保险库拉取凭据）、可观测性（OTel + GenAI 属性 + 成本归因）、防护栏（PII 脱敏、越狱检测）。网关的价值是将应用层与提供商层解耦——应用代码只调用一个统一的 API。

### Q2：为什么 Kong 在基准测试中比 LiteLLM 快 859%？（难度：⭐⭐）

**参考答案：**
Kong 建在 Kong Gateway 之上——使用 lua+OpenResty，是成熟的高性能 API 网关。OpenResty 基于 Nginx 的事件驱动架构，每个请求占用极少内存。LiteLLM 是 Python 实现——受 GIL 限制，在高并发下性能下降。Kong 在 12 CPU 等效上处理的 RPS 远高于 LiteLLM。但 LiteLLM 的优势是 Python 生态和 100+ 提供商支持——选择取决于规模和生态需求。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 网关 | "API 代理" | 坐在应用和提供商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 提供商，2000 RPS 时崩溃 |
| Portkey | "防护栏网关" | 控制平面+可观测性，Apache 2.0 |
| Kong AI Gateway | "规模化那个" | 建在 Kong Gateway 之上，基准测试冠军 |
| PII 脱敏 | "数据清洗" | 正则+NER 在发送到模型前遮蔽敏感信息 |
| 越狱检测 | "提示词注入防护" | 对用户输入运行分类器 |
| 审计追踪 | "受监管日志" | 每次 LLM 调用的不可变记录 |
| 速率限制 | "令牌桶/滑动窗口" | 基于令牌桶或滑动窗口的速率限制器 |

---

## 📚 小结

AI 网关将应用层与多个 LLM 提供商解耦——统一 API、故障切换、速率限制、密钥管理。2026 年四大网关各有定位：LiteLLM（MIT 开源，Python 生态，<500 RPS）、Portkey（Apache 2.0，防护栏+可观测性，受监管行业）、Kong AI（高性能，>1000 RPS）、Cloudflare/Vercel（托管，边缘）。网关延迟直接叠加到 TTFT——选择时必须考虑延迟预算。数据驻留决定了自托管 vs 托管的决策。

---

## ✏️ 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管的故障切换。在 5% 提供商错误率下的预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200ms，基线 300ms。哪些网关在预算内？
3. 一个医疗客户需要自托管 + PII 脱敏 + 审计。选择 Portkey OSS 还是 Kong。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 网关路由模拟器 | `code/main.py` | 带故障切换的网关路由 |
| 网关选型建议 | `outputs/skill-gateway-picker.md` | 根据规模和合规选择网关 |

---

## 📖 参考资料

1. [GitHub] LiteLLM. https://github.com/BerriAI/litellm
2. [GitHub] Portkey Gateway. https://github.com/Portkey-AI/gateway
3. [官方文档] Kong AI Gateway. https://docs.konghq.com/gateway/latest/ai-gateway/
4. [基准] Kong AI Gateway Benchmark. https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm
