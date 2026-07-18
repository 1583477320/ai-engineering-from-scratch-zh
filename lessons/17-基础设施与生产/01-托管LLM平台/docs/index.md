# 托管 LLM 平台——Bedrock、Vertex AI、Azure OpenAI

> 三家云厂商，三种不同的策略。AWS Bedrock 是模型市场，Azure OpenAI 是独家合作，Vertex AI 是 Gemini 优先。2026 年的决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 能力匹配我的产品"。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 11（LLM 工程）、阶段 13（工具与协议）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出三家云厂商的平台策略（市场 vs 独家 vs Gemini 优先）并将每种策略匹配到产品用例
- [ ] 解释 Provisioned Throughput Units 在 Azure 上的作用，以及为什么按需模式的 Bedrock 在 405B 规模下通常慢约 25ms
- [ ] 画出三家平台的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex 项目制 vs Azure 作用域 + PTU 预留）
- [ ] 写出"至少两家供应商"策略并解释为什么单厂商锁定在 2026 年是代价最高的错误

---

## 1. 问题

你选定了 Claude 3.7 Sonnet 作为你的产品模型。现在需要让它上线提供服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过一个网关。直接调用最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障切换、统一计费和跨提供商的速率限制。

更深层的问题是模型目录。如果你在同一款产品中需要 Claude、Llama 和 Gemini，你不能从一个地方买到全部——除非你同时使用 Bedrock、Vertex AI 和 Azure OpenAI。三家云厂商是不可互换的——它们各自对"谁来拥有模型层"做出了不同的赌注。

---

## 2. 概念

### 2.1 三种策略

**AWS Bedrock——模型市场。** Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入）、Mistral，外加图像和嵌入子目录。一个 API，一个 IAM 面，一个 CloudWatch 导出。Bedrock 的赌注是客户更需要可选择性而不是单一的模型。

**Azure OpenAI——独家合作。** 你获得 GPT-4/4o/5/o-series、DALL-E、Whisper 以及 Azure 数据中心内的 OpenAI 模型微调。在 "Azure OpenAI Service" 目录中没有非 OpenAI 模型——那些归入 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 仍然保持着前沿，而客户希望在这个特定关系上获得企业控制。

**Vertex AI——Gemini 优先，其他在后。** Gemini 1.5/2.0/2.5 Flash 和 Pro，外加 Model Garden（第三方）。Vertex 的赌注是多模态长上下文——1M 词元的 Gemini 上下文是差异化优势。

### 2.2 规模下的延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 的中位首次词元延迟约 50ms；Bedrock 约 75ms。这个差距不是 AWS 的失败——而是不同的容量模型。Azure 销售 PTU（预置吞吐量单元），为你的租户预留 GPU 容量。Bedrock 的等效方案（Provisioned Throughput）存在但起步价为每单元约 $21/小时，大多数客户停留在共享按需模式。

按需共享容量与所有其他客户的流量竞争。专用容量不用。如果你的产品 SLA 要求 P99 TTFT < 100ms，你要么在 Azure 上买 PTU，要么买 Bedrock Provisioned Throughput，要么接受默认的方差。

### 2.3 预置吞吐量的经济学

Azure PTU：预留的推理计算块。对于可预测的工作负载，相比按需可节省高达约 70%。无论流量如何，按固定小时收费——即使空闲也要付钱。盈亏平衡点通常在约 40-60% 的持续利用率。

Bedrock Provisioned Throughput：$21-$50/小时，取决于模型和区域。类似的数学——盈亏平衡点是约一半的峰值利用率。

Vertex 预置容量按 Gemini SKU 销售；定价因模型和地区而异，不太公开。

### 2.4 FinOps 面——真正的差异化因素

**Bedrock Application Inference Profiles** 是市场上最干净的归因方式。用 `team`、`product`、`feature` 标签标记一个配置；将所有的模型调用路由通过它；CloudWatch 无需后处理即可按配置分解成本。2025 年新增，仍然是粒度最高的云厂商原生方案。

**Vertex** 的归因方式是项目制 + 全量标签。你将每个团队建模为一个 GCP 项目，在每项资源上打标签，使用 BigQuery 计费导出 + DataStudio 做汇总。工作量更大，但 BigQuery 让你可以在成本数据上运行任意 SQL。

**Azure** 依赖于订阅/资源组作用域加上标签，PTU 预留作为一等成本对象。标签从资源组继承，而不是从请求继承——所以每条请求的归因需要 Application Insights 自定义指标或一个打标签的网关。

模式：Bedrock 的原生方案最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明除非你做了额外的检测。

### 2.5 锁定是 2026 年的风险

单一云厂商的承诺在一个模型占主导地位的时期是可以接受的。但在 2026 年，前沿每个月都在移动——Claude 3.7 这一季，Gemini 2.5 下一季，GPT-5 再下一季。锁定在一个平台上，你就被锁在了三分之二的前沿之外。

有效团队采纳的模式：**任何产品级的 LLM 调用至少使用两家供应商。** Bedrock + Azure OpenAI 是常见的配对——Claude 来自一家，GPT 来自另一家，在它们之间故障切换，同一个网关。成本增加可以忽略不计，因为网关可以路由到最优的低价方；但宕机时的可用性提升（如 Azure OpenAI 在 2025 年 1 月的事故、AWS us-east-1 的宕机）是决定性的。

### 2.6 数据驻留与受监管行业

Bedrock：大多数区域有 BAA；VPC 端点；防护栏。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；受监管企业的默认选择。
Vertex：HIPAA、GDPR、各区域数据驻留；Google Cloud 的合规堆栈。

三家都满足基本的合规要求。区别在于数据保留策略、日志处理方式以及滥用监控是否读取你的流量（大多数默认 opt-in；企业版提供 opt-out）。

### 2.7 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位 TTFT：约 50ms（使用 PTU）
- Bedrock 按需模式中位 TTFT：约 75ms
- Bedrock Provisioned Throughput：$21-$50/小时/单元
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率
- PTU 在高度用下相比按需节省：高达约 70%

---

## 3. 从零实现

### 第 1 步：平台对比模拟器

```python
@dataclass
class Workload:
    model: str
    tokens_per_day: int
    sustained_pct: float  # 0~1，持续利用率
    peak_pct: float       # 0~1，峰值利用率
    latency_sla_ms: int   # P99 TTFT SLA


def compare_platforms(w: Workload):
    """模拟三家平台的成本和延迟对比。"""
    # 简化的定价模型
    platforms = {
        "Bedrock 按需": {"ttft_ms": 75, "cost_per_mt": 1.50},
        "Bedrock PT":  {"ttft_ms": 55, "cost_per_hour": 35.0},
        "Azure PTU":   {"ttft_ms": 50, "cost_per_hour": 30.0},
        "Vertex 按需": {"ttft_ms": 65, "cost_per_mt": 1.60},
    }

    daily_tokens_m = w.tokens_per_day / 1_000_000

    for name, p in platforms.items():
        if "hour" in p:
            # 预置模式：按小时固定收费
            daily_cost = p["cost_per_hour"] * 24
        else:
            daily_cost = daily_tokens_m * p["cost_per_mt"]

        meets_sla = p["ttft_ms"] <= w.latency_sla_ms
        print(f"  {name:15s} 延迟={p['ttft_ms']}ms "
              f"每日费用=${daily_cost:.1f} SLA={'✓' if meets_sla else '✗'}")
```

### 第 2 步：两家供应商策略

```python
def two_provider_plan(primary: str, fallback: str):
    """输出两家供应商的部署计划。"""
    print(f"\n主要提供商: {primary}")
    print(f"备用提供商: {fallback}")
    print(f"网关: 统一 API 网关，带故障切换")
    print(f"切换条件: P99 延迟 > 2s 或错误率 > 5%")
    print(f"成本影响: 通过智能路由，成本增加 < 5%")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 统一网关

```python
# LiteLLM 示例
import litellm

# 同一 API 调用不同提供商
response = litellm.completion(
    model="bedrock/claude-3.5-sonnet",
    messages=[{"role": "user", "content": "分析数据"}],
    fallback_models=["azure/gpt-4o"],  # 超时或错误时自动切换
)
```

### 4.2 平台选型对照表

| 维度 | Bedrock | Azure OpenAI | Vertex AI |
|---|---|---|---|
| 策略 | 模型市场 | 独家合作 | Gemini 优先 |
| 中位 TTFT | ~75ms | ~50ms (PTU) | ~65ms |
| 归因粒度 | Application Profiles（最细） | 资源组 + Tags（需自建） | BigQuery 导出（最灵活） |
| 合规 | BAA、VPC | HIPAA、SOC 2 | HIPAA、GDPR |
| 模型广度 | 最广 | 仅 OpenAI | Gemini + Model Garden |
| 盈亏平衡 | PT ~$21-$50/h | PTU ~30-40/h | 不公开 |

---

## 5. 工程最佳实践

### 5.1 至少两家供应商

任何产品级 LLM 调用必须至少路由到两个提供商。原因不是成本——原因是可用性。2025 年 Azure OpenAI 宕机 6 小时、AWS us-east-1 宕机 4 小时——单一依赖意味着你的产品跟着宕机。

### 5.2 从按需起，到 PTU 升级

不要一开始就签 PTU 合同。先用按需模式运行，测量你真实的持续利用率。当利用率稳定超过 40-50% 时才切换。

### 5.3 FinOps 从第一天做起

一旦部署了三个模型 × 两个提供商 × 两个环境（生产/暂存），成本数据就变得不可管理了。从第一天就在每个 API 调用上打标签 `team`、`product`、`feature`。Bedrock Application Inference Profiles 是最干净的方式。

### 5.4 中文场景特别建议

- **国内替代方案：** 三家云厂商的国内版本（AWS 中国、Azure 中国、Google Cloud 中国）模型目录不同。Azure 中国没有 GPT-5 的访问权；AWS 中国的 Bedrock 模型目录也受限。国内团队需要额外评估阿里云百炼、百度千帆等本土平台
- **数据驻留要求：** 中国监管要求所有数据存储在中国境内。Bedrock 和 Azure OpenAI 的中国区版本满足要求，但功能更新比国际版滞后 3-6 个月
- **PTU 在国内的可用性：** Azure 中国区的 PTU 定价和可用性与国际版不同，需要单独询价

---

## 6. 常见错误

### 错误 1：单提供商锁定

**现象：** 一个 API 提供商宕机，产品完全不可用。

**原因：** 所有 LLM 调用只路由到一个提供商。

**修复：** 至少配置两家提供商，网关自动故障切换。

### 错误 2：盲目签 PTU

**现象：** 签了一年 PTU 合同后才发现实际利用率只有 20%——每月多付了 3 倍的钱。

**原因：** 没有先用按需模式测量真实利用率就签了长期合同。

**修复：** 先用按需跑 1-2 个月，测量持续利用率，确认超过 40% 后再签 PTU。

---

## 7. 面试考点

### Q1：为什么 Azure OpenAI 的延迟（~50ms）比 Bedrock（~75ms）低？（难度：⭐⭐）

**参考答案：**
区别在于容量模型。Azure 销售 PTU（预置吞吐量单元），为你的租户预留专用 GPU 容量。按需模式你的请求和所有其他客户争抢共享资源。Bedrock 的大多数客户使用按需模式，因为其 Provisioned Throughput 起步价为 $21/小时，大多数客户不选择升级。这不是硬件差异——是容量模型差异。

### Q2：什么是"两家供应商最低"策略？它的成本影响是什么？（难度：⭐⭐）

**参考答案：**
任何产品级 LLM 调用至少路由到两个不同的云厂商，网关在两者之间做故障切换。成本影响很小（<5%），因为网关可以智能路由到最优的低价方。收益是对云厂商宕机的免疫力——2025 年多家云厂商出现过多小时宕机，单一依赖意味着你的产品跟着宕机。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Bedrock | "AWS 的 LLM 服务" | Claude、Llama、Titan、Mistral 等模型的统一市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心内的独家 OpenAI 模型 + 企业控制 |
| Vertex AI | "Google 的 LLM" | Gemini 优先 + Model Garden 第三方模型 |
| PTU | "预置容量" | 预置吞吐量单元——预留推理 GPU，按小时定价 |
| Application Inference Profile | "Bedrock 标签" | 按产品/功能/团队的 Bedrock 成本和用量 Profile |
| 两家供应商最低 | "LLM 冗余" | 每条 LLM 路径至少走两个云厂商的策略 |

---

## 📚 小结

三家云厂商代表了三种不同的策略：Bedrock 的市场模式（可选性）、Azure 的独家模式（企业控制）、Vertex 的 Gemini 优先模式（多模态长上下文）。延迟差异（~50ms vs ~75ms）来自容量模型而不是硬件。FinOps 归因是真正的差异化因素——Bedrock 的标签最干净，Vertex 的 BigQuery 最灵活。2026 年的工程共识：至少两家供应商，从按需起，在利用率达到 40% 以上时再升级到 PTU。

下一课我们将讨论推理平台经济学——在 Fireworks、Together、Baseten 等独立平台之间做选择的决策框架。

---

## ✏️ 练习

1. 运行 `code/main.py`。在什么持续利用率下 PTU 比按需更划算？计算你和文中 40-60% 经验法则的差异。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个两家供应商的部署方案——谁做主、谁做备、用什么网关、什么切换策略？
3. 一个受监管的医疗客户要求 BAA、美东数据驻留和 <100ms P99 TTFT。选一个平台并用三个具体特性说明理由。
4. 读 Azure OpenAI 和 Bedrock 的定价页。对于每月 1 亿词元的 Claude 工作负载，直接 Anthropic API、Bedrock 按需、Bedrock PTU 哪个最便宜？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 平台对比模拟器 | `code/main.py` | 三家平台的成本和延迟对比 |
| 托管平台选型建议 | `outputs/skill-managed-platform-picker.md` | 根据工作负载推荐主平台和备用方案 |

---

## 📖 参考资料

1. [定价] AWS Bedrock Pricing. https://aws.amazon.com/bedrock/pricing/
2. [定价] Azure OpenAI Service Pricing. https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
3. [基准] Artificial Analysis LLM Leaderboard. https://artificialanalysis.ai/
4. [报告] Finout — Bedrock vs Vertex vs Azure FinOps. https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend
