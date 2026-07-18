# LLM 可观测性技术栈选型

> 2026 年的可观测性市场分成两类。开发平台（LangSmith、Langfuse、Comet Opik）捆绑监控、评估、提示词管理、会话回放。网关/检测工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 开源内核，提供 50K 事件/月免费云服务。Phoenix 是 OpenTelemetry 原生的，擅长漂移/RAG 可视化但不适合持久化生产后端。生产中的常见模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/Langfuse），通过 OpenTelemetry 粘合。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 08（推理指标）、阶段 14（智能体工程）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分开发平台（捆绑评估+提示词+会话）和网关/遥测工具（仅追踪+指标）
- [ ] 将六个主要工具映射到其许可、定价和最佳场景
- [ ] 解释通过 OpenTelemetry 粘合模式将网关工具与独立评估平台组合
- [ ] 命名 2026 年的成本差异化因素（Arize AX 的零拷贝方法 vs 单体化摄取）

---

## 1. 问题

你发布了一个 LLM 功能。它能工作。你对提示词失败、工具循环、延迟回归、成本飙升、提示词缓存命中率一无所知。你搜索"LLM 可观测性"，得到八个工具，都声称在三个不同价位解决同一个问题。

它们解决的不是同一个问题。LangSmith 回答"这次 LangGraph 运行为什么失败？"Phoenix 回答"我的 RAG 流水线是否在漂移？"Helicone 回答"哪个应用在烧词元？"Langfuse 回答"我能自托管整个系统吗？"不同工具，不同受众。

选型涉及四个维度：技术栈（LangChain？原始 SDK？多供应商？）、许可证容忍度（只要 MIT？Elastic 可以？商业也行？）、预算（免费层？$100/月？$1000/月？）、自托管（必须？有最好？永远不要？）。

---

## 2. 概念

### 2.1 两类工具

**开发平台**捆绑可观测性与评估、提示词管理、数据集版本控制、会话回放。你运行实验，看哪个提示词有效，对新提示词做数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**检测推理调用——提示词、响应、词元、延迟、模型、成本。极简。可以通过 OpenTelemetry 与独立的评估工具组合。

### 2.2 Langfuse——开源平衡

- 核心 Apache/MIT 许可；通过 Docker 自托管
- 云免费层：50K 事件/月。付费版 $29/月
- 评估、提示词管理、追踪、数据集——四大开发平台功能的合理覆盖
- 最佳场景：你需要 LangSmith 级功能但必须自托管或使用 OSS 许可

### 2.3 Phoenix (Arize)——遥测优先、OpenTelemetry 原生

- Elastic License 2.0；自托管简单
- 擅长 RAG 和漂移可视化；嵌入空间散点图是一等公民
- 不设计为持久化生产后端——主要是开发时可观测性
- 最佳场景：RAG 流水线开发、漂移调试、配合独立网关用于生产

### 2.4 Arize AX——规模化方案

- 商业。零拷贝数据湖集成（Iceberg/Parquet）
- 声称在规模上比单体可观测性（Datadog 级）便宜约 100 倍
- 最佳场景：>1000 万条追踪/天，已有数据湖，想要 LLM 特定仪表盘但不需要 Datadog 定价

### 2.5 LangSmith——LangChain/LangGraph 优先

- 商业，$39/用户/月。自托管仅在企业版
- LangChain 和 LangGraph 栈的最佳选择
- 最佳场景：团队使用 LangChain，愿意付费

### 2.6 Helicone——代理模式的最小可行方案

- 15-30 分钟设置：将 `OPENAI_API_BASE` 换成 Helicone 代理
- MIT 许可；100K 请求/月免费，付费 $20/月起
- 包含故障切换、缓存、速率限制——同时也充当网关
- 智能体/多步追踪深度不足
- 最佳场景：快速启动、单栈应用、需要网关+可观测性二合一

### 2.7 Opik (Comet)——OSS 开发平台

- Apache 2.0，完全 OSS
- 与 Langfuse 类似的功能集，Comet 背景
- 最佳场景：已在使用 Comet 的 ML 团队

### 2.8 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。2026 年浮现的生产模式：

1. 每次 LLM 调用发出带 GenAI 约定的 OTel
2. 路由到网关（Helicone/Portkey）用于日常
3. 双发到评估平台（Phoenix/Langfuse）用于回归
4. 归档到数据湖（Iceberg）用于长期分析

### 2.9 陷阱：在错误的层检测

在你的智能体框架内检测（如添加 LangSmith 追踪）会将你耦合到该框架。在 HTTP/OpenAI SDK 层检测（通过 OpenLLMetry 或网关）是可移植的。

### 2.10 采样——你无法保留一切

超过 100 万请求/天时，全量追踪保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合；长尾保留原始。

---

## 3. 从零实现

### 第 1 步：采样策略模拟器

```python
import random


def simulate_retention(total_requests, strategy, cost_per_trace=0.001):
    """模拟不同保留策略的成本。"""
    if strategy == "full":
        retained = total_requests
    elif strategy == "errors_only":
        error_rate = 0.05
        retained = int(total_requests * error_rate)
    elif strategy == "sampled":
        retained = int(total_requests * 0.05)  # 5% 采样
    else:  # errors + sampled
        errors = int(total_requests * 0.05)
        sampled = int(total_requests * 0.05)
        retained = errors + sampled

    cost = retained * cost_per_trace
    return {"retained": retained, "cost": cost, "ratio": retained / total_requests}


# 对比不同策略
for strategy in ["full", "sampled", "errors_only", "errors+sampled"]:
    r = simulate_retention(1_000_000, strategy)
    print(f"{strategy:20s}  保留: {r['retained']:>10,}  "
          f"比例: {r['ratio']:>6.1%}  成本: ${r['cost']:,.0f}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 工具选型对照

| 工具 | 许可证 | 定价 | 最佳场景 | 自托管 |
|---|---|---|---|---|
| Langfuse | MIT/Apache | 免费 50K 事件/月 | 需要 OSS 的开发平台 | Docker |
| Phoenix | Elastic 2.0 | 免费自托管 | RAG 漂移调试 | 简单 |
| Arize AX | 商业 | 规模化定价 | >1000 万条追踪/天 | 数据湖集成 |
| LangSmith | 商业 | $39/用户/月 | LangChain/LangGraph 栈 | 企业版 |
| Helicone | MIT | 100K 请求/月免费 | 快速启动，网关+可观测性 | 代理 |
| Opik | Apache 2.0 | 免费 | Comet 用户 | 完全 OSS |

---

## 5. 工程最佳实践

### 5.1 用 OpenTelemetry 粘合

不要将可观测性工具与框架耦合。在 HTTP/OpenAI SDK 层发出 OTel，然后路由到你选择的后端。这样换后端只需改配置，不改代码。

### 5.2 采样是必须的

超过 100 万请求/天时，全量保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。

### 5.3 中文场景特别建议

- **Langfuse 在国内可用。** Langfuse 是开源的，可以自托管。中文团队可以部署在国内服务器上
- **Helicone 代理延迟。** 在国内网络环境下，Helicone 的代理可能增加 20-50ms 延迟。对于 TTFT SLA 严格的应用需要评估
- **OpenTelemetry 的中文生态。** OpenTelemetry 是语言无关的，中文团队可以完全使用。国内有活跃的 OpenTelemetry 中文社区

---

## 6. 常见错误

### 错误 1：在框架层检测

**现象：** 从 LangChain 切换到原生 SDK 时，所有可观测性追踪丢失。

**原因：** 在 LangChain 框架内添加了 LangSmith 追踪，切换框架后追踪代码无法工作。

**修复：** 在 HTTP/OpenAI SDK 层检测（通过 OpenLLMetry 或网关），不绑定到特定框架。

### 错误 2：不采样就上线

**现象：** 生产环境每天 500 万条追踪，存储费用每月 $50K——超过 LLM 调用费用。

**原因：** 全量保留所有追踪。没有采样策略。

**修复：** 按规则采样：100% 错误、100% 高成本、5% 成功。

---

## 7. 面试考点

### Q1：开发平台和网关/遥测工具的区别是什么？（难度：⭐⭐）

**参考答案：**
开发平台（LangSmith、Langfuse）捆绑了评估、提示词管理、数据集版本控制、会话回放——适合迭代和调试。网关/遥测工具（Helicone、Phoenix）专注于遥测——追踪、指标、成本分析。两者可以通过 OpenTelemetry 粘合：网关用于日常流量，评估平台用于回归测试。生产中的常见模式是网关 + 评估平台的组合。

### Q2：OpenTelemetry 粘合模式的优势是什么？（难度：⭐⭐⭐）

**参考答案：**
优势是可移植性。如果你在 HTTP/OpenAI SDK 层发出 OTel 追踪，切换 LLM 提供商或框架时不需要修改追踪代码——只需更改后端配置。如果你在 LangChain 层添加了 LangSmith 追踪，切换到原生 SDK 时所有追踪丢失。OTel 粘合让你可以在网关（Helicone）做日常监控，在评估平台（Phoenix）做回归测试，在数据湖（Arize AX）做长期分析——三者通过标准协议互操作。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| OpenLLMetry | "LLM 的 OTel" | 开源的 OpenTelemetry LLM 检测 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性名 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "OSS 版 LangSmith" | MIT 开源，类似功能集 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生的开发/评估平台 |
| Helicone | "代理可观测性" | HTTP 代理收集 LLM 遥测 + 网关功能 |
| 会话回放 | "追踪重放" | 重新运行完整的智能体会话含工具调用 |
| 评估 | "离线测试" | 在标注数据集上运行候选模型/提示词 |

---

## 📚 小结

LLM 可观测性市场分成两类：开发平台（捆绑评估+提示词+会话）和网关/遥测工具（仅追踪+指标）。通过 OpenTelemetry 粘合模式可以将两者组合——网关做日常监控，评估平台做回归测试。采样是必须的：超过 100 万请求/天时全量保留的成本超过 LLM 调用本身。在 HTTP/SDK 层检测而非框架层，确保可移植性。

---

## ✏️ 练习

1. 你的团队使用 LangChain，想要 OSS 自托管可观测性。在 Langfuse 和 Opik 之间选择并说明理由。
2. 每天 500 万条追踪，Datadog 报价 $15 万/月。计算 Arize AX 的盈亏平衡点。
3. 设计一个组织指南，规定每次 LLM 调用必须发出的 OpenTelemetry GenAI 属性集。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 采样策略模拟器 | `code/main.py` | 不同保留策略的成本对比 |
| 可观测性选型建议 | `outputs/skill-observability-stack.md` | 根据技术栈和规模选择工具 |

---

## 📖 参考资料

1. [官方文档] OpenTelemetry GenAI Semantic Conventions. https://opentelemetry.io/docs/specs/semconv/gen-ai/
2. [文档] Langfuse. https://langfuse.com/
3. [文档] Arize Phoenix. https://docs.arize.com/phoenix
4. [文档] Helicone. https://docs.helicone.ai/
