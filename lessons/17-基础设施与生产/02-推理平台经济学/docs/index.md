# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 租赁。它分叉为定制芯片、GPU 平台和 API 优先市场三个细分。每个平台的定价模式不同——每词元、每分钟、每秒——你不能直接对标，必须按工作负载建模才能比较。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 04（vLLM 在线服务内部原理）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出推理平台的三个细分市场（定制芯片、GPU 平台、API 市场）并将每个供应商映射到对应的细分
- [ ] 解释为什么"每词元"API 定价模型趋向于推理引擎的成本曲线，而不是硬件的
- [ ] 计算至少三个供应商之间的单次请求有效成本，并说明每分钟计费（Baseten、Modal）何时击败每词元计费
- [ ] 为给定的工作负载选择正确的默认平台（Serverless 突发型、稳定高吞吐、微调变体、多模态）

---

## 1. 问题

你评估了托管云厂商平台。你决定需要一个更专注、更快的提供商——Fireworks 用于低延迟、Together 用于模型广度、Baseten 用于微调的自定义模型。现在你有六个真正的选择，而它们的定价页面不一致。Fireworks 显示 $/M 词元；Baseten 显示 $/分钟；Modal 显示 $/秒；Replicate 显示 $/预测。不对工作负载建模，你无法直接比较它们。

更糟糕的是，每个定价页面背后的商业模式都不同。Fireworks 在共享 GPU 上运行自己的自定义引擎（FireAttention）；每词元费率反映了它们的利用率曲线。Baseten 给你 Truss + 专用 GPU；每分钟计费反映了独占性。Modal 是真正的 Python serverless——每秒计费，冷启动低于秒级。同样的输出（一个 LLM 响应），三种不同的成本函数。

---

## 2. 概念

### 2.1 三个细分市场

**定制芯片：** Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在相同模型上通常比基于 GPU 的集群快 5-10 倍的解码速度。每词元价格更高（Groq 在 Llama 70B 上约 $0.99/M，2025 年底），但对于延迟敏感的场景无可匹敌。Groq 是语音智能体和实时翻译的生产选择。

**GPU 平台：** Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（H100、H200、B200）或有时 AMD 上。位于"裸 GPU 租赁"（RunPod、Lambda）和"云厂商托管服务"（Bedrock）之间的经济层。

**API 优先市场：** Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒付费，强调从零到首次调用的时间。

### 2.2 Fireworks——延迟优化的 GPU 平台

- FireAttention 引擎（自定义）；号称在等效配置下比 vLLM 低 4 倍的延迟
- 批处理层定价约为 Serverless 层的一半（适用于非交互式工作负载）
- **微调模型以与基础模型相同的费率提供**——这是与那些对 LoRA 加价的提供商相比的真正差异化优势
- 2026 年中：2026 年 5 月 1 日起按需 GPU 租赁涨价 $1/小时。大规模用量可协商
- 财务信号：$4B 估值，每天处理 10T+ 词元

### 2.3 Together——广度优化

- 200+ 模型，包括上游发布后几天内的开源版本
- 在等效 LLM 模型上比 Replicate 便宜 50-70%——"AI 原生云"的定位是量和目录
- 推理 + 微调 + 训练在同一 API 中

### 2.4 Baseten——企业级打磨优化

- Truss 框架：模型打包 + 依赖、密钥、服务配置在一个清单中
- GPU 范围从 T4 到 B200。每分钟计费，冻启动缓解合理
- SOC 2 Type II、HIPAA 就绪。常见的金融科技和医疗选择
- $5B 估值，2026 年 1 月 E 轮融资（$3 亿，来自 CapitalG、IVP、NVIDIA）

### 2.5 Modal——Python 原生体验优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令部署
- 每秒计费。冷启动 2-4s（预热身）；小模型 <1s
- $87M B 轮融资，估值 $1.1B（2025 年）。在独立调查中开发者体验得分最高

### 2.6 Replicate——多模态广度

- 按预测计费。图像、视频和音频模型的默认平台
- 集成生态（Zapier、Vercel、CMS 插件）
- 在 LLM 每词元费率上竞争力较弱，但在多模态多样性上胜出

### 2.7 Anyscale——Ray 原生

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）
- 最适合分布式 Python 工作负载——推理步骤是大图中的一个节点
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成

### 2.8 每词元 vs 每分钟——什么时候哪个胜出

每词元计费在工作负载对延迟不敏感且突发性强时合理——你只为用了的部分付费。每分钟计费在利用率高且可预测时合理——一旦你饱和了 GPU，综合成本就低于每词元。

经验法则：对于持续利用率高于 ~30% 的专用 GPU 工作负载，每分钟计费（Baseten、Modal）开始击败每词元计费（Fireworks、Together）。低于这个阈值，每词元胜出——因为你避免了为空闲付费。

### 2.9 自定义引擎是真正的护城河

上述每个平台都在 vLLM 和 SGLang 之上声称拥有自定义引擎。FireAttention、RayTurbo、Baseten 的推理栈。自定义引擎的声明中营销水分不小——诚实地说，vLLM + SGLang 代表了开源推理生产环境的大约 80%，而平台层的差异化因素是开发者体验、归因和 SLA。

### 2.10 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起涨价 $1/小时
- Fireworks 声称：等效配置下比 vLLM 低 4x 延迟
- Together：在 LLM 上比 Replicate 便宜 50-70%
- Baseten 估值：$5B（2026 年 1 月 E 轮，$3 亿）
- Modal 估值：$1.1B（2025 年 B 轮）
- 每分钟计费在持续利用率 > 30% 时击败每词元计费

---

## 3. 从零实现

### 第 1 步：六家供应商成本对比模拟器

```python
@dataclass
class InferenceWorkload:
    model_size: str          # "7B" | "70B" | "405B"
    tokens_per_day: int
    sustained_utilization: float  # 0~1
    peak_requests_per_sec: int


VENDORS = {
    "Fireworks 按需": {"type": "per_token", "rate_per_m": 0.60},
    "Fireworks 批处理": {"type": "per_token", "rate_per_m": 0.30},
    "Together": {"type": "per_token", "rate_per_m": 0.55},
    "Baseten": {"type": "per_min", "rate_per_min": 0.80},
    "Modal": {"type": "per_sec", "rate_per_sec": 0.015},
    "Replicate": {"type": "per_prediction", "rate_per_pred": 0.015},
}


def compare_cost(w: InferenceWorkload):
    """比较六家供应商的日成本和有效每 M 词元成本。"""
    daily_tokens_m = w.tokens_per_day / 1_000_000
    daily_minutes = 24 * 60 * w.sustained_utilization

    for name, v in VENDORS.items():
        if v["type"] == "per_token":
            daily_cost = daily_tokens_m * v["rate_per_m"]
        elif v["type"] == "per_min":
            daily_cost = daily_minutes * v["rate_per_min"]
        elif v["type"] == "per_sec":
            daily_cost = daily_minutes * 60 * v["rate_per_sec"]
        else:
            daily_cost = w.peak_requests_per_sec * 86400 * v["rate_per_pred"]

        eff_rate = daily_cost / daily_tokens_m if daily_tokens_m > 0 else 0
        print(f"  {name:20s} 日费=${daily_cost:.1f} 有效=${eff_rate:.3f}/M")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 平台选型对照表

| 平台 | 定价模式 | 最佳场景 | 差异化优势 | 2026 估值 |
|---|---|---|---|---|
| Fireworks | 每词元 | 延迟敏感的文本生成 | FireAttention 引擎，微调无加价 | $4B |
| Together | 每词元 | 需要大量模型可选 | 200+ 模型，比 Replicate 便宜 50-70% | — |
| Baseten | 每分钟 | 企业级微调模型 | Truss 框架，SOC 2/HIPAA | $5B |
| Modal | 每秒 | Python 原生开发体验 | 纯 Python IaC，冷启动 <1s | $1.1B |
| Replicate | 每预测 | 多模态（图像/视频/音频） | Zapier/Vercel 集成 | — |
| Anyscale | 每分钟 | Ray 生态分布式工作负载 | RayTurbo + Ray Serve | — |

### 4.2 选型决策树

```
工作负载类型？
├── 延迟敏感（语音、实时翻译）→ Groq（定制芯片）
├── 稳定高吞吐文本生成
│   ├── 需要微调模型 → Baseten
│   └── 不需要微调 → Fireworks 或 Together
├── 多模态（图像+视频+音频）→ Replicate
├── Python 原生开发 → Modal
└── Ray 生态分布式 → Anyscale
```

---

## 5. 工程最佳实践

### 5.1 按利用率选择定价模式

30% 持续利用率是分水岭。低于 30% → 每词元（Fireworks、Together）。高于 30% → 每分钟/秒（Baseten、Modal）。

### 5.2 批处理层是你的朋友

如果你的工作负载有可预测的闲时（夜间、周末），配置一个批处理队列。Fireworks 的批处理层约为 Serverless 层的一半价格。

### 5.3 不要为自定义引擎的营销买单

每个平台都声称自定义引擎比 vLLM 快 X 倍。在生产环境中用自己的工作负载验证，不要相信宣称的倍数。

### 5.4 中文场景特别建议

- **国内访问延迟。** 上述平台在中国大陆的访问延迟较高（200-500ms）。如果用户主要在中国，建议优先考虑国内推理平台（阿里云 PAI-EAS、百度千帆、火山引擎）
- **国内微调模型支持。** Baseten 的 Truss 框架在国内不可用。如果主要使用 LoRA 微调模型，可以考虑阿里云的 PAI-EAS + 自定义引擎

---

## 6. 常见错误

### 错误 1：不看工作负载直接比较每词元费率

**现象：** 选择了一个每词元看起来很便宜的供应商，但月度账单更高。

**原因：** 利用率高的场景下，每分钟计费比每词元更划算。

**修复：** 先建模你的工作负载（日词元数、持续利用率），再按多种定价模式计算。

### 错误 2：忽视冷启动成本

**现象：** Serverless 平台的账单里出现了大量"空闲"费用。

**原因：** 每分钟/秒计费平台即使空闲也要收费。如果你有 50% 的时间是空闲的，你为那一半时间付了钱。

---

## 7. 面试考点

### Q1：什么情况下每分钟/秒计费比每词元计费更划算？（难度：⭐⭐）

**参考答案：**
当持续利用率超过约 30% 时。每分钟计费（Baseten、Modal）按墙钟时间收费——无论你用不用 GPU，都按固定费率计费。每词元计费（Fireworks、Together）按实际消耗的词元收费——空闲时不收费。所以低利用率时每词元胜出；当你饱和 GPU 时每分钟胜出。

### Q2：为什么 Fireworks 声称 LoRA 微调模型按基础模型费率收费是一个差异化优势？（难度：⭐⭐）

**参考答案：**
大多数提供商对 LoRA 请求收取高于基础模型的溢价，因为 LoRA 加载和切换增加了工程复杂性。Fireworks 不收取溢价，意味着微调模型的生产成本与基础模型相同。这降低了微调的经济门槛——不再需要在"微调带来的质量提升"和"微调带来的成本增加"之间权衡。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 定制芯片 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——针对解码优化 |
| FireAttention | "Fireworks 引擎" | 自定义注意力内核；号称比 vLLM 低 4x 延迟 |
| Truss | "Baseten 格式" | 模型打包清单；依赖+密钥+服务配置 |
| 每词元 | "API 定价" | 按消费的词元收费；不为空闲付费 |
| 每分钟 | "专用定价" | 按墙钟时间收费；高利用率时胜出 |
| 批处理层 | "半价" | 非交互式队列以折扣价运行 |

---

## 📚 小结

2026 年的推理平台市场分叉为三个细分：定制芯片（5-10x 解码速度）、GPU 平台（Fireworks、Baseten、Modal 等）、API 市场（Replicate 等）。定价模式不统一——每词元、每分钟、每秒、每预测——必须按工作负载建模才能真正比较。30% 持续利用率是每分钟 vs 每词元的经验分水岭。FinOps 实践：从按需起，测量利用率，优化定价模型。

下一课我们将讨论 GPU 在 Kubernetes 上的自动扩缩容——Karpenter、KAI Scheduler 和三层扩缩容架构。

---

## ✏️ 练习

1. 运行 `code/main.py`。在什么持续利用率下 Baseten（每分钟）击败 Fireworks（每词元）？推导交叉点。
2. 你的产品需要图像生成 + 聊天 + 语音转文本。为每种模态选择一个平台，并命名统一的网关模式。
3. Fireworks 将你的主要模型涨价 $1/小时。建模对综合成本的影，假设 40% 的流量移到批处理层（半价）。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 推理平台对比器 | `code/main.py` | 六家供应商的成本对比 |
| 平台选型建议 | `outputs/skill-inference-platform-picker.md` | 根据工作负载推荐推理平台 |

---

## 📖 参考资料

1. [定价] Fireworks Pricing. https://fireworks.ai/pricing
2. [定价] Baseten Pricing. https://www.baseten.co/pricing/
3. [定价] Modal Pricing. https://modal.com/pricing
4. [报告] Infrabase — AI Inference API Providers 2026. https://infrabase.ai/blog/ai-inference-api-providers-compared
