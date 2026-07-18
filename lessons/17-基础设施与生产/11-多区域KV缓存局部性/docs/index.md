# 多区域 LLM 服务与 KV 缓存局部性

> Round-robin 负载均衡对缓存推理是主动有害的。不落在持有其前缀的节点上的请求会支付全额 prefill 成本——长提示下 P50 约 800ms 对比缓存命中约 80ms。2026 年的生产模式是缓存感知路由器（vLLM Router、llm-d router），消费 KV 缓存事件并在前缀哈希匹配上路由。商业"跨区域推理"服务（Bedrock CRI、GKE Multi-Cluster Gateway）将推理视为不透明——它们处理可用性，不处理 TTFT。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 在线服务）、阶段 17 · 06（SGLang RadixAttention）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么 round-robin 负载均衡破坏缓存推理，并量化 TTFT 惩罚
- [ ] 画出缓存感知路由器：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）
- [ ] 命名 LLM 灾难恢复的 32% 失败原因（缺少分词器文件或量化配置），并列出三文件 DR 检查清单
- [ ] 区分商业跨区域服务（Bedrock CRI、GKE Multi-Cluster Gateway）和 KV 感知路由

---

## 1. 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 做 round-robin。生产环境中的前缀缓存命中率跌到 8%。TTFT P50 翻了三倍。vLLM 日志显示每个请求都在支付全额 prefill 成本。

Round-robin 对无状态服务是最优的。LLM 推理在设计上是有状态的——KV 缓存编码了模型看到的一切。盲目路由就是路由到了错误的缓存。

另外，你的团队有一个 DR 计划。你把模型权重备份到了 S3 跨区域存储。区域宕机发生时，你尝试故障切换——副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE 缩放配置在另一个没有同步的存储桶里。

---

## 2. 概念

### 2.1 缓存感知路由

请求带着提示词到达。路由器哈希前缀（如前 512 词元）；它询问每个副本"你有这个前缀的缓存吗？"。副本在 pub/sub 频道上发布 KV 缓存事件（分配和驱逐块时）。路由器选择有匹配的副本；如果没有匹配，退回到 GPU 利用率最低的副本。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希→副本索引的映射，O(1) 查找路由。无匹配时退回到最少队列深度。

**llm-d router**：同样模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（第 17 · 06 课）是副本内的等效机制。跨副本路由严格在上游。

### 2.2 数字

2K 词元提示、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- 缓存命中（同副本，前缀驻留）：约 80ms
- 缓存未命中（冷 prefill）：约 800ms

10 倍差距。如果你的路由器在跨副本上达到 60-80% 的前缀缓存命中率，你可以在 N 副本容量下逼近单副本性能。如果只命中 10%，你就在逼近朴素扩展。

### 2.3 跨区域有新约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65ms
- us-east-1 ↔ eu-west-1：约 75ms
- us-east-1 ↔ ap-southeast-1：约 220ms

如果路由器将请求从 us-east-1 路由到 ap-southeast-1 上的热前缀，节省的 prefill（800→80ms）被 440ms 往返延迟吞没。GORGO（2026 研究）将这个显式化——最小化 `prefill_time + network_latency`，而不是仅 prefill。通常答案是保持区域路由，除非前缀极大（数 MB）时 prefill 主导。

### 2.4 商业"跨区域推理"不解决这个问题

AWS Bedrock 跨区域推理在容量压力时自动将请求路由到其他区域。它优化的是可用性，不是 TTFT，且将推理视为不透明。GKE Multi-Cluster Gateway 同理——服务级故障切换，没有 KV 缓存感知。

你仍然需要应用层的缓存感知路由器。它们处理"us-east-1 挂了"的情况。缓存感知路由处理 TTFT 的情况。

### 2.5 DR 卫生——32% 缺文件问题

2026 年广泛引用的数据：32% 的 LLM DR 失败是因为团队备份了权重但忘了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ 缩放因子、GPTQ 零点）
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

修复方案：三文件最小 DR 清单：
1. HF 模型仓库中的所有文件（权重 + 配置 + 分词器）
2. 引擎特定的服务配置
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）

加上：每季度做一次 DR 演练。JPMorgan 的 us-east-1 演练在 2024 年 11 月达到 22 分钟恢复，只因为剧本被排练过。

### 2.6 数据驻留是正交的

EU 客户的 PHI 不能离开 EU。如果你的缓存感知路由器将巴黎来源的请求路由到 us-east-1 上的前缀匹配，你已经违反了 GDPR——不管 TTFT 有多大收益。在优化缓存之前先按驻留边界分区路由器。

---

## 3. 从零实现

### 第 1 步：缓存感知路由器

```python
class CacheAwareRouter:
    """简化版缓存感知路由器。"""

    def __init__(self):
        self.replica_cache = {}  # replica_id -> set of prefix_hashes

    def update_cache(self, replica_id: str, prefix_hash: str, has_cached: bool):
        """副本通知路由器缓存变化。"""
        if replica_id not in self.replica_cache:
            self.replica_cache[replica_id] = set()
        if has_cached:
            self.replica_cache[replica_id].add(prefix_hash)
        else:
            self.replica_cache[replica_id].discard(prefix_hash)

    def route(self, prefix_hash: str) -> str:
        """路由到持有该前缀的副本。"""
        candidates = [r for r, hashes in self.replica_cache.items()
                      if prefix_hash in hashes]
        if candidates:
            return candidates[0]
        # 退回到默认策略（最空闲的副本）
        return list(self.replica_cache.keys())[0]

    def hit_rate(self) -> float:
        """缓存命中率。"""
        total = sum(len(h) for h in self.replica_cache.values())
        unique = set()
        for hashes in self.replica_cache.values():
            unique.update(hashes)
        if not unique:
            return 0.0
        return total / (len(self.replica_cache) * max(len(unique), 1))


# 演示
router = CacheAwareRouter()
router.update_cache("replica-1", "hash_A", True)
router.update_cache("replica-1", "hash_B", True)
router.update_cache("replica-2", "hash_C", True)

print(f"路由 hash_A → {router.route('hash_A')}")
print(f"路由 hash_B → {router.route('hash_B')}")
print(f"路由 hash_C → {router.route('hash_C')}")
print(f"路由 hash_D → {router.route('hash_D')} (回退)")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 缓存感知路由架构

| 组件 | 工具 | 功能 |
|---|---|---|
| KV 缓存事件 | vLLM Router (Rust) | 订阅块分配/驱逐事件 |
| 前缀哈希查找 | vLLM Router / llm-d | O(1) 缓存命中路由 |
| 跨区域路由 | GORGO 研究 | 最小化 prefill + 网络延迟 |
| 故障切换 | Bedrock CRI / GKE | 可用性 failover（无 KV 感知） |
| DR 备份 | 三文件清单 | 权重 + 配置 + 部署清单 |

---

## 5. 工程最佳实践

### 5.1 永远不要对 LLM 使用 round-robin

LLM 推理是有状态的——KV 缓存编码了上下文。Round-robin 路由到错误的节点会支付全额 prefill 成本。使用缓存感知路由。

### 5.2 DR 清单不止权重

32% 的 LLM DR 失败是因为忘了 tokenizer/量化配置。备份时用三文件清单：所有 HF 文件 + 引擎配置 + 部署清单。每季度演练。

### 5.3 数据驻留优先于缓存优化

如果你的用户在 EU，不要为了缓存命中率将请求路由到 US——这违反 GDPR。先按驻留边界分区，再在每个分区内优化缓存。

### 5.4 中文场景特别建议

- **中国境内的延迟更低。** 阿里云多区域（杭州-北京-上海）之间的 RTT 约 20-40ms，比 AWS 美国跨区域低得多。缓存感知路由的收益更明显
- **跨境延迟很高。** 中国到 AWS 美国的 RTT 约 150-250ms。跨境路由时 prefill 节省可能被网络延迟吞没
- **国内 DR 方案。** 国内云厂商的跨区域备份服务（阿里云跨地域复制、华为云跨区域容灾）可以作为 DR 基础。但 DR 清单仍然需要包含 tokenizer 和量化配置

---

## 6. 常见错误

### 错误 1：用 round-robin 路由 LLM 请求

**现象：** 前缀缓存命中率跌到 8%，TTFT P50 翻了三倍。

**原因：** Round-robin 无视前缀局部性——每个请求随机落到不同副本。

**修复：** 使用缓存感知路由——路由器根据前缀哈希将请求路由到持有该前缀的副本。

### 错误 2：DR 备份只备份权重

**现象：** 区域宕机后尝试故障切换，副本启动失败。检查发现缺少 tokenizer.json。

**原因：** 备份只包含模型权重，没有包含分词器、量化配置、引擎配置。

**修复：** 用三文件清单备份：所有 HF 文件 + 引擎配置 + 部署清单。

---

## 7. 面试考点

### Q1：为什么 round-robin 对 LLM 推理是有害的？（难度：⭐⭐）

**参考答案：**
LLM 推理是有状态的——KV 缓存编码了模型已经看到的上下文。如果请求 A 在副本 1 上缓存了前缀，请求 B 也在同一前缀上，round-robin 可能将 B 路由到副本 2——副本 2 没有这个前缀的缓存，B 需要全额 prefill。对于 2K 词元的提示，缓存命中约 80ms，未命中约 800ms——10 倍差距。round-robin 的缓存命中率约 1/N（N 个副本），而缓存感知路由可以达到 60-80%。

### Q2：32% 的 LLM DR 失败原因是什么？（难度：⭐⭐）

**参考答案：**
团队备份了模型权重但忘了 tokenizer.json、量化配置（AWQ 缩放因子、GPTQ 零点）、模型特定配置（RoPE 缩放、聊天模板）和引擎配置（vllm_config.yaml）。修复方案是三文件最小 DR 清单：所有 HF 模型文件 + 引擎配置 + 部署清单。每季度做一次 DR 演练。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 缓存感知路由 | "智能 LB" | 基于前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV 缓存事件 | "缓存 pub/sub" | 副本发布块分配/驱逐事件；路由器建立索引 |
| 前缀哈希 | "缓存键" | 前 N 个词元的哈希，用于路由器查找 |
| GORGO | "跨区域路由研究" | 将网络延迟显式纳入路由目标的 2026 研究 |
| DR 清单 | "备份列表" | 恢复所需的所有文件——不只是权重 |
| 数据驻留 | "GDPR 边界" | 用户数据能被哪个区域看到的法律约束 |

---

## 📚 小结

多区域 LLM 服务有三个问题：缓存命中率（round-robin 破坏）、灾难恢复（32% 因缺失文件失败）、数据驻留（跨区域路由可能违反 GDPR）。缓存感知路由器通过前缀哈希匹配将请求路由到持有该前缀的副本——TTFT 从 800ms 降到 80ms。DR 需要三文件清单（不只是权重）加季度演练。先按数据驻留分区，再在每个分区内优化缓存。

---

## ✏️ 练习

1. 运行 `code/main.py`。在什么提示长度下，跨区域路由优于本地路由？给定 75ms RTT。
2. 你的缓存命中率从 70% 降到 12%。诊断三个可能的原因和确认每个的可观测指标。
3. 设计一个 70B AWQ 量化模型 + 5 个 LoRA 适配器的 DR 清单。列出每个文件。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 缓存感知路由器 | `code/main.py` | 模拟前缀哈希路由 |
| 多区域路由器设计 | `outputs/skill-multi-region-router.md` | 根据区域和 SLA 设计路由方案 |

---

## 📖 参考资料

1. [论文] GORGO (arXiv:2602.11688) — 跨区域 KV 缓存复用
2. [官方文档] AWS Bedrock Cross-Region Inference. https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
3. [GitHub] vLLM Production Stack Router. https://github.com/vllm-project/production-stack
