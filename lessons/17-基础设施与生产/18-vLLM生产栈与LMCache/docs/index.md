# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的 production-stack 是参考的 Kubernetes 部署——路由器、引擎和可观测性连接在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来，在查询和引擎之间复用（CPU DRAM，然后磁盘/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）使这个异步且通过 Connector API 可插拔。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV 槽位用尽时，被抢占的请求可以从 CPU 恢复而不是重新 prefill。即使 KV 缓存超过 HBM，原生 CPU 卸载和 LMCache 都显著提升了吞吐量。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 内部原理）、阶段 17 · 06（SGLang/RadixAttention）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 画出 vLLM production-stack 层：路由器、引擎、KV 卸载、可观测性
- [ ] 解释 KV Offloading Connector API（v0.9.0+）和 0.11.0 异步路径如何隐藏卸载延迟
- [ ] 量化 LMCache CPU-DRAM 何时有用（KV > HBM）vs 何时增加开销（KV 足够小可以装入 HBM）
- [ ] 在原生 vLLM CPU 卸载和 LMCache connector 之间根据部署约束选择

---

## 1. 问题

你的 vLLM 在线服务显示 GPU 的 HBM 100% 使用，且并发升高时有抢占事件。请求被驱逐、重新排队，你在一分钟内对同一个 2K 词元提示词重新 prefill 了四次。GPU 计算花在了冗余的 prefill 上；goodput 远低于原始吞吐量。

增加更多 GPU 成本线性增长。增加更多 HBM 不可能。但 CPU DRAM 很便宜——一个插槽有 512GB+，延迟比 HBM 差几个数量级但对"暂时温热"的 KV 缓存足够。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，且跨引擎的重复前缀共享缓存而不需要每个引擎重新 prefill。

---

## 2. 概念

### 2.1 vLLM production-stack

`github.com/vllm-project/production-stack` 是参考的 Kubernetes 部署：

- **路由器——** 缓存感知（第 17 · 11 课）。消费 KV 事件
- **引擎——** vLLM worker。每个 GPU 或 TP/PP 组一个
- **KV 缓存卸载——** LMCache 部署或原生 connector
- **可观测性——** Prometheus 采集、Grafana 仪表盘、OTel 追踪
- **控制平面——** 服务发现、配置、滚动更新

作为 Helm chart + operator 发布。

### 2.2 KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的 Connector API。引擎将块卸载到 connector；connector 存储它们（RAM、磁盘、对象存储、LMCache）。请求需要块时，connector 将其加载回来。

vLLM 0.11.0（2026 年 1 月）添加了异步卸载路径——卸载可以在后台进行，引擎在常见情况下不阻塞。

### 2.3 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载：** 引擎本地。将 KV 块存储在主机 RAM 中。实现快，零网络跳转。不跨引擎。

**LMCache connector：** 集群级。将块存储在共享 LMCache 服务器（CPU DRAM + Ceph/S3 层）。任何引擎都可访问这些块。

单引擎有 HBM 压力时选原生。多引擎共享前缀（RAG 共享系统提示词、多租户共享模板）时选 LMCache。

### 2.4 基准行为

16x H100（80GB HBM）分散在 4 个 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示、低并发）：所有配置匹配基线，LMCache 增加约 3-5% 开销
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提升吞吐量；LMCache 因跨引擎共享增益更大

### 2.5 什么时候 LMCache 是决定性的

- 多租户服务中系统提示词跨租户共享
- RAG 中文档块跨查询重复
- 同一基础模型上的微调变体（LoRA），基础模型 KV 复用减少冗余工作
- 抢占密集型工作负载：从 CPU 恢复比重新 prefill 便宜

### 2.6 什么时候不该启用

- HBM 压力小——你付出了开销但没有收益
- 短上下文（<1K 词元）——传输时间 > 重新 prefill
- 单租户单提示词工作负载——没有可捕获的复用

---

## 3. 从零实现

### 第 1 步：KV 溢出模拟器

```python
def kv_spill_simulation(hbm_gb, kv_per_request_gb, requests_per_sec, duration_s):
    """模拟 KV 缓存溢出和 LMCache 恢复。"""
    total_kv = kv_per_request_gb * min(requests_per_sec * duration_s, 1000)
    overflow = max(0, total_kv - hbm_gb)

    # 无卸载：抢占后的请求需要重新 prefill
    reprefill_cost_s = overflow * 2  # 重新 prefill 的时间

    # 有 LMCache 卸载：从 CPU 恢复
    lmcache_restore_s = overflow * 0.1  # CPU DRAM 恢复很快

    return {
        "total_kv_gb": total_kv,
        "overflow_gb": overflow,
        "reprefill_time_s": reprefill_cost_s,
        "lmcache_restore_time_s": lmcache_restore_s,
        "speedup": reprefill_cost_s / max(lmcache_restore_s, 0.01),
    }


r = kv_spill_simulation(80, 0.5, 50, 10)
print(f"总 KV: {r['total_kv_gb']:.0f}GB  溢出: {r['overflow_gb']:.0f}GB")
print(f"重新 prefill 时间: {r['reprefill_time_s']:.0f}s")
print(f"LMCache 恢复时间: {r['lmcache_restore_time_s']:.1f}s")
print(f"加速比: {r['speedup']:.1f}x")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 production-stack 架构

```
┌─────────────┐
│   控制平面   │  Helm chart + operator
├─────────────┤
│   路由器    │  缓存感知，消费 KV 事件
├─────────────┤
│   引擎      │  vLLM worker（每 GPU/TP/PP 组）
├─────────────┤
│  KV 卸载    │  LMCache 或原生 CPU connector
├─────────────┤
│  可观测性    │  Prometheus + Grafana + OTel
└─────────────┘
```

### 4.2 选择对照

| 维度 | 原生 CPU 卸载 | LMCache |
|---|---|---|
| 范围 | 引擎本地 | 集群级 |
| 网络跳转 | 零 | 有（共享服务器） |
| 跨引擎复用 | 不支持 | 支持 |
| 实现复杂度 | 低 | 中 |
| 最佳场景 | 单引擎 HBM 压力 | 多引擎共享前缀 |

---

## 5. 工程最佳实践

### 5.1 KV 卸载不是默认开启的

只有当 KV 缓存占用超过 HBM 或有跨引擎前缀复用时才值得。低占用场景下 LMCache 增加 3-5% 开销。

### 5.2 监控抢占率

抢占是性能下降的信号——被抢占的请求需要重新 prefill。监控抢占率；如果 > 10%，考虑 KV 卸载或扩大 HBM。

### 5.3 中文场景特别建议

- **LMCache 国内部署。** LMCache 是开源项目，可以自托管。但需要运维共享 KV 缓存服务器——在小型部署中可能过于复杂
- **vLLM production-stack 国内可用性。** vLLM production-stack 依赖 Kubernetes。国内云厂商（阿里云 ACK、华为云 CCE）完全支持
- **中文提示词的 KV 缓存特点。** 中文提示词通常更长（同等语义信息），KV 缓存占用更高。KV 卸载对中文 RAG 场景更有价值

---

## 6. 常见错误

### 错误 1：KV 占用低时启用 LMCache

**现象：** 启用 LMCache 后吞吐量下降了 3-5%。

**原因：** KV 缓存足够小，可以完全装入 HBM。LMCache 的网络开销和 CPU 开销没有被收益抵消。

**修复：** 监控 HBM 利用率。只有在 KV 缓存 > 80% HBM 时才启用 LMCache。

### 错误 2：单租户场景使用 LMCache

**现象：** LMCache 服务器部署了但没有跨引擎复用。

**原因：** 单租户只有一个提示词模式——没有共享前缀。LMCache 的收益是跨引擎复用。

**修复：** 单租户场景用原生 CPU 卸载（零网络开销）。LMCache 只在多租户/RAG 场景下启用。

---

## 7. 面试考点

### Q1：LMCache 在 KV 缓存足够小可以装入 HBM 时还有价值吗？（难度：⭐⭐⭐）

**参考答案：**
仍然有价值——当有抢占时。即使 KV 缓存可以装入 HBM，并发高峰时 HBM 可能被填满导致抢占。被抢占的请求如果没有 KV 卸载需要完全重新 prefill（数百毫秒到数秒）。LMCache 允许从 CPU DRAM 恢复（数十毫秒），避免重新 prefill。基准测试显示，当 KV 缓存足够小可以装入 HBM 时，LMCache 增加约 3-5% 开销但提供抢占恢复能力。

### Q2：vLLM production-stack 的五个组件是什么？（难度：⭐⭐）

**参考答案：**
五个组件：路由器（缓存感知，消费 KV 事件）、引擎（vLLM worker，每 GPU/TP/PP 组一个）、KV 缓存卸载（LMCache 或原生 CPU connector）、可观测性（Prometheus + Grafana + OTel 追踪）、控制平面（服务发现、配置、滚动更新）。作为 Helm chart + operator 发布。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载 | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | 跨引擎 KV 缓存服务器（CPU DRAM + 磁盘） |
| 0.11.0 异步 | "非阻塞卸载" | 卸载隐藏在引擎流之后 |
| 抢占 | "驱逐以腾出空间" | HBM 满时的 KV 缓存洗牌 |
| 前缀复用 | "相同系统提示词" | 多个查询共享开头；缓存命中 |

---

## 📚 小结

vLLM production-stack 是参考的 Kubernetes 部署——路由器、引擎、KV 卸载、可观测性连接在一起。LMCache 是跨引擎的 KV 缓存服务器，将 KV 缓存从 GPU 提取到 CPU DRAM。即使 KV 缓存足够装入 HBM，抢占时 LMCache 也有价值——从 CPU 恢复比重新 prefill 快得多。低占用时 LMCache 增加 3-5% 开销——只在 HBM 压力大或有跨引擎前缀复用时启用。KV Offloading Connector（v0.9.0+）使 KV 卸载可插拔；v0.11.0 的异步路径隐藏了卸载延迟。

---

## ✏️ 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始盈利？
2. 一个租户每小时 200 次查询共享一个 6K 词元的系统提示词。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| KV 溢出模拟器 | `code/main.py` | 抢占负载下的 LMCache 收益模拟 |
| vLLM 栈选型器 | `outputs/skill-vllm-stack-decider.md` | 原生 vs LMCache vs 不启用的决策 |

---

## 📖 参考资料

1. [博客] vLLM — KV Offloading Connector (Jan 2026). https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
2. [GitHub] vLLM Production Stack. https://github.com/vllm-project/production-stack
3. [论文] LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665). https://arxiv.org/html/2510.09665v2
4. [GitHub] LMCache. https://github.com/LMCache/LMCache
