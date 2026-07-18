# 分离式 Prefill/Decode——NVIDIA Dynamo 和 llm-d

> Prefill 是计算密集型的；decode 是内存带宽密集型的。在同一个 GPU 上运行两者会浪费一种资源。分离式将它们拆分到独立池并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）传输 KV 缓存。NVIDIA Dynamo 坐在 vLLM/SGLang/TRT-LLM 之上——它的 Planner Profiler + SLA Planner 自动匹配 prefill:decode 比例以满足 SLO。2026 年的经济学：切换到分离式服务可以在相同 SLA 下节省约 30-40% 的推理支出。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 内部原理）、阶段 17 · 08（推理指标）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么 prefill 和 decode 有不同的最优 GPU 分配，以及共置下的浪费量化
- [ ] 画出分离式架构：prefill 池、decode 池、KV 传输（NIXL）、路由器
- [ ] 命名分离式不值得的场景（短提示、短输出）
- [ ] 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并将每个匹配到运维场景

---

## 1. 问题

你在 8 张 H100 上运行 Llama 3.3 70B。混合负载（长提示+短输出）下 GPU 在 decode 期间空闲，因为大部分计算花在了 prefill 上。不同负载（短提示+长输出）下情况相反。共置 prefill+decode 意味着你对两者都过度配置。

预算影响：20-40% 的 GPU 时间被浪费在错误的资源上。你在买 H100 计算力来运行内存带宽密集型的 decode，或者买 H100 HBM 带宽来运行计算密集型的 prefill。两者都是昂贵的浪费。

分离式将 prefill 和 decode 拆分到针对各自瓶颈优化的独立池。KV 缓存通过高带宽互联从 prefill 池传输到 decode 池。

---

## 2. 概念

### 2.1 为什么瓶颈不同

**Prefill——** 在一次前向传播中运行整个输入提示词。矩阵乘法主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 有用吞吐。

**Decode——** 每次迭代生成一个词元，读取完整权重。内存带宽密集型。HBM3 提供约 3TB/s。

共置两者：你购买同时优化两者的 GPU。H100 两者都不错但成本相同。规模扩大时，prefill 池用 H100/计算密集型；decode 池用 H200/内存密集型，或用激进量化。

### 2.2 架构

```
            ┌──────────────┐
  请求 →   │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │ (仅提示词)                      ▼
            ┌──────────────┐    KV 缓存    ┌───────▼──────┐
            │ Prefill 池  │ ─── NIXL ───► │ Decode 池   │
            │  (计算)      │               │  (内存)      │
            └──────────────┘               └──────┬───────┘
                                                   │ 词元
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输。使用 RDMA/InfiniBand（如果可用），TCP 回退。传输延迟真实存在——70B FP8 上 4K 提示词的 KV 缓存通常 20-80ms。这就是短提示不值得分离式的原因：传输税超过了节省。

### 2.3 Dynamo vs llm-d

**NVIDIA Dynamo（GTC 2025 发布，1.0 GA）：**
- 坐在 vLLM、SGLang、TRT-LLM 之上作为编排器
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例
- Rust 核心，Python 扩展
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 中等延迟场景下约 6 倍提升
- GB300 NVL72 + Dynamo：对比 Hopper 最高 50 倍 MoE 吞吐量

**llm-d（Red Hat + AWS，Kubernetes 原生）：**
- Prefill / decode / 路由器作为独立 Kubernetes Service
- 每角色 HPA，使用队列深度（prefill）/ KV 利用率（decode）信号
- `topologyConstraint packDomain: rack` 将 prefill+decode 集群打包到同一机架以实现高带宽 KV 传输
- llm-d 0.5（2026）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩到零

### 2.4 什么时候不该分离

- 提示词 < 512 词元 + 输出 < 200 词元：传输税主导收益
- 小集群（< 4 GPU）：池多样性不足
- 团队无法运维两个带逐角色扩缩的 GPU 池
- 没有 RDMA 网络：TCP 传输税更重

### 2.5 经济学

内部综合数据（非单个发布案例，量级锚点）：

- $2M/年共置服务推理支出
- 切换到 Dynamo 分离式
- 相同请求量、相同 P99 延迟 SLA
- 报告节省：$600K-$800K/年（30-40% 降低）
- 无需新硬件

节省来自正确调整每个池的规模——prefill 密集型工作负载（RAG 带 8K+ 前缀）比平衡型工作负载受益更多。

### 2.6 MoE 在 Blackwell 上才是真正的大数字

GB300 NVL72 + Dynamo 对比 Hopper 基线显示 50 倍 MoE 吞吐量。MoE 专家路由在 prefill 上计算密集但在 decode 上内存密集（专家缓存）——分离式是双赢。2026 年前沿模型服务以 MoE 为主导（DeepSeek-V3、未来 GPT-5 变体）。

---

## 3. 从零实现

### 第 1 步：共置 vs 分离式吞吐量对比

```python
def throughput_colocated(prefill_tokens, decode_tokens, gpu_flops, gpu_bandwidth):
    """计算共置模式下的吞吐量。"""
    prefill_time = prefill_tokens * 1e9 / gpu_flops  # 简化
    decode_time = decode_tokens * 1e9 / gpu_bandwidth
    return 1 / (prefill_time + decode_time)


def throughput_disaggregated(prefill_tokens, decode_tokens,
                             prefill_flops, decode_bandwidth,
                             kv_transfer_ms=50):
    """计算分离式模式下的吞吐量。"""
    prefill_time = prefill_tokens * 1e9 / prefill_flops
    decode_time = decode_tokens * 1e9 / decode_bandwidth
    total = prefill_time + decode_time + kv_transfer_ms / 1000
    return 1 / total


# 对比
print("=== 吞吐量对比 ===")
for pt in [512, 2048, 8192]:
    dt = 200
    coloc = throughput_colocated(pt, dt, 2000, 3)
    disag = throughput_disaggregated(pt, dt, 2000, 3, 50)
    ratio = disag / coloc
    print(f"提示词={pt:5d}  共置={coloc:.2f}  分离={disag:.2f}  比率={ratio:.2f}x")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Dynamo vs llm-d 对照

| 维度 | NVIDIA Dynamo | llm-d |
|---|---|---|
| 部署模型 | 栈上层编排器 | Kubernetes 原生 Service |
| 语言 | Rust + Python | Kubernetes + Go |
| 自动扩缩 | Planner Profiler + SLA Planner | Per-role HPA（队列深度/KV 利用率） |
| KV 传输 | NIXL（RDMA/TCP） | NIXL + UCCL |
| 最佳场景 | 想要托管编排栈 | 想要 K8s 原生原语 |

---

## 5. 工程最佳实践

### 5.1 先测量再分离

短提示（< 512 词元）+ 短输出（< 200 词元）的工作负载不需要分离——传输税超过了收益。先测量你的工作负载分布。

### 5.2 预分配 + decode 池的扩缩信号不同

prefill 池扩缩信号：队列深度。decode 池扩缩信号：KV 缓存利用率。不要用同一个 HPA 覆盖两者。

### 5.3 中文场景特别建议

- **NIXL 在国内的网络环境。** RDMA/InfiniBand 在国内数据中心的部署成本较高。TCP 回退的传输延迟约 50-100ms——对于短提示可能不值得分离式
- **Dynamo 国内可用性。** NVIDIA Dynamo 是开源项目，可以在国内 GPU 实例上部署。但需要 CUDA 12.0+ 和 NVLink 支持
- **中文提示词的分离式收益。** 中文提示词通常更长（同等语义信息需要更多词元），分离式的 prefill 池收益可能比英文更明显

---

## 6. 常见错误

### 错误 1：对短提示使用分离式

**现象：** 分离式部署后吞吐量没有提升，反而因为 KV 传输延迟增加了。

**原因：** 提示词只有 200 词元，prefill 只需约 50ms。KV 传输需要 50ms——收益为零。

**修复：** 测量提示词长度分布。如果 P50 < 512 词元，分离式不值得。

### 错误 2：prefill 和 decode 用同一个 HPA

**现象：** prefill 池和 decode 池的扩缩不同步——prefill 池扩了但 decode 池没扩，KV 传输成为瓶颈。

**原因：** 两者用同一个 HPA——扩缩信号冲突。

**修复：** prefill 池用队列深度扩缩，decode 池用 KV 利用率扩缩。独立 HPA。

---

## 7. 面试考点

### Q1：分离式为什么能节省 30-40% 的推理支出？（难度：⭐⭐⭐）

**参考答案：**
共置模式下 GPU 资源被浪费——prefill 用计算密集型 GPU 但 decode 需要内存带宽，decode 期间 GPU 计算空闲。分离式将 prefill 和 decode 拆分到针对各自瓶颈优化的独立池——prefill 池用计算密集型 GPU，decode 池用内存密集型 GPU 或激进量化。每个池的利用率从约 60% 提升到约 90%——总 GPU 时间减少约 30-40%，相同 SLA。

### Q2：Dynamo 和 llm-d 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
Dynamo 是栈上层编排器——坐在 vLLM/SGLang/TRT-LLM 之上，用 Rust 核心+Python 扩展自动配置 prefill:decode 比例。适合想要托管编排栈的团队。llm-d 是 Kubernetes 原生的——prefill/decode/路由器作为独立 K8s Service，使用原生 HPA 和拓扑约束。适合已经深度投入 K8s/CNCF 生态的团队。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 分离式服务 | "拆分 prefill/decode" | 为每个阶段使用独立 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | 坐在 vLLM/SGLang/TRT-LLM 之上的协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS 的 K8s 分离式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| MoE 专家路由 | "每词元一个专家" | DeepSeek-V3 模式；分离式是双赢 |

---

## 📚 小结

Prefill（计算密集型）和 decode（内存带宽密集型）在同一 GPU 上运行会浪费资源。分离式将两者拆分到独立池——prefill 池优化计算，decode 池优化带宽。NIXL 传输 KV 缓存（20-80ms）。Dynamo（栈上层）和 llm-d（Kubernetes 原生）是两种实现方式。分离式在提示词 >512 词元+输出 >200 词元时才值得——短提示的传输税超过了节省。MoE 模型在分离式中是双赢——prefill 计算密集，decode 内存密集。

---

## ✏️ 练习

1. 运行 `code/main.py`。在什么提示词长度下分离式优于共置？
2. 设计一个 RAG 服务的 prefill 池和 decode 池——P99 前缀长度 8K，输出 300。
3. 计算 KV 传输成本：4K prefill 在 70B FP8 上约 500MB KV。RDMA 100GB/s 传输 = 5ms。TCP 10GB/s = 50ms。哪个对你的 SLA 重要？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 分离式对比器 | `code/main.py` | 共置 vs 分离式的吞吐量和成本对比 |
| 分离式决策器 | `outputs/skill-disaggregation-decider.md` | 根据工作负载和集群判断是否分离 |

---

## 📖 参考资料

1. [NVIDIA] Introducing Dynamo. https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
2. [NVIDIA] Disaggregated LLM Inference on Kubernetes. https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/
3. [GitHub] llm-d. https://github.com/llm-d/llm-d
4. [官方文档] TensorRT-LLM Disaggregated Serving. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html
