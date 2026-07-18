# GPU 自动扩缩容与 Kubernetes——Karpenter、KAI Scheduler、Gang Scheduling

> 三层扩缩容，不是一层。Karpenter 动态拉起节点（<1 分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理组调度和拓扑感知——它防止 7-of-8 部分分配陷阱。应用层扩缩器（Dynamo Planner、llm-d）使用推理特定的信号——队列深度、KV 缓存利用率——而不是 CPU/DCGM 占空比。经典的 HPA 陷阱：`DCGM_FI_DEV_GPU_UTIL` 是占空比测量——100% 可能对应 10 个请求或 100 个请求。本课教你组合三层扩缩容，避开默认的 Karpenter 合并策略——它会在推理中间终止正在运行的 GPU 任务。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 02（推理平台经济学）、阶段 17 · 04（vLLM 在线服务内部原理）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 画出三层扩缩容架构（节点供应层、组调度层、应用层）并说出每层使用的工具
- [ ] 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA 信号，并说出两个替代信号
- [ ] 描述组调度（Gang Scheduling）和 KAI Scheduler 防止的"7 卡等 1 卡"部分分配陷阱
- [ ] 命名 Karpenter 中会终止正在运行的 GPU 作业的合并策略（`WhenEmptyOrUnderutilized`），并说出 2026 年的安全替代方案

---

## 1. 问题

你的团队在 Kubernetes 上部署了一个 LLM 在线服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。服务在营业时间保持 100% 利用率。HPA 从不扩缩——它认为你已经饱和了。你手动添加一个副本；TTFT 下降。HPA 仍然不扩缩。信号在欺骗你。

另外，你用 Cluster Autoscaler 管理节点。凌晨 2 点收到一个 1M 词元的提示词；集群花了 3 分钟拉节点，请求超时。

另外，你部署了一个需要 8 个 GPU 的 70B 模型。集群有 7 个空闲 GPU，第 8 个分布在 3 个节点上。Cluster Autoscaler 为缺失的 1 个 GPU 拉了一个节点。7 台机器等待了 4 分钟，耗着电费，等 Kubernetes 把最后一块 GPU 拉起来。

三个故障模式，三个不同的层。2026 年的 GPU 感知扩缩容不是"打开 HPA"——而是组合节点供应、组调度和应用层信号。

---

## 2. 概念

### 2.1 第 1 层——节点供应（Karpenter）

Karpenter 监视待处理的 Pod，在约 45-60 秒内拉起节点（Cluster Autoscaler 通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 块 H100 而集群没有匹配的节点，Karpenter 直接拉一个节点而不是扩展现有组。

**合并陷阱：** Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说很危险。它会终止正在运行的 GPU 节点，将 Pod 迁移到更便宜的实例。对于推理工作负载，这意味着驱逐正在运行的请求并在新节点上重新加载 70B 模型——损失是数分钟的容量加请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在真正空闲的节点上合并，但绝不驱逐正在运行的作业。

### 2.2 第 2 层——组调度（KAI Scheduler）

KAI Scheduler（项目名"Karp"后改名）处理默认 kube-scheduler 不做的事情：

**组调度（Gang Scheduling）——全有或全无。** 一个需要 8 块 GPU 的分布式推理 Pod 要么一起启动全部 8 个，要么一个都不启动。否则就会出现部分分配陷阱：7/8 的 Pod 启动了，无限等待，干耗钱。

**拓扑感知——** 知道哪些 GPU 共享 NVLink、哪些在同一个机架上、哪些之间有 InfiniBand。相应地放置 Pod。一个 DeepSeek-V3 67B 的张量并行工作负载必须保持在一个 NVLink 域内；KAI Scheduler 遵守这一点。

**层级队列——** 多个团队竞争同一个 GPU 池，带优先级和配额。团队 A 的生产型任务可以在优先级规则允许的情况下被团队 B 的训练作业抢占。

KAI 作为副调度器与 kube-scheduler 一起部署；你通过注解标注工作负载以使用它。Ray 和 vLLM 生产栈都已集成。

### 2.3 第 3 层——应用层信号

**HPA 陷阱：** `DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它测量 GPU 在每个采样间隔是否在做计算。100% 利用率可能意味着 10 个或 100 个并发请求——无论如何 GPU 都是忙的。基于占空比扩缩容是盲目扩缩。

更糟糕的是，vLLM 等引擎预分配 KV 缓存内存（到 `--gpu-memory-utilization` 限制）。即便只有一个请求，内存使用也保持在 90% 附近。基于内存的 HPA 永远不会缩容。

**2026 年的替代信号：**

- **队列深度**——等待 prefll 的请求数
- **KV 缓存利用率**——已分配给活跃序列的块的比例
- **每副本 P99 TTFT**——你的 SLA 信号
- **Goodput**——每秒满足所有 SLO 的请求数

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。它们在 LLM 在线服务中完全替代 HPA。

### 2.4 什么时候用什么

| 扩缩决策 | 工具 |
|---|---|
| 增删节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 增删副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 2.5 分离式 Prefill/Decode 让一切更复杂

如果你运行分离式 Prefill/Decode（第 17 · 17 课），你有两个 Pod 类别，各有不同的扩缩触发器：prefill Pod 基于队列深度扩缩，decode Pod 基于 KV 缓存压力扩缩。llm-d 将它们暴露为独立的 `Services`，每角色一个 HPA。不要尝试用一个 HPA 同时覆盖两者。

### 2.6 冷启动在这里也很重要

冷启动缓解（第 17 · 10 课）就是节点供应时间变得用户可见的地方。Karpenter 的 45-60 秒预热 + 20GB 模型加载 + 引擎初始化意味着从零到一的请求需要 2-5 分钟。为 SLO 关键的路径保持一个热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 2.7 你应该记住的数字

- Karpenter 节点供应：约 45-60s vs Cluster Autoscaler 约 90-120s（GPU 节点）
- KAI Scheduler 防止部分分配浪费——7-of-8 陷阱
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：有害；改用队列深度或 KV 利用率
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 作业。推理场景使用 `WhenEmpty + consolidateAfter: 1h`

---

## 3. 从零实现

### 第 1 步：三层扩缩容模拟器

```python
import random
import time
from collections import deque


class AutoscalerSimulator:
    """模拟三层扩缩容在突发 GPU 工作负载上的表现。"""

    def __init__(self, strategy: str):
        self.strategy = strategy  # "naive_hpa" | "queue_depth" | "gang"
        self.queue = deque()
        self.active_replicas = 3
        self.total_requests = 0
        self.dropped = 0
        self.idle_minutes = 0

    def tick(self, incoming_requests: int):
        """模拟一个时间步。"""
        # 入队
        for _ in range(incoming_requests):
            self.queue.append(1)

        # 扩缩决策
        if self.strategy == "queue_depth":
            # 基于队列深度扩缩
            desired = max(1, len(self.queue) // 2)
            self._scale_to(desired)
        elif self.strategy == "naive_hpa":
            # 基于"利用率"模拟的 HPA
            utilization = min(1.0, len(self.queue) / (self.active_replicas * 5))
            if utilization > 0.8:
                self._scale_to(self.active_replicas + 1)
            elif utilization < 0.3 and self.active_replicas > 1:
                self._scale_to(self.active_replicas - 1)

        # 处理请求
        capacity = self.active_replicas * 2
        served = 0
        while self.queue and served < capacity:
            self.queue.popleft()
            served += 1
            self.total_requests += 1

        self.dropped += max(0, len(self.queue) - 100)
        self.idle_minutes += max(0, self.active_replicas - served / 2)

    def _scale_to(self, n: int):
        self.active_replicas = max(1, min(n, 10))

    def report(self) -> dict:
        return {
            "strategy": self.strategy,
            "total": self.total_requests,
            "dropped": self.dropped,
            "idle_minutes": self.idle_minutes,
        }


# 演示：对比三种策略
def run_comparison():
    sims = [AutoscalerSimulator(s) for s in ["naive_hpa", "queue_depth", "gang"]]

    for step in range(100):
        incoming = 5 + int(20 * (0.5 + 0.5 * (step % 20 > 15)))  # 突峰
        for sim in sims:
            sim.tick(incoming)

    for sim in sims:
        r = sim.report()
        print(f"{r['strategy']:15s} 处理={r['total']} 丢弃={r['dropped']} "
              f"空闲分钟={r['idle_minutes']}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Karpenter GPU NodePool 配置

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-gpu-count
          operator: In
          values: ["8"]        # 只选择 8 GPU 实例
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
  disruption:
    consolidationPolicy: WhenEmpty    # 安全模式：不驱逐运行中作业
    consolidateAfter: 1h
```

### 4.2 三层扩缩容对照

| 层 | 工具 | 信号 | 时间尺度 |
|---|---|---|---|
| 节点供应 | Karpenter | 待处理 Pod 数 | ~45-60s |
| 组调度 | KAI Scheduler | GPU 拓扑/机架位置 | 调度时 |
| 应用层 | Dynamo Planner / llm-d | 队列深度 / KV 利用率 / TTFT | 秒级 |

---

## 5. 工程最佳实践

### 5.1 不要用 GPU 利用率做 HPA 信号

`DCGM_FI_DEV_GPU_UTIL` 是占空比——100% 意味着 GPU 在工作，但不代表需要更多副本。用**队列深度**做 prefill 扩缩，用 **KV 缓存利用率**做 decode 扩缩。

### 5.2 Karpenter 合并策略用 WhenEmpty

推理工作负载中永远不要用 `WhenEmptyOrUnderutilized`。它会终止正在处理的请求。`WhenEmpty + consolidateAfter: 1h` 是安全选择。

### 5.3 多 GPU 模型必须用 KAI Scheduler

任何需要 >1 块 GPU 的推理 Pod 都应该通过 KAI Scheduler 调度。否则你会遇到部分分配陷阱——7 块 GPU 空闲但第 8 块没到位，全部干耗。

### 5.4 保持热池

扩缩容从零开始的时间是 2-5 分钟（节点拉起 + 模型加载 + 引擎初始化）。为 SLO 关键路径保持 `min_workers=1`。

### 5.5 中文场景特别建议

- **国内 Karpenter 替代。** AWS 的 Karpenter 在 AWS 中国区域可用，但阿里云、腾讯云没有 Karpenter 等效工具。国内用户可能需要使用 Cluster Autoscaler + 节点组的手动配置
- **KAI Scheduler 国内集成。** KAI Scheduler 在华为云、阿里云等平台的 Kubernetes 服务上尚未官方支持。需要自行部署作为副调度器
- **国内 GPU 实例的拓扑结构。** 国内云厂商的 GPU 实例（阿里云 GN7、GNV4、华为云 P2s）的 NVLink 拓扑可能不同。在使用 KAI 调度拓扑感知前，需要先了解具体实例的互联结构

---

## 6. 常见错误

### 错误 1：用 GPU 利用率做 HPA 信号

**现象：** HPA 从不在高负载时扩缩——因为利用率已经 100% 了。你手动加了副本，但 HPA 又不缩容——因为缩放后利用率下降了但内存使用没变。

**原因：** `DCGM_FI_DEV_GPU_UTIL` 是占空比，不是负载信号。vLLM 的 KV 缓存预分配让内存始终接近 90%。

**修复：** 改用队列深度（prefill 端）和 KV 缓存利用率（decode 端）。

### 错误 2：Karpenter 合并策略导致推理中断

**现象：** 生产环境中看到周期性的 TTFT 尖峰，持续 2-3 分钟。日志显示 Pod 被驱逐了。

**原因：** Karpenter 的 `WhenEmptyOrUnderutilized` 发现了更便宜的实例类型，决定迁移 Pod——驱逐了正在处理的推理请求。

**修复：** 将 GPU NodePool 的 `consolidationPolicy` 设为 `WhenEmpty`。

### 错误 3：多 GPU 模型的部分分配

**现象：** 一个需要 8 块 GPU 的模型启动后卡在 Pending，日志显示 7/8 个容器在运行。

**原因：** Kube-scheduler 逐个调度 Pod 而不是原子调度。7 个 Pod 启动后等待第 8 个，但第 8 个所需的 GPU 尚未到位。

**修复：** 使用 KAI Scheduler 的组调度，确保所有 8 个 Pod 要么一起启动，要么一个都不启动。

---

## 7. 面试考点

### Q1：为什么 `DCGM_FI_DEV_GPU_UTIL` 不是正确的 HPA 信号？应该用什么替代？（难度：⭐⭐）

**参考答案：**
`DCGM_FI_DEV_GPU_UTIL` 是占空比——它测量 GPU 在采样间隔内是否在做计算，而不是在服务多少个请求。100% 利用率可能对应 10 个请求或 100 个请求。vLLM 引擎加剧了这个问题：它预分配 KV 缓存内存，所以内存基信号也永远不会触达缩容阈值。正确的替代信号有两个：prefill 端用**队列深度**（等待处理的请求数），decode 端用 **KV 缓存利用率**（已分配缓存块占总缓存块的比例）。

### Q2：Karpenter 的 `WhenEmptyOrUnderutilized` 对推理工作负载有什么危险？正确的配置是什么？（难度：⭐⭐）

**参考答案：**
`WhenEmptyOrUnderutilized` 会让 Karpenter 在检测到节点利用率低时（例如夜间流量低峰期）终止节点并迁移 Pod 到更小的实例。对于推理工作负载，这意味着驱逐正在处理的请求——用户看到超时。70B 模型加载到 GPU 需要 1-2 分钟，这段时间内容量为 0。正确的配置是 `WhenEmpty` + `consolidateAfter: 1h`——只合并真正空闲的节点，且等待至少 1 小时。

### Q3：什么是组调度？KAI Scheduler 防止的"部分分配陷阱"具体是什么？（难度：⭐⭐⭐）

**参考答案：**
组调度（Gang Scheduling）要求一组 Pod 要么全部同时调度，要么一个都不调度。部分分配陷阱是：一个需要 8 块 GPU 的推理 Pod 在调度时，kube-scheduler 逐个调度 Pod。如果集群有 7 块空闲 GPU，第 8 块分布在其他节点上，7 个 Pod 启动后等待一个永远不会来的第 8 个。7 台 GPU 机器全程空转，耗电、费钱，直到超时。KAI Scheduler 通过原子调度防止这个问题：只有所有 8 个 GPU 都就位时才启动任何 Pod。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Karpenter | "节点供应器" | Kubernetes 自动扩缩器；在约 45-60 秒内拉节点 |
| Cluster Autoscaler | "旧的扩缩器" | 之前的 Kubernetes 自动扩缩器；按组扩展，速度较慢 |
| KAI Scheduler | "GPU 调度器" | 用于组调度 + 拓扑 + 队列的副调度器 |
| 组调度 | "全有或全无" | 原子调度 N 个 Pod 或全部推迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架放置 Pod |
| 占空比 | "GPU 利用率" | 占空比指标；不是 LLM 的扩缩信号 |
| 队列深度 | "等待的请求数" | Prefill 端正确的 HPA 信号 |
| KV 缓存利用率 | "内存压力" | Decode 端正确的 HPA 信号 |
| 合并 | "Karpenter 合并" | 为更便宜的实例类型终止节点 |

---

## 📚 小结

GPU 感知扩缩容是三层的组合，不是单个 HPA 的事：节点层（Karpenter，<1 分钟拉节点）、调度层（KAI Scheduler，组调度防部分分配）、应用层（Dynamo Planner/llm-d，基于队列深度和 KV 缓存利用率扩缩）。`DCGM_FI_DEV_GPU_UTIL` 不适合做扩缩信号——改用队列深度。Karpenter 的 `WhenEmptyOrUnderutilized` 对推理工作负载有破坏性——换成 `WhenEmpty + consolidateAfter: 1h`。

下一课我们将进入 vLLM 的在线服务内部原理——理解推理引擎如何工作以便正确配置它。

---

## ✏️ 练习

1. 运行 `code/main.py`。在突发工作负载下，基于占空比的 HPA 相比基于队列深度的 HPA 多丢了多少请求？
2. 为 serve Llama 3.3 70B FP8的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter` 和防止非 GPU 工作负载的 taint。
3. 你的团队报告部署卡在 Pending，说"GPU 可用但 Pod 不能调度"。诊断——是 Karpenter、kube-scheduler 还是 KAI Scheduler？
4. 选择信号来扩缩分离式 prefill 和 decode Pod。分别为两者说明理由。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 三层扩缩容模拟器 | `code/main.py` | 对比朴素 HPA、队列深度 HPA 和组调度 |
| GPU 扩缩容方案设计 | `outputs/skill-gpu-autoscaler-plan.md` | 根据集群拓扑和工作负载设计三层扩缩容方案 |

---

## 📖 参考资料

1. [GitHub] KAI Scheduler. https://github.com/kai-scheduler/KAI-Scheduler
2. [文档] Karpenter Disruption Controls. https://karpenter.sh/docs/concepts/disruption/
3. [NVIDIA] Disaggregated LLM Inference on Kubernetes. https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/
4. [AWS] EKS Compute and Autoscaling Best Practices. https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html
