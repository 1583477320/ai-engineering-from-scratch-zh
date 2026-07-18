# SGLang 与 RadixAttention——前缀密集型工作负载的优化

> SGLang 将 KV 缓存视为可复用的一等资源，存储在 Radix 树中。vLLM 按 FCFS（先到先服务）调度请求，SGLang 的缓存感知调度器优先调度共享更长前缀的请求——本质上是深度优先的 Radix 树遍历，让热分支留在 HBM 中。在 Llama 3.1 8B 上以 ShareGPT 风格的 1K 提示词测试，SGLang 达到约 16,200 tok/s，比 vLLM 的约 12,500 快约 29%。在前缀密集的 RAG 工作负载上优势达到 6.4 倍。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 内部原理）、阶段 14（智能体 RAG）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 画出 RadixAttention：前缀如何存储在 Radix 树中，KV 块如何在同分支的序列间共享
- [ ] 解释缓存感知调度以及为什么 FCFS 对前缀密集型流量是错误的
- [ ] 根据前缀缓存命中率和提示长度分布计算预期加速比
- [ ] 命名让 6.4 倍加速成为现实的提示排序纪律

---

## 1. 问题

经典在线服务将每个请求的提示视为不透明的。即使 5000 个 RAG 请求都以相同的 2000 词元系统提示加相同的检索前缀开头，vLLM 也会预填充这 2000 词元前缀 5000 次。GPU 做着完全相同的工作。

观察：智能体和 RAG 工作负载中的提示几乎总是共享很长的前缀。系统提示、工具 schema、少样本示例、检索头、对话历史——跨请求重复。如果存储一次那个前缀的 KV 缓存并复用，就不再需要预填充它。

**RadixAttention 正是这样做的。** 词元被索引在 Radix 树（紧凑前缀树）中；每个节点拥有从根到该节点路径上的词元序列对应的 KV 块。新请求遍历树：任何词元匹配的节点都复用该节点的 KV 块。Prefill 成本变为与"新"后缀成比例，而不是整个提示。

---

## 2. 概念

### 2.1 Radix 树作为 KV 索引

```
根
 ├─ "You are a helpful assistant..."   (2000 词元, 124 KV 块)
      ├─ "Context: <doc A>..."         (500 词元, 31 块)
           ├─ "Question: Alice..."     (80 词元, 5 块)
           ├─ "Question: Bob..."       (95 词元, 6 块)
      ├─ "Context: <doc B>..."         (520 词元, 33 块)
```

新请求以系统提示 + "Context: <doc A>" + "Question: Carol" 到达。调度器遍历：系统前缀匹配（复用 124 块），doc-A 分支匹配（复用 31 块），然后只为 "Question: Carol" 分配新块（4 块）。Prefill 成本：4 块新词元。没有树时：160 块。prefill 节省约 40 倍。

### 2.2 缓存感知调度

Radix 树复用的前提是缓存不抖动。两个关键策略：

1. **深度优先调度。** 从队列中选择下一个请求时，优先选择与当前运行集合共享同分支的请求。让热分支保持驻留
2. **分支级 LRU。** 从最短未使用的叶节点开始驱逐整个分支，而不是逐块驱逐——缓存形状匹配 Radix 树形状

FCFS 违反了两条。一个共享 2000 词元的请求排在一个只共享 50 词元的请求后面——然后热分支被驱逐以接纳下一个请求。

### 2.3 基准数据

- Llama 3.1 8B、H100、ShareGPT 1K 提示：SGLang ~16,200 tok/s vs vLLM ~12,500（~29% 优势）
- 前缀密集 RAG（相同系统提示+相同文档，变化问题）：最高 6.4 倍
- 语音克隆工作负载：86.4% 前缀缓存命中率
- SGLang 客户的生产命中率：50-99%（取决于提示纪律）

### 2.4 排序陷阱

6.4 倍数字依赖于**一致的提示模板排序**。如果客户端在一些请求中按 `[system, tools, context, history, question]` 构建提示，在另一些请求中按 `[system, context, tools, history, question]`——树无法找到共享前缀。对人类来说是共享前缀，对 Radix 树来说是两个不同的序列。

**工程师的杠杆：你的提示模板就是缓存键。** 固定顺序。不可变的内容（系统提示、工具、schema）放最前面。检索上下文放中间。用户问题放最后。不要在前缀中插入动态内容。

**真实案例：** 将动态内容移出可缓存前缀，一个部署就将缓存命中率从 7% 提升到 74%。

### 2.5 RadixAttention 赢和输的场景

**赢的场景：**
- RAG（相同检索前缀，不同问题）
- 智能体（相同工具 schema，不同查询）
- 带长系统提示的聊天
- 语音/视觉工作负载中的重复前缀

**输的场景（回到 vLLM 级吞吐）：**
- 每个提示都唯一的单次生成（代码补全、无系统提示的开放式聊天）
- 动态提示中每个请求的前缀都穿插了唯一内容

### 2.6 为什么这是调度器问题而非仅仅是内核问题

KV 复用可以作为内核技巧实现。SGLang 的洞察是：只有调度器让热分支保持驻留时，复用才真正产生收益。朴素的"如果可用就复用"策略在混合负载下会抖动缓存。**Radix 树索引的调度器才是将内核技巧转化为生产优势的关键。**

### 2.7 与 vLLM 的关系

2026 年 vLLM 也添加了前缀缓存（`--enable-prefix-caching`）和缓存感知路由（vLLM Router）。差距缩小但没有完全消除——SGLang 的整个栈是 Radix 优先的，vLLM 是后加的。对于前缀复用占主导的工作负载，SGLang 仍然是默认选择。对于没有强前缀模式的通用服务，vLLM 仍然相当或更好。

---

## 3. 从零实现

### 第 1 步：Radix 树前缀缓存

```python
class RadixCache:
    """简化版 Radix 树 KV 缓存。"""

    def __init__(self, block_size=16):
        self.block_size = block_size
        self.tree = {}  # node_id -> {"tokens": str, "children": dict, "kv_blocks": int}

    def lookup(self, token_ids: list[int]) -> tuple[int, int]:
        """遍历树，返回（匹配的块数，未匹配的词元数）。"""
        matched_tokens = 0
        node = "root"

        for i, token_id in enumerate(token_ids):
            child_key = f"{node}:{token_id}"
            if child_key in self.tree:
                node = child_key
                matched_tokens += 1
            else:
                break

        matched_blocks = matched_tokens // self.block_size
        unmatched = len(token_ids) - matched_tokens
        return matched_blocks, unmatched

    def insert(self, token_ids: list[int], kv_blocks: int):
        """将词元序列插入树中。"""
        node = "root"
        for token_id in token_ids:
            child_key = f"{node}:{token_id}"
            if child_key not in self.tree:
                self.tree[child_key] = {"children": {}, "kv_blocks": 0}
            self.tree[child_key]["kv_blocks"] = kv_blocks
            node = child_key

    def hit_rate(self) -> float:
        """缓存命中率（简化）。"""
        total_blocks = sum(n.get("kv_blocks", 0) for n in self.tree.values())
        if total_blocks == 0:
            return 0.0
        return min(1.0, total_blocks / 1000)  # 简化模拟
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 SGLang 部署

```bash
pip install sglang

python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp 1 \
    --enable-radix-attention \
    --mem-fraction-static 0.8
```

### 4.2 SGLang vs vLLM 对照

| 维度 | vLLM | SGLang |
|---|---|---|
| KV 缓存分配 | PagedAttention（块） | PagedAttention + Radix 树索引 |
| 调度策略 | FCFS（先到先服务） | 缓存感知（热分支优先） |
| 前缀缓存 | 启动参数 opt-in | 默认开启 |
| 最佳场景 | 通用在线服务 | RAG/智能体前缀密集工作负载 |
| 通用吞吐 | 高 | 高（无前缀优势时相当） |
| 400K+ GPU 生产部署 | vLLM | SGLang |

---

## 5. 工程最佳实践

### 5.1 固定提示模板排序

你的提示模板就是缓存键。将不可变部分（系统提示、工具、schema）放最前面，检索上下文放中间，用户问题放最后。**不要在不同请求中调换顺序。**

### 5.2 前缀长度是关键

前缀越长，RadixAttention 的优势越大。2000 词元的系统提示意味着每次前缀命中可以节省约 124 个 KV 块的 prefill。500 词元的短前缀节省很少。

### 5.3 部署后测量缓存命中率

SGLang 会报告 Radix 树的缓存命中率。目标是 70%+。如果命中率低于 50%，检查提示排序是否一致。

### 5.4 中文场景特别建议

- **中文前缀复用率更高。** 中文 RAG 和智能体的系统提示词通常更长（包含格式说明、示例等），且中文提示词的词元密度更高——同样的语义信息需要更少的词元。RadixAttention 对中文 RAG 的加速可能比英文更显著
- **SGLang 的中文模型支持。** SGLang 支持 Qwen2.5、GLM-4 等主流中文模型。与 vLLM 使用相同的模型格式
- **国内 SGLang 部署。** SGLang 是开源项目，可在国内任何 GPU 实例上部署。不依赖海外服务

---

## 6. 常见错误

### 错误 1：提示模板排序不一致导致缓存命中率低

**现象：** 部署 SGLang 后缓存命中率只有 7%，没有看到预期的加速。

**原因：** 不同的请求中，提示组件的顺序不一致——有时候工具 schema 在前，有时候检索上下文在前。Radix 树无法识别共享前缀。

**修复：** 固定提示模板顺序——不可变部分放最前，动态部分放最后。

### 错误 2：在非前缀密集场景使用 SGLang

**现象：** 每个请求的提示都完全不同（如代码补全）。SGLang 的吞吐量和 vLLM 相当，但没有优势。

**原因：** 没有可复用的前缀——RadixAttention 的优势无法发挥。

**修复：** 对于通用在线服务（无前缀模式），vLLM 或 SGLang 都可。对于前缀密集（RAG/智能体），优先 SGLang。

---

## 7. 面试考点

### Q1：RadixAttention 如何将 RAG 的 prefill 成本降低 40 倍？（难度：⭐⭐⭐）

**参考答案：**
RAG 请求通常共享 2000+ 词元的系统提示+检索前缀。Radix 树将这个前缀存储为一个分支，拥有对应的 KV 块。新请求遍历树时匹配到该分支——124 个 KV 块被直接复用。只需要为新的用户问题（约 4 词元）分配新块。没有 Radix 树时，每个请求都要重新 prefill 全部 160 块。节省 = 160/4 = 40 倍。

### Q2：SGLang 的缓存感知调度与 FCFS 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
FCFS 按到达顺序服务请求——先到先得。SGLang 优先调度与当前运行集合共享最长前缀的请求——深度优先遍历 Radix 树。这保持了热分支在 HBM 中驻留。FCFS 在混合负载下会驱逐热分支来接纳短前缀请求，导致缓存抖动。SGLang 通过深度优先调度防止了这个问题。

### Q3：如果客户的缓存命中率只有 8%，你会诊断哪些问题？（难度：⭐⭐⭐）

**参考答案：**
三个可能的原因：第一，提示模板排序不一致——不同请求的组件顺序不同，树无法共享。诊断：检查不同请求的提示构建代码。第二，每个请求的提示都不同（无共享前缀）。诊断：检查是否有系统提示、工具 schema 等可复用部分。第三，缓存大小不够——HBM 容量不足以容纳热分支。诊断：检查 `--mem-fraction-static` 配置和 HBM 使用率。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| RadixAttention | "SGLang 的特性" | KV 缓存按 Radix 树索引，共享前缀复用块 |
| Radix 树 | "紧凑前缀树" | 每节点拥有词元范围和对应 KV 块 |
| 缓存感知调度器 | "热分支优先" | 优先调度共享驻留分支的请求 |
| 前缀缓存命中率 | "多少提示词是免费的" | 从复用的 KV 块中提供的提示词元比例 |
| FCFS | "先到先服务" | 默认调度策略，破坏前缀局部性 |
| 分支级 LRU | "驱逐叶子节点" | 与 Radix 树形状匹配的驱逐策略 |
| 提示模板排序 | "缓存键" | 提示的组件顺序决定了树能共享什么 |

---

## 📚 小结

SGLang 通过 RadixAttention 将 KV 缓存变为可复用资源——共享前缀的请求不再重复 prefill。缓存感知调度器（深度优先+分支级 LRU）保持热分支驻留。在 RAG 和智能体工作负载上优势可达 6.4 倍，但前提是提示模板排序一致——这是工程师的关键杠杆。固定提示顺序 = 固定缓存键 = 高命中率。vLLM 也添加了前缀缓存，但 SGLang 的整个栈是 Radix 优先的——对于前缀密集场景，SGLang 仍是默认选择。

第 17 章（基础设施与生产）第 1-6 课完成。下一课我们将讨论 TensorRT-LLM 和 NVIDIA Blackwell 架构。

---

## ✏️ 练习

1. 运行 `code/main.py`。对比 FCFS 和缓存感知调度在相同工作负载上的表现。差距来自 prefill 节省、decode 节省还是排队延迟？
2. 修改工作负载，让提示随机排列 `[system, tools, context]`。重跑。命中率发生了什么变化？为什么？
3. 计算 Llama 3.1 8B 上保持 2000 词元系统提示作为 Radix 分支常驻的 HBM 成本。与 16 序列批处理无前缀复用的成本对比。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| Radix 缓存模拟器 | `code/main.py` | 对比 FCFS 和缓存感知调度 |
| Radix 调度器选型建议 | `outputs/skill-radix-scheduler-advisor.md` | 根据工作负载决定是否采用 SGLang |

---

## 📖 参考资料

1. [GitHub] SGLang. https://github.com/sgl-project/sglang — 源码和文档
2. [论文] Zheng, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs". arXiv:2312.07104 — 设计参考
3. [博客] LMSYS — SGLang with RadixAttention. https://www.lmsys.org/blog/2024-01-17-sglang/ — 基准数据
4. [官方文档] vLLM — Prefix Caching. https://docs.vllm.ai/en/latest/features/prefix_caching.html — vLLM 的 Radix 类实现
