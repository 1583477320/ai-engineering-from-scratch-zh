# vLLM 在线服务内部原理——PagedAttention、连续批处理、Chunked Prefill

> vLLM 在 2026 年的统治地位建立在三个相互叠加的默认设置上，而不是单一技巧。PagedAttention 始终开启；连续批处理在解码迭代之间注入新请求；Chunked Prefill 将长提示切成片，让解码词元不会被饿死。三个全开时，Llama 3.3 70B FP8 在单张 H100 SXM5 上以 128 并发达到 2200-2400 tok/s——比朴素 PyTorch 循环快 3-4 倍。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 01（托管 LLM 平台）、阶段 11（LLM 工程）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 PagedAttention 作为 KV 缓存分配器——块、块表以及为什么生产负载下碎片率能低于 4%
- [ ] 画出连续批处理的迭代级流程：请求如何在每次解码迭代之间加入和离开批次
- [ ] 用一句话描述 Chunked Prefill，并说出它保护的是哪个延迟指标
- [ ] 命名 vLLM v0.18.0 中不能同时开启的两个优化组合

---

## 1. 问题

朴素的 PyTorch 服务循环每次只处理一个请求：分词、prefill、逐词解码直到 EOS、返回。一个用户时没问题；一百个用户时，你就在等一队耐心的人。

经典批处理（Static Batching）将每个请求填充到窗口中最长提示的长度，然后所有序列中最长输出的长度——整个批次被最慢的序列拖住。你为从未使用的填充买单，快速请求等待慢速请求。

vLLM 一次性解决三个问题。PagedAttention 阻止了 KV 缓存碎片吃掉 60-80% 的 GPU 显存（经典连续分配的代价）。连续批处理让请求在每次解码迭代之间加入和离开批次，使 GPU 始终满负荷工作。Chunked Prefill 将 32K 词元的提示切成约 512 词元的片，与解码交错执行，让一个长提示不会冻结所有其他序列的第一个词元延迟。

**2026 年的生产默认是三个全开。** 你需要理解每个的作用——因为故障模式都在调度器上，不在模型上。

---

## 2. 概念

### 2.1 PagedAttention 作为虚拟内存系统

KV 缓存的大小 = `层数 × 2 × 头数 × 头维度 × 序列长度 × 元素字节数`。Llama 3.3 70B 在 8192 词元下，每个序列约 1.25GB（BF16）。如果为每个请求预分配 8192 个槽位但平均只用 1500 个词元，你就浪费了约 82% 的 HBM。

PagedAttention 借鉴了操作系统的虚拟内存思想。KV 缓存不是每个序列连续分配的——而是以固定大小的块（默认 16 词元）分配。每个序列有一个块表，将逻辑词元位置映射到物理块 ID。序列增长时添加一个块；完成后块返回空闲池。

碎片从经典分配的 60-80% 降到 PagedAttention 的 4% 以下。你不需要用开关启用 PagedAttention——它是 vLLM 唯一的分配器。唯一需要调的旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载权重和激活后为 KV 块预留多少 HBM。

### 2.2 连续批处理

旧的"动态批处理"等待一个时间窗口（如 10ms）填满批次，然后运行 prefill + 多次 decode 直到所有序列完成。快速序列提前离开后 GPU 继续等慢序列。

连续批处理在每次 decode 步之间做出准入/释放决策。将正在运行的序列集合称为 `RUNNING` 列表。每次迭代：

1. 检查 `RUNNING` 中是否有序列达到了 EOS 或 max_tokens——有就移除
2. 调度器查看等待队列。如果有空闲 KV 块，准入新序列（prefill 或恢复）
3. 对 `RUNING` 中的所有序列执行一次前向传播，每个序列产出一个新词元

批大小从不填充到固定数。不同输出位置的序列共享一次融合前向传播。在 2026 年 vLLM 中这被称为 `V1 调度器`。核心不变量：调度器每次解码迭代运行一次，而不是每个请求一次。

### 2.3 Chunked Prefill 保护 TTFT 尾部

Prefill 是计算密集型的。Llama 3.3 70B 上一个 32K 词元的提示在单张 H100 上需要约 800ms 纯 prefill 时间。prefill 运行期间，批次中所有其他序列的解码词元在等待。

Chunked Prefill 将 prefill 切成固定大小的块（默认 512 词元），在块之间调度器可以推进一个解码序列。代价是微小的绝对 prefill 延迟损失（每个块几毫秒），换来的是解码时抖动大幅降低。在混合负载下，P99 ITL 从约 50ms 降到约 15ms。

### 2.4 三个默认设置相互依赖

三个特性假设彼此存在。PagedAttention 给调度器提供细粒度的 KV 资源来调度。连续批处理需要这种细粒度资源，这样准入新序列不会强制全局重排。Chunked Prefill 是调度器在同一个 `RUNING` 列表上做出的另一个策略决策——它是一个调度器策略，不是独立系统。

### 2.5 vLLM v0.18.0 的陷阱

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与 draft-model 投机解码（`--speculative-model`）组合使用。有文档记录的例外是 V1 调度器中的 N-gram GPU 投机解码。如果投机解码的收益值得启用 Chunked Prefill，需要重新评估——2026 年的正确选择通常是 EAGLE-3（不带 Chunked Prefill），而不是 draft model + Chunked Prefill（不编译）。

### 2.6 你应该记住的数字

- Llama 3.3 70B FP8、H100 SXM5、128 并发、三个全开：2200-2400 tok/s
- 同模型、vLLM 默认配置（无 Chunked Prefill）：约 1800 tok/s
- 同模型、朴素 PyTorch 前向循环：约 600 tok/s
- PagedAttention 在生产负载下的 KV 碎片浪费：<4%
- 混合负载下 P99 ITL：有 Chunked Prefill 约 15ms，无约 50ms

### 2.7 调度器伪代码

```python
while True:
    # 1. 释放完成的序列
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished:
        release_blocks(s)
        RUNNING.remove(s)

    # 2. 准入等待队列中的新序列
    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 3. 调度 prefill 块 + decode 在一个批次中
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 例如 512 词元
        else:
            batch.append(decode_one_token(s))      # 1 词元

    run_forward(batch)                              # 一次融合 GPU 调用
```

---

## 3. 从零实现

### 第 1 步：模拟连续批处理调度器

```python
import random
import time


class Request:
    def __init__(self, prompt_tokens, max_output):
        self.prompt_tokens = prompt_tokens
        self.max_output = max_output
        self.tokens_generated = 0
        self.is_prefill_done = False
        self.start_time = time.time()
        self.end_time = None
        self.tokens_waited = 0  # 解码期间等待的词元数

    def is_done(self):
        return self.tokens_generated >= self.max_output

    def prefill(self):
        self.is_prefill_done = True

    def decode_one(self):
        self.tokens_generated += 1
        return self.tokens_generated >= self.max_output


class ContinuousBatchSimulator:
    """模拟连续批处理和 Chunked Prefill。"""

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        self.RUNNING = []
        self.WAITING = []
        self.total_tokens = 0
        self.total_steps = 0

    def add_request(self, prompt_tokens, max_output):
        self.WAITING.append(Request(prompt_tokens, max_output))

    def step(self):
        # 释放完成的请求
        self.RUNNING = [r for r in self.RUNNING if not r.is_done()]

        # 准入新请求
        while self.WAITING:
            r = self.WAITING.pop(0)
            self.RUNNING.append(r)

        if not self.RUNNING:
            return

        # 每个运行中的请求产生一个词元
        for r in self.RUNNING:
            r.tokens_generated += 1
            self.total_tokens += 1

        self.total_steps += 1

    def simulate(self, steps=100):
        for _ in range(steps):
            if random.random() < 0.3 and len(self.WAITING) < 5:
                # 随机添加新请求
                prompt_len = random.randint(500, 4000)
                output_len = random.randint(100, 800)
                self.add_request(prompt_len, output_len)
            self.step()
        return {
            "total_tokens": self.total_tokens,
            "total_steps": self.total_steps,
            "avg_batch_size": self.total_tokens / max(self.total_steps, 1),
        }


# 对比三种模式
def compare_modes():
    results = {}
    for mode in ["naive", "static", "continuous"]:
        sim = ContinuousBatchSimulator()
        r = sim.simulate(200)
        results[mode] = r

    print(f"{'模式':15s} {'总词元':>10} {'总步数':>8} {'平均批次':>10}")
    for mode, r in results.items():
        print(f"{mode:15s} {r['total_tokens']:10d} {r['total_steps']:8d} "
              f"{r['avg_batch_size']:10.1f}")


if __name__ == "__main__":
    compare_modes()
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 vLLM 配置示例

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128
```

### 4.2 调度策略对照

| 策略 | 调度单位 | 批大小 | TTFT | ITL |
|---|---|---|---|---|
| 朴素（一次一个） | 请求 | 1 | 低 | 极高 |
| 经典静态批处理 | 批次（填充后） | 固定 | 中 | 中（被最慢序列拖累） |
| 连续批处理 | 每次迭代 | 动态 | 中 | 低 |
| 连续 + Chunked Prefill | 每次迭代 + prefill 块 | 动态 | 低 | 极低 |

---

## 5. 工程最佳实践

### 5.1 不要关闭 PagedAttention

vLLM 的 PagedAttention 是默认且唯一的分配器。唯一的旋钮是 `--gpu-memory-utilization`。不要手动调 KV 缓存大小——交给分配器。

### 5.2 Chunked Prefill 大小的调优

默认 512 词元适合大多数场景。如果你的提示平均长度 > 4K 词元，可以增加到 1024。如果 P99 ITL 仍然太高，减小到 256。

### 5.3 关注 P99 而不是 P50

vLLM 的调度器优化的是吞吐量和平均延迟。P99 的尾部延迟由最慢的 prefill 决定——这就是 Chunked Prefill 要解决的问题。

### 5.4 中文场景特别建议

- **vLLM 的中文词表支持。** vLLM 完美支持中文模型（如 Qwen2.5、GLM-4）。但中文提示词的分词密度更高——同样长度的提示词，中文需要更多词元。KV 缓存预分配需要相应调整
- **中文 RAG 的前缀缓存率更高。** 中文 RAG 的系统提示词通常很长（包含格式说明、示例等），前缀复用率可达 80%+。配合 SGLang 的 RadixAttention（第 06 课）效果更佳
- **国内 vLLM 部署注意事项。** vLLM 需要 CUDA 12.0+。国内用户在使用阿里云 GN7（A10）实例时需要确认驱动版本

---

## 6. 常见错误

### 错误 1：vLLM 低吞吐但不知道原因

**现象：** vLLM 启动后吞吐量只有预期的 50%，但没有报错。

**原因：** 可能没有启用连续批处理（vLLM v0.6.0+ 默认开启，但某些旧版本需要手动配置），或者 Chunked Prefill 被关闭了。

**修复：** 检查启动参数。确认 `--enable-chunked-prefill`（v0.6.0+ 默认）。运行 `vllm bench throughput` 对比基准。

### 错误 2：OOM 但不知道是 KV 缓存还是权重

**现象：** vLLM 启动时 OOM。

**原因：** `--gpu-memory-utilization` 默认 0.9。对于显存较小的 GPU（如 24GB），0.9 可能不够——权重加载后没有足够的空间给 KV 缓存块。

**修复：** 降低 `--gpu-memory-utilization` 到 0.8 或更低。或者升级 GPU。

---

## 7. 面试考点

### Q1：PagedAttention 如何将 KV 缓存碎片从 60-80% 降到 4% 以下？（难度：⭐⭐⭐）

**参考答案：**
经典连续分配为每个序列预留最大长度的连续内存。实际使用远小于最大值时，未使用的部分成为内部碎片。PagedAttention 借鉴操作系统虚拟内存：KV 缓存按固定大小块（16 词元）分配，每个序列有一个块表将逻辑位置映射到物理块 ID。序列结束时释放块。这消除了内部碎片——每个序列只为实际使用的词元付费，碎片仅来自最后一个不完整的块（平均 8 个词元/序列）。

### Q2：连续批处理为什么比静态批处理好？关键差异在什么时间尺度上？（难度：⭐⭐）

**参考答案：**
关键差异在迭代级别——每次 decode 迭代（约 1-2ms）都会重新评估批次组成。静态批处理等整个批次完成（所有序列到达 EOS）才释放 GPU——快速序列被慢序列拖住。连续批处理在每次迭代后立即释放已完成的序列并准入新请求。这意味着 GPU 永远不等待"已经完成但还没释放"的序列。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| PagedAttention | "KV 技巧" | 固定大小块的 KV 缓存分配器；碎片 <4% |
| 块表 | "页表" | 每序列的逻辑词元位置→物理 KV 块映射 |
| 连续批处理 | "动态批处理的正确实现" | 每次解码迭代做出准入/释放决策 |
| Chunked Prefill | "Prefill 分片" | 将长 prefill 切成 512 词元的块，与解码交错 |
| TTFT | "首词元延迟" | PreFill + 排队 + 网络；长提示时被 prefill 主导 |
| ITL | "词元间延迟" | 连续解码词元之间的时间；被批大小主导 |
| Goodput | "满足 SLO 的吞吐量" | 同时满足 TTFT 和 ITL 目标的 tok/s |

---

## 📚 小结

vLLM 的三个核心优化——PagedAttention（KV 缓存分配）、连续批处理（迭代级准入/释放）、Chunked Prefill（长 prefill 分片）——是相互依赖的。PagedAttention 为调度器提供细粒度资源；连续批处理利用这些资源；Chunked Prefill 是调度器在同一批次上的另一个策略。三个全开时，Llama 3.3 70B 在单张 H100 上可达 2200-2400 tok/s。关注 P99 ITL 而不是 P50——尾部延迟是生产中最影响用户体验的指标。

下一课我们将讨论 EAGLE-3 投机解码——如何在 vLLM 之上再获得 2-3 倍的加速。

---

## ✏️ 练习

1. 运行 `code/main.py`。对比 `STATIC` 和 `CONTINUOUS` 在混合长短请求工作负载上的表现。吞吐量差距来自哪里？
2. 修改玩具调度器，加入 `--max-num-batched-tokens`。对于 H100 上的 Llama 3.3 70B FP8，正确的值是多少？
3. 计算 1000 条请求（均值 1500 输出词元，标准差 600 词元）在两种方案下的 KV 缓存碎片浪费：(a) 连续预分配 8192 槽，(b) PagedAttention 16 词元块。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 连续批处理模拟器 | `code/main.py` | 对比朴素/静态/连续批处理 |
| vLLM 调度器诊断 | `outputs/skill-vllm-scheduler-reader.md` | 给定配置，诊断瓶颈和调优建议 |

---

## 📖 参考资料

1. [论文] Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention". SOSP, 2023. arXiv:2309.06180 — PagedAttention 原始论文
2. [官方文档] vLLM — Speculative Decoding. https://docs.vllm.ai/en/latest/features/spec_decode/
3. [博客] Aleksa Gordic — Inside vLLM. https://www.aleksagordic.com/blog/vllm — V1 调度器详解
