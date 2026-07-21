# 综合项目79——流水线并行与气泡分析（Pipeline Parallel and Bubble Analysis）

> 张量并行在设备间切分矩阵乘法。流水线并行在设备间切分模型——每设备一层。微批次在流水线中流动。开始和结束的空闲时间是气泡；最小化它是全部技艺。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第76节
**预计时间：** 90分钟

---

## 学习目标

- 将顺序模型分割为 N 个阶段，模拟前向流水线
- 使用 GPipe 调度运行 M 个微批次并计算气泡比例
- 对比 1F1B 调度的气泡差异
- 讨论阶段分配：每阶段计算量相等比参数量相等更重要

---

## 1. 问题

70B 参数模型需要 140GB 参数。ZeRO-3 跨设备分片参数，但仍需每层 allgather。流水线并行另辟蹊径：将模型切成 N 阶段，每设备一个阶段。层 1 在设备 0 上完成前向后将激活传给设备 1，设备 1 运行层 2，以此类推。内存线性下降，但计算是串行的——这是气泡问题。

气泡比例 = (N-1)/(M+N-1)，其中 M 是微批次数，N 是阶段数。M=8, N=4 时气泡 27%。M=64, N=4 时气泡 4.5%。

---

## 2. 核心概念

### 2.1 GPipe 调度

前向填充所有 M 个微批次，然后反向清空。激活内存随 M 线性增长。前向 M+N-1 周期，反向 M+N-1 周期。有用工作 2M 周期，气泡 2(N-1) 周期。

### 2.2 1F1B 调度

交错：微批次到达最后一个阶段时立即启动反向。每设备前向和反向交替。气泡仍为 N-1，但激活内存被流水线深度限制。

### 2.3 阶段分配

如果阶段 0 耗 50ms 而阶段 1 耗 100ms，每周期都卡在阶段 1 上。其他设备等待 50ms。均衡每阶段的 FLOPs，而非参数量。

---

## 3. 从零实现

```python
"""流水线并行与气泡分析——GPipe + 1F1B。"""
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class Stage:
    id: int; compute_ms: float; name: str = ""

class Pipeline:
    def __init__(self, stages, num_microbatches):
        self.stages = stages
        self.M = num_microbatches
        self.N = len(stages)
        self.timeline = []

    def simulate_gpipe(self):
        """GPipe 调度模拟。"""
        for m in range(self.M):
            for s in range(self.N):
                t = m + s
                self.timeline.append(("fwd", t, s, m))
        for m in range(self.M):
            for s in range(self.N - 1, -1, -1):
                t = self.M + (self.M - 1 - m) + s
                self.timeline.append(("bwd", t, s, m))

    def bubble_fraction(self):
        useful = 2 * self.M
        total = 2 * (self.M + self.N - 1)
        return 1 - useful / total if total > 0 else 0

    def summary(self):
        total = max(t for _, t, _, _ in self.timeline) + 1 if self.timeline else 0
        idle = sum(1 for t in range(total)
                   if not any(t == time for _, time, _, _ in self.timeline))
        return {"total_cycles": total, "bubble_cycles": idle,
                "bubble_fraction": self.bubble_fraction()}


def main():
    stages = [Stage(i, 1.0, f"stage_{i}") for i in range(4)]
    pipe = Pipeline(stages, num_microbatches=8)
    pipe.simulate_gpipe()
    summary = pipe.summary()
    print(f"阶段数: {pipe.N}  微批次数: {pipe.M}")
    print(f"总周期: {summary['total_cycles']}  气泡周期: {summary['bubble_cycles']}")
    print(f"气泡比例: {summary['bubble_fraction']:.3f}")
    print(f"公式预测: {(pipe.N-1)/(pipe.M+pipe.N-1):.3f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 调度 | 特点 |
|:----|:-----|:-----|
| Megatron-LM | 1F1B | 大规模流水线 |
| DeepSpeed Pipeline | GPipe/1F1B | 与 ZeRO 集成 |
| PyTorch Pipe | 同步 | 原生实现 |

---

## 5. 工程最佳实践

- M >> N 是隐藏气泡的关键——M 至少是 N 的 4-8 倍
- 激活检查点与流水线配对——M 个微批次的激活内存需要检查点削减
- **中文场景建议**：流水线并行在 4 GPU 以上的小集群上效果有限

---

## 6. 常见错误

- **M 太小**：气泡比例大，GPU 空闲时间过长
- **阶段不均衡**：一个慢阶段拖累整个流水线
- **死锁**：所有设备先发后收——需要交错发送和接收

---

## 7. 面试考点

**Q1：为什么 1F1B 比 GPipe 更适合长序列？**（难度：⭐⭐⭐）

**参考答案：** GPipe 持有 M 个微批次的激活直到反向传播——内存随 M 线性增长。1F1B 每个微批次到达最后阶段时立即反向，激活内存被流水线深度 N 限制而非微批次数 M。长序列的激活很大，1F1B 的内存优势显著。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 气泡 | 流水线空闲时间比例 = (N-1)/(M+N-1) |
| GPipe | 全部前向后全部反向——激活内存随 M 线性增长 |
| 1F1B | 一前向一反向交替——激活内存被 N 限制 |
| 微批次 | 单次前向/反向单元，气泡随 M 增大而缩小 |

---

## 📚 小结

流水线并行通过将模型分割到多设备上解决大模型内存问题。气泡是串行化的代价——通过增大微批次数可缩小。下一节构建分片检查点。

---

## ✏️ 练习

1. 【实验】用不同的 M（2, 4, 8, 16, 32）和 N=4 运行 GPipe 气泡分析
2. 【实现】实现 1F1B 调度，验证气泡比例与 GPipe 相同但激活内存更低

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 流水线并行 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Huang et al. "GPipe: Efficient Training of Giant Neural Networks". NeurIPS 2019.
2. [论文] Narayanan et al. "PipeDream". MLSys 2019.
