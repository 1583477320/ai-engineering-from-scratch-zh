# 智能体可观测性：Langfuse、Phoenix、Opik

> 2026 年有三个开源智能体可观测性平台主导市场。Langfuse（MIT）——每月 600 万+安装，追踪+提示词管理+评估+会话重放。Arize Phoenix（Elastic 2.0）——深度智能体特定评估、RAG 相关性、OpenInference 自动插桩。Comet Opik（Apache 2.0）——自动提示词优化、护栏、LLM-Judge 幻觉检测。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 23（OTel GenAI）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出三个顶级开源智能体可观测性平台及其许可
- [ ] 对比 Langfuse、Phoenix、Opik 的功能差异
- [ ] 实现生产级追踪——记录工具调用、LLM 输出、成本
- [ ] 设计一个可观测性方案——选择适合项目需求的平台

---

## 1. 问题

智能体出了问题——但你不知道是哪个环节出了问题。是 LLM 幻觉了？是工具返回了错误数据？还是检索结果不相关？没有可观测性，调试就是猜测。

三个平台在 2026 年主导市场——各有优劣。

---

## 2. 概念

### 2.1 三大平台对比

| 平台 | 许可 | 核心功能 | 安装量 |
|------|------|---------|--------|
| **Langfuse** | MIT | 追踪+提示词管理+评估+会话重放 | 600 万+/月 |
| **Arize Phoenix** | Elastic 2.0 | 智能体评估、RAG 相关性、OpenInference | 研究导向 |
| **Comet Opik** | Apache 2.0 | 自动提示词优化、护栏、幻觉检测 | 生产级 |

### 2.2 追踪的关键指标

| 指标 | 说明 | 报警阈值 |
|------|------|---------|
| **延迟** | 每个 span 的执行时间 | P95 > 5s |
| **成本** | 每次调用的 token 成本 | > 预算阈值 |
| **错误率** | 失败的工具调用比例 | > 1% |
| **幻觉率** | LLM 输出与检索内容不一致 | > 5% |

### 2.3 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 开发/研究 | Phoenix | 深度评估 |
| 生产部署 | Langfuse | 追踪+监控 |
| 提示词优化 | Opik | 自动优化 |

---

## 3. 从零实现

### Step 1：简单追踪器

```python
import time

class SimpleTracer:
    """简单追踪器——模拟可观测性平台。"""
    def __init__(self):
        self.traces = []

    def trace(self, name, fn, **kwargs):
        """追踪函数执行。"""
        start = time.time()
        try:
            result = fn(**kwargs)
            latency = (time.time() - start) * 1000
            self.traces.append({"name": name, "latency_ms": latency,
                               "status": "ok", "input": str(kwargs)[:50]})
            return result
        except Exception as e:
            latency = (time.time() - start) * 1000
            self.traces.append({"name": name, "latency_ms": latency,
                               "status": "error", "error": str(e)})
            raise

    def summary(self):
        ok = sum(1 for t in self.traces if t["status"] == "ok")
        total = len(self.traces)
        avg_lat = sum(t["latency_ms"] for t in self.traces) / max(total, 1)
        return {"total_spans": total, "ok": ok, "error_rate": f"{(total-ok)/max(total,1):.1%}",
                "avg_latency": f"{avg_lat:.1f}ms"}
```

---

## 4. 工具

### 4.1 平台对比

| 平台 | 特点 | 适用 |
|------|------|------|
| Langfuse | 追踪+评估+会话重放 | 生产 |
| Arize Phoenix | 深度智能体评估 | 研发 |
| Comet Opik | 自动优化+护栏 | 生产 |

### 4.2 集成方式

```python
# Langfuse
from langfuse import Langfuse
langfuse = Langfuse()

@langfuse.observe()
def process_request(query):
    # Langfuse 自动追踪
    return response
```

---

## 5. 工程最佳实践

### 5.1 追踪设计

- **每个工具调用一个 span**：细粒度追踪
- **包含成本和延迟**：量化每个步骤的消耗
- **记录输入输出**：便于调试和审查
- **采样追踪**：生产环境不需要追踪所有请求

---

## 6. 常见错误

### 错误 1：只追踪 LLM 调用

**现象：** 工具失败时无法定位问题。

**修复：** 追踪整个管道——LLM、工具、检索、每个步骤都要。

---

## 7. 面试考点

### Q1：如何选择智能体可观测性平台？（难度：⭐⭐）

**参考答案：**
如果主要需求是追踪和会话重放→Langfuse。如果需要深度智能体评估（RAG 相关性、幻觉检测）→Phoenix。如果需要自动提示词优化和护栏→Opik。大多数生产场景推荐 Langfuse——它最成熟、社区最大。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Langfuse | "智能体追踪平台" | 开源——追踪+评估+会话重放，每月 600 万+安装 |
| Phoenix | "智能体评估" | Arize 开源——深度智能体评估、RAG 相关性 |
| Opik | "提示词优化" | Comet 开源——自动优化、护栏、幻觉检测 |

---

## 📚 小结

三大开源可观测性平台：Langfuse（追踪为主）、Phoenix（评估为主）、Opik（优化为主）。选择取决于主要需求：追踪→Langfuse，评估→Phoenix，优化→Opik。

---

## ✏️ 练习

1. **【对比】** 对比 Langfuse 和 Phoenix 在同一智能体应用上的追踪信息
2. **【设计】** 为一个 RAG 管道设计可观测性方案——需要追踪哪些指标？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 简单追踪器 | `code/main.py` | 追踪 + 摘要 + 错误率统计 |

---

## 📖 参考资料

1. [GitHub] Langfuse: https://github.com/langfuse/langfuse
2. [GitHub] Arize Phoenix: https://github.com/Arize-ai/phoenix
3. [GitHub] Comet Opik: https://github.com/comet-ml/opik
