# 监督者/编排者-工作者模式

> 一个领智能体计划和委派；专家工作者在独立上下文中并行执行并报告回来。这是 Anthropic Research 系统背后的模式（Claude Opus 4 作为领，Sonnet 4 作为子智能体），在内部研究评估中比单智能体 Opus 4 高出 +90.2%。Anthropic 的工程文章报告 BrowseComp 80% 的方差仅由词元用量解释——多智能体获胜主要是因为每个子智能体获得一个新的上下文窗口。

**类型：** 概念课 + 实现课
**语言：** Python（标准库，`threading`）
**前置知识：** 阶段 16 · 04（原语模型）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释监督者模式的三个胜利机制：新鲜上下文、专业化提示词、并行化
- [ ] 实现一个带并行工作者的监督者——领智能体分解任务、工作者并行执行、领智能体合成
- [ ] 识别监督者模式的三种失败模式：领幻觉分解、工作者过度探索、合成冲突
- [ ] 理解 Anthropic 的工程课程：规模努力适配查询复杂度、宽然后窄、彩虹部署

---

## 1. 问题

研究是单智能体系统失败的典型任务。你问"2023 到 2026 年多智能体系统发生了什么变化？" 单智能体顺序读五篇论文，用它们的文本填满一半上下文，然后必须同时推理所有内容。它读到第五篇论文时已经忘了第一篇。它无法并行化。

监督者模式修复了这个：一个领智能体计划搜索，将每个子问题委派给工作者，然后合成。每个工作者为一个问题获得自己的 200K 词元窗口。领智能体永远看不到原始材料——只看到工作者摘要。

Anthropic 的生产 Research 系统报告在内部研究评估中比单 Opus 4 高出 +90.2%。同一文章指出 BrowseComp 80% 的方差仅由词元用量解释。每个子智能体获得新上下文是主要机制。

---

## 2. 概念

### 2.1 模式结构

```
                 ┌──────────────┐
                 │   领智能体    │  计划、分解、
                 │  (Opus 4)    │  合成
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ 工作者1  │  │ 工作者2  │  │ 工作者3  │
      │(Sonnet)  │  │(Sonnet)  │  │(Sonnet)  │
      │ 新鲜     │  │ 新鲜     │  │ 新鲜     │
      │ 上下文   │  │ 上下文   │  │ 上下文   │
      └─────────┘  └─────────┘  └─────────┘
```

领智能体永远不读原始材料。工作者直到领智能体合成时才看到彼此的工作。每条箭头是带有窄工件的移交。

### 2.2 三个胜利机制

| 机制 | 说明 | 量化影响 |
|------|------|---------|
| 每个子智能体的新鲜上下文 | 工作者不携带领的 40K 词元规划 | BrowseComp 80% 方差来自词元用量 |
| 专业化提示词 | 领："分解和合成"；工作者："找到 X 的变化" | 专注提示词产生专注输出 |
| 并行化 | 工作者并发运行 | 墙钟时间 ≈ max(工作者时间) + 计划 + 合成 |

### 2.3 Anthropic 的工程课程（2025）

| 课程 | 说明 |
|------|------|
| 规模努力适配查询复杂度 | 简单查询：1 个智能体；复杂查询：10+ 工作者 |
| 宽然后窄 | 先分解为宽子问题，然后每个子问题按需生成更多工作者 |
| 彩虹部署 | 长期运行有状态智能体需要渐进版本切换 |
| 词元用量主导 | 多智能体约为单智能体的 15× 词元。仅在任务价值证明成本时使用 |

### 2.4 三种失败模式

| 失败 | 说明 | 后果 |
|------|------|------|
| 领幻觉分解 | 领生成不分解真实问题的子问题 | 工作者在错误目标上做精确研究 |
| 工作者过度探索 | 无明确范围边界，工作者漂移到子问题之外 | 污染合成步骤 |
| 合成冲突 | 两个工作者返回矛盾事实 | 领必须标记分歧而不是静默选一边 |

### 2.5 何时监督者模式不合适

- **顺序任务**——如果步骤 2 字面上需要步骤 1 的输出，并行没有帮助。用流水线。
- **简单查询**——单智能体更快更便宜。在生成工作者之前用领的"规模努力"检查。
- **严格确定性**——监督者使用 LLM 选择委派。审计/重放更重要时用静态图。

---

## 3. 从零实现

### 第 1 步：定义工作者和轨迹

```python
import threading
import time
from dataclasses import dataclass, field

@dataclass
class WorkerResult:
    sub_question: str
    summary: str
    tokens_spent: int
    wall_time: float

@dataclass
class Trace:
    entries: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, worker_id, event, sub_question=""):
        with self._lock:
            self.entries.append({"worker_id": worker_id, "event": event,
                                  "t": time.time(), "sub_question": sub_question})
```

### 第 2 步：实现并行工作者

```python
def fake_web_fetch(query: str) -> str:
    time.sleep(0.3)
    return f"'{query}' 的摘要：来自 5 个来源的 3 个关键发现。"

class Worker:
    def __init__(self, worker_id, trace):
        self.worker_id = worker_id
        self.trace = trace

    def run(self, sub_question, results, idx):
        self.trace.log(self.worker_id, "start", sub_question)
        summary = fake_web_fetch(sub_question)
        results[idx] = WorkerResult(sub_question, summary, 800, 0.3)
        self.trace.log(self.worker_id, "done", sub_question)
```

### 第 3 步：实现领智能体

```python
class Lead:
    def __init__(self, trace):
        self.trace = trace

    def plan(self, query):
        return [f"{query} -- 历史起源",
                f"{query} -- 2026 年现状",
                f"{query} -- 开放问题"]

    def synthesize(self, query, results):
        ok = [r for r in results if r is not None]
        parts = [f"- {r.sub_question}: {r.summary}" for r in ok]
        return f"对 '{query}' 的回答：\n" + "\n".join(parts)

    def run(self, query):
        t0 = time.time()
        sub_questions = self.plan(query)
        results = [None] * len(sub_questions)
        threads = []
        for i, sq in enumerate(sub_questions):
            w = Worker(i, self.trace)
            th = threading.Thread(target=w.run, args=(sq, results, i))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        synthesis = self.synthesize(query, [r for r in results if r is not None])
        total_tokens = sum(r.tokens_spent for r in results if r is not None) + 1200
        return synthesis, {"wall_clock_seconds": round(time.time() - t0, 3),
                           "total_tokens": total_tokens,
                           "worker_count": len(sub_questions)}
```

### 第 4 步：运行演示

```python
def main():
    trace = Trace()
    lead = Lead(trace)
    answer, stats = lead.run("2023 到 2026 年多智能体系统发生了什么变化？")
    print(f"合成结果: {answer}")
    print(f"统计: {stats}")
    print(f"顺序基线约 0.9s (3 × 0.3s)。并行实际约 0.35s。")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Anthropic Research 系统

| 特性 | 值 |
|------|---|
| 架构 | Claude Opus 4 领 + Sonnet 4 工作者 |
| 性能提升 | 比单 Opus 4 高 +90.2% |
| 关键机制 | 每个子智能体新上下文（BrowseComp 80% 方差来自词元用量） |

### 4.2 部署检查清单

| 检查项 | 说明 |
|--------|------|
| 模型配对 | 领用推理模型（Opus/o3），工作者用更快更便宜模型（Sonnet/o4-mini） |
| 工作者超时 | 超过 2× 中位运行时间则杀死 |
| 每工作者词元上限 | 硬限制（如预期合成输入的 10×） |
| 可观测性 | 追踪领的计划、每个工作者的工具调用、合成 |
| 彩虹部署 | 有状态长期运行智能体需要渐进版本切换 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 规模努力适配查询复杂度 | 简单查询：1 个智能体；复杂查询：10+ 工作者 |
| 宽然后窄 | 先分解为宽子问题，然后按需细化 |
| 工作者超时 | 超过 2× 中位运行时间则杀死 |
| 合成冲突必须显式标记 | 不要静默选一边——标记分歧 |

---

## 6. 常见错误

### 错误 1：领幻觉分解

**现象：** 领生成不分解真实问题的子问题。工作者在错误目标上做精确研究。

**修复：** 领的计划必须覆盖原始查询的所有关键方面。可添加金丝雀工作者检查原始查询。

### 错误 2：工作者过度探索

**现象：** 工作者漂移到子问题之外，返回不相关的信息，污染合成。

**修复：** 每个工作者有明确范围边界。超过范围的结果被丢弃或标记。

### 错误 3：合成冲突静默选边

**现象：** 两个工作者返回矛盾事实。领静默选一边。用户永远不知道有分歧。

**修复：** 领必须标记分歧——"工作者 A 说 X，工作者 B 说 Y"——而不是静默选边。

---

## 7. 面试考点

### Q1：监督者模式的三个胜利机制是什么？（难度：⭐）

**参考答案：**
1. 新鲜上下文——每个工作者获得自己的 200K 窗口，不携带领的历史
2. 专业化提示词——领："分解和合成"；工作者："找到 X 的变化"
3. 并行化——工作者并发运行，墙钟时间 ≈ max(工作者时间) + 计划 + 合成

### Q2：Anthropic Research 系统的关键数据是什么？（难度：⭐⭐）

**参考答案：**
比单 Opus 4 高 +90.2%。BrowseComp 80% 的方差仅由词元用量解释——每个子智能体获得新上下文是主要机制。

多智能体约为单智能体的 15× 词元。仅在任务价值证明成本时使用。

### Q3：监督者模式的三种失败模式是什么？（难度：⭐⭐）

**参考答案：**
1. 领幻觉分解——领生成不分解真实问题的子问题
2. 工作者过度探索——无明确范围边界，工作者漂移到子问题之外
3. 合成冲突——两个工作者返回矛盾事实，领必须标记分歧而非静默选边

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 监督者 | "领智能体" | 计划、分解、合成。不直接做工作 |
| 工作者 | "子智能体" | 由监督者调用的专注智能体，窄范围、独立上下文 |
| 新鲜上下文 | "干净窗口" | 工作者的上下文从其系统提示词和分配的问题开始 |
| 彩虹部署 | "渐进推出" | 有状态长期运行智能体需要渐进版本切换 |
| 词元用量主导 | "上下文是变量" | 80% 的研究评估方差来自总词元用量，而非模型选择 |

---

## 📚 小结

监督者模式：领智能体计划和委派，工作者在并行新上下文中执行，领合成。Anthropic Research 系统比单智能体高 +90.2%，80% 方差来自词元用量。三种失败：领幻觉分解、工作者过度探索、合成冲突。规模努力适配查询复杂度：简单查询用单智能体，复杂查询用 10+ 工作者。

下一课：层级架构及其失败模式——监督者套监督者。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`，然后修改领智能体生成 5 个工作者而非 3 个。观察墙钟效果。工作者数量多少时生成开销超过并行收益？

2. **【实现】** 添加工作者超时：杀死运行超过 0.5 秒的工作者，领用剩余结果合成。需要什么可观测性来知道工作者被切断了？

3. **【实现】** 为领的合成添加冲突检测：如果两个工作者返回矛盾答案，领标记分歧而非选边。

4. **【阅读】** 阅读 Anthropic 的 Research 系统工程文章。列出本玩具演示要在生产中运行需要采用的三个实践。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 监督者演示 | `code/main.py` | 领 + 3 并行工作者 + 合成 |
| 技能提示词 | `outputs/skill-supervisor-designer.md` | 为新研究型智能体系统生成监督者模式设计 |

---

## 📖 参考资料

1. [博客] Anthropic. "How We Built Our Multi-Agent Research System". https://www.anthropic.com/engineering/multi-agent-research-system
2. [文档] LangGraph. https://docs.langchain.com/oss/python/langgraph/workflows-agents
3. [文档] LangGraph Supervisor. https://reference.langchain.com/python/langgraph-supervisor
4. [博客] OpenAI Cookbook. "Orchestrating Agents: Routines and Handoffs". https://developers.openai.com/cookbook/examples/orchestrating_agents

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
