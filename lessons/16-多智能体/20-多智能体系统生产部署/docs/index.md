# 多智能体系统生产部署

> 多智能体系统在 Jupyter Notebook 里跑得挺好。一旦部署到生产环境，它会用一百种方式告诉你"我还没准备好"。故障隔离、成本控制、延迟管理——这三个问题不解决，多智能体系统就是实验室玩具。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 16 · 19（多智能体协商与拍卖机制）、阶段 17（基础设施与生产）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 16 · 21（多智能体评估与基准测试）— 生产部署后需要评估系统在真实负载下的表现

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计多智能体系统的故障隔离机制——一个智能体崩溃不应拖垮整个系统
- [ ] 实现智能体调用的成本追踪和预算控制——防止单次请求烧掉一个月预算
- [ ] 构建多智能体流水线的延迟监控——定位瓶颈智能体，优化端到端响应时间
- [ ] 在 LangGraph 中实现生产级的重试、超时和回退策略

---

## 1. 问题

你的多智能体系统在演示时完美运行——数据分析师、报告撰写者、图表生成器三个智能体协作生成了一份漂亮的市场分析报告。老板说："下周上线。"

上线第一天，三个问题同时爆发。第一，数据分析师调用的 API 限流了，整个系统卡死——没有超时机制，后续请求全部排队。第二，一个用户请求触发了 15 轮智能体对话（因为智能体之间反复确认细节），单次请求消耗了 20000 个词元——按 GPT-4o 的价格，这一条请求花了 0.6 美元。第三，图表生成器返回了格式错误的 JSON，下游智能体无法解析——没有错误处理，整个流水线崩溃。

**多智能体系统的生产部署，本质上是解决三个问题：故障隔离、成本控制、延迟管理。** 任何一个没做好，系统就不可用。

---

## 2. 概念

### 2.1 故障隔离

单智能体系统只有一个故障点。多智能体系统有 N 个故障点——每个智能体的 API 调用都可能失败、超时、返回异常结果。更糟糕的是，智能体之间的依赖关系形成了**故障传播链**——上游智能体的失败会级联到所有下游智能体。

```
故障传播链示例：

数据分析师 (API 限流) ──失败──► 报告撰写者 (无数据可写) ──失败──► 图表生成器 (无数据可画)
         │                              │                              │
    超时等待                          返回空结果                       崩溃
         │                              │                              │
    用户等待 30 秒                   用户看到空报告                  用户看到错误页面
```

**隔离策略：**

1. **超时强制断开。** 每个智能体调用设置硬超时（如 30 秒）。超时 = 该智能体不可用，不等待
2. **回退到降级。** 智能体失败时，用预设的默认值或简化逻辑代替。数据分析师失败→用缓存的上周数据
3. **断路器模式。** 某个智能体连续失败 N 次后，暂时跳过它（断路），直接用替代方案。5 分钟后半开——试探性调用一次，成功则恢复，失败则继续断路

### 2.2 成本控制

多智能体系统的成本是单智能体的 N 倍——每个智能体调用都是一次 API 请求。更隐蔽的成本是**智能体对话膨胀**——智能体之间互相传递消息，每条消息都消耗词元。

```
成本放大效应：

用户请求: 500 词元
    │
    ▼
规划智能体: 500 输入 + 200 输出 = 700 词元
    │
    ▼
数据智能体: 700 输入 + 500 输出 = 1200 词元（接收规划结果 + 自己的分析）
    │
    ▼
写作智能体: 1200 输入 + 800 输出 = 2000 词元（接收数据 + 自己的撰写）
    │
    ▼
总消耗: 3900 词元（用户的 500 词元被放大了 7.8 倍）
```

**成本控制策略：**

1. **全局预算上限。** 每个用户请求设置总词元预算（如 10000 词元）。超过预算 = 强制终止
2. **逐智能体配额。** 每个智能体有独立的词元配额。数据分析师最多 3000 词元，写作智能体最多 5000 词元
3. **对话轮次限制。** 智能体之间的对话最多 5 轮。超过 = 强制收敛，用当前最好的结果
4. **成本追踪。** 每次 API 调用记录实际消耗和费用，累积到用户/请求维度

### 2.3 延迟管理

多智能体系统的延迟是各智能体延迟的叠加。串行执行时，总延迟 = 各智能体延迟之和。并行执行时，总延迟 = 最慢智能体的延迟。

```
串行执行（总延迟 = 5+8+3 = 16 秒）：
  数据分析 ████████ 5s
                  报告撰写 ████████████ 8s
                                          图表生成 ████ 3s

并行执行（总延迟 = max(5,8,3) = 8 秒）：
  数据分析 ████████ 5s
  报告撰写 ████████████ 8s
  图表生成 ████ 3s
```

**延迟优化策略：**

1. **并行化。** 无依赖的智能体并行调用（LangGraph 的 `Send` API）
2. **流式输出。** 智能体生成内容时流式返回，用户看到部分结果而非等待全部完成
3. **缓存。** 相同输入的智能体调用缓存结果（智能体的输出通常是确定性或近似确定性的）
4. **超时降级。** 慢智能体超时后，用快速的替代方案（小模型或规则引擎）

---

## 3. 从零实现

### 第 1 步：带超时和重试的智能体调用

```python
import time
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """单个智能体的配置。"""
    name: str
    timeout: float = 30.0       # 单次调用超时（秒）
    max_retries: int = 2        # 最大重试次数
    max_tokens: int = 3000      # 词元预算上限


class ResilientAgentRunner:
    """带故障隔离的智能体运行器。"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.consecutive_failures = 0  # 连续失败次数
        self.circuit_open = False      # 断路器状态
        self.circuit_reset_time = 0    # 断路器重置时间

    def run(self, prompt: str, llm_call) -> dict:
        """执行智能体调用，带超时、重试和断路器。"""
        # 断路器检查：连续失败 3 次后跳过 60 秒
        if self.circuit_open:
            if time.time() < self.circuit_reset_time:
                return {"status": "circuit_open", "output": None}
            # 半开状态：尝试一次
            self.circuit_open = False

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start = time.time()
                result = llm_call(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                elapsed = time.time() - start

                # 成功：重置断路器计数
                self.consecutive_failures = 0
                return {
                    "status": "success",
                    "output": result,
                    "latency": elapsed,
                    "attempt": attempt + 1,
                }
            except TimeoutError:
                last_error = "超时"
            except Exception as e:
                last_error = str(e)

        # 所有重试都失败：触发断路器
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.circuit_open = True
            self.circuit_reset_time = time.time() + 60  # 60 秒后半开

        return {"status": "failed", "output": None, "error": last_error}
```

### 第 2 步：成本追踪器

```python
from collections import defaultdict


class CostTracker:
    """追踪每个请求的词元消耗和费用。"""

    # GPT-4o 的价格（美元/词元）
    PRICE_PER_TOKEN = {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000}

    def __init__(self, budget_limit: float = 1.0):
        self.budget_limit = budget_limit  # 单请求预算上限（美元）
        self.records = []                 # 所有调用记录
        self.total_cost = 0.0

    def record(self, agent_name: str, input_tokens: int, output_tokens: int):
        """记录一次 API 调用的词元消耗。"""
        cost = (
            input_tokens * self.PRICE_PER_TOKEN["input"]
            + output_tokens * self.PRICE_PER_TOKEN["output"]
        )
        self.total_cost += cost
        self.records.append({
            "agent": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative": self.total_cost,
        })

        if self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"预算超限: {self.total_cost:.4f} > {self.budget_limit:.4f} 美元"
            )

    def summary(self) -> dict:
        """返回成本摘要。"""
        by_agent = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        for r in self.records:
            by_agent[r["agent"]]["tokens"] += r["input_tokens"] + r["output_tokens"]
            by_agent[r["agent"]]["cost"] += r["cost"]
        return {
            "total_cost": self.total_cost,
            "budget_remaining": self.budget_limit - self.total_cost,
            "by_agent": dict(by_agent),
        }


class BudgetExceededError(Exception):
    """预算超限异常。"""
    pass
```

### 第 3 步：端到端多智能体流水线

```python
def run_pipeline(user_request: str, cost_tracker: CostTracker):
    """带故障隔离和成本控制的多智能体流水线。"""

    # 智能体配置
    analyst_config = AgentConfig(name="数据分析师", timeout=30, max_tokens=3000)
    writer_config = AgentConfig(name="报告撰写者", timeout=45, max_tokens=5000)

    analyst_runner = ResilientAgentRunner(analyst_config)
    writer_runner = ResilientAgentRunner(writer_config)

    # 第 1 步：数据分析
    analysis_prompt = f"分析以下需求的数据维度：{user_request}"
    analysis_result = analyst_runner.run(analysis_prompt, mock_llm_call)

    if analysis_result["status"] == "failed":
        # 降级：用默认分析模板
        analysis_result["output"] = "标准分析模板：趋势分析、对比分析、异常检测"

    # 第 2 步：报告撰写（依赖第 1 步的输出）
    report_prompt = f"基于以下分析结果撰写报告：{analysis_result['output']}"
    report_result = writer_runner.run(report_prompt, mock_llm_call)

    if report_result["status"] == "circuit_open":
        return {"status": "degraded", "message": "报告撰写智能体暂时不可用"}

    # 记录成本
    if analysis_result["status"] == "success":
        cost_tracker.record("数据分析师", 500, 300)
    if report_result["status"] == "success":
        cost_tracker.record("报告撰写者", 800, 600)

    return {"status": "success", "report": report_result.get("output", "")}


# 模拟 LLM 调用（教学用）
def mock_llm_call(prompt, max_tokens=1000, timeout=30):
    """模拟 LLM 调用，实际使用时替换为真实 API。"""
    time.sleep(0.1)  # 模拟网络延迟
    return f"[模拟输出] 基于 '{prompt[:30]}...' 生成的内容"
```

---

## 4. 工业工具

### 4.1 LangGraph 生产配置

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.pregel import RetryPolicy
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", request_timeout=30)

graph = StateGraph(MessagesState)

# 添加节点时配置重试策略
graph.add_node(
    "analyst",
    analyst_node,
    retry=RetryPolicy(max_attempts=2, retry_on=TimeoutError),
)
graph.add_node("writer", writer_node, retry=RetryPolicy(max_attempts=2))

graph.add_edge(START, "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)

# 编译时配置容错
app = graph.compile()
```

LangGraph 原生支持 `RetryPolicy`——在节点级别配置重试策略，无需手动实现断路器。

### 4.2 LangSmith 追踪

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# 所有 LangChain/LangGraph 调用自动追踪
result = app.invoke({"messages": [{"role": "user", "content": "分析市场"}]})

# 在 LangSmith Dashboard 中查看：
# - 每个智能体的延迟
# - 每次调用的词元消耗
# - 调用链路的完整追踪
# - 错误和重试记录
```

### 4.3 部署架构对比

| 组件 | 教学环境 | 生产环境 |
|---|---|---|
| 智能体编排 | 本地 Python 进程 | LangGraph Cloud / 自托管 LangGraph |
| LLM 调用 | 直接调用 API | 通过代理（LiteLLM）统一管理多提供商 |
| 追踪 | print() | LangSmith / LangFuse / Phoenix |
| 成本控制 | 手动检查 | 代理层自动限额 + 告警 |
| 故障恢复 | 手动重启 | Kubernetes 健康检查 + 自动重启 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

OpenAI 的 Assistants API 内置了运行管理——每次对话有 `run` 对象追踪状态、成本、错误。Anthropic 的 Claude API 支持 `max_tokens` 和 `stop_sequences` 控制输出长度。这些是单智能体层面的成本控制。

### 5.2 LLM 时代什么变了？

多智能体系统的成本控制比单智能体复杂一个数量级。单智能体：一次 API 调用 = 一次成本。多智能体：一次用户请求 = N 次 API 调用 × M 轮对话 = N × M 次成本。**成本乘数效应是多智能体部署的核心挑战。**

### 5.3 什么没变？

故障隔离的基本模式（超时、重试、断路器）来自分布式系统领域，与 LLM 无关。这些模式在微服务架构中已经被验证了 15 年。**多智能体系统的生产部署，本质上是微服务架构的 LLM 版本。**

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的 GPTs 功能时，系统内部就在运行多智能体协作——主模型决定何时调用代码解释器、联网搜索、图像生成。每个工具调用都有超时和错误处理。如果你的 GPT 响应突然变慢，很可能是因为某个工具调用超时了。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 快速原型 | 直接调用 API + print 追踪 | 够用就行 |
| 中等规模 | LangGraph + LangSmith | 开箱即用的追踪和调试 |
| 大规模生产 | 自托管 LangGraph + LiteLLM 代理 | 统一管理多提供商 + 自定义限额 |
| 超大规模 | Kubernetes + 自定义编排层 | 完全控制，但维护成本高 |

### 6.2 中文场景特别建议

- **中文 LLM 的 API 限流更严格。** 国内提供商（百度文心、阿里通义、讯飞星火）的 QPS 限制通常低于 OpenAI——多智能体系统更容易触发限流。建议增加退避时间（从 1 秒开始指数退避）
- **中文智能体对话的词元消耗更高。** 同等信息量，中文需要更多词元。成本预算要按英文的 1.5-2 倍设置
- **使用国产模型时注意并发限制。** 部分国产模型的并发连接数限制较低（如 5-10 个）——多智能体并行调用时容易触发。建议串行化或降低并行度

### 6.3 踩坑经验

- **忘记设置全局超时。** 单个智能体没有超时→整个流水线可能挂起 5 分钟（API 的默认超时）。解决方案：在 HTTP 客户端和智能体运行器两层都设置超时
- **重试风暴。** 多个智能体同时失败→同时重试→同时冲击 API→全部失败→同时重试……指数退避 + 随机抖动是必须的
- **成本追踪遗漏。** 只追踪了"成功"的调用，失败的重试调用没有计入成本。失败的调用也消耗了词元——必须全部追踪
- **缓存键设计不当。** 用用户原始请求做缓存键→同一请求的微小差异导致缓存未命中。解决方案：对输入做归一化（去空白、统一大小写）后再做缓存键

---

## 7. 常见错误

### 错误 1：没有断路器导致级联失败

**现象：** 一个智能体的 API 限流后，所有请求都卡在这个智能体的重试上，队列堆积，整个系统响应时间从 5 秒飙升到 2 分钟。

**原因：** 没有断路器。每次调用失败后重试 3 次，每次重试等待 2 秒。10 个并发请求 × 3 次重试 × 2 秒 = 60 秒的队列。

**修复：**

```python
# ❌ 无限重试
for attempt in range(5):
    try:
        return call_agent(prompt)
    except Exception:
        time.sleep(2)  # 固定退避，没有断路

# ✓ 断路器 + 指数退避 + 随机抖动
import random

class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.open_since = 0

    def call(self, func, *args, **kwargs):
        if self.failures >= self.threshold:
            if time.time() - self.open_since < self.reset_timeout:
                raise RuntimeError("断路器打开，跳过调用")
            self.failures = 0  # 半开：重置计数

        try:
            result = func(*args, **kwargs)
            self.failures = 0  # 成功：重置
            return result
        except Exception as e:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open_since = time.time()
            raise
```

### 错误 2：对话轮次无限制导致成本爆炸

**现象：** 一个用户请求消耗了 50000 词元，费用 1.5 美元。查看日志发现智能体之间进行了 20 轮对话——反复确认细节、互相质疑、要求澄清。

**原因：** 没有对话轮次限制。LLM 智能体倾向于"完美主义"——总觉得信息不够，想要更多轮次确认。

**修复：**

```python
# ❌ 无限制
while not task_complete:
    response = agent.chat(message)

# ✓ 硬性轮次上限
MAX_TURNS = 5
for turn in range(MAX_TURNS):
    response = agent.chat(message)
    if is_task_complete(response):
        break
    message = response
else:
    # 达到上限：用当前最好的结果返回
    response = force_conclude(agent, message)
```

### 错误 3：并行调用时忽略速率限制

**现象：** 5 个智能体并行调用同一个 API 提供商，全部返回 429 Too Many Requests。

**原因：** 并行调用没有考虑 API 的速率限制。5 个并发请求同时冲击 API，超过了提供商的 QPS 限制。

**修复：**

```python
# ❌ 无限制并行
results = await asyncio.gather(*[agent.run(task) for agent in agents])

# ✓ 带信号量的并行
import asyncio

semaphore = asyncio.Semaphore(3)  # 最多 3 个并发

async def limited_run(agent, task):
    async with semaphore:
        return await agent.run(task)

results = await asyncio.gather(*[limited_run(a, t) for a, t in zip(agents, tasks)])
```

---

## 8. 面试考点

### Q1：多智能体系统的故障隔离和微服务的故障隔离有什么异同？（难度：⭐⭐）

**参考答案：**
相同点：都使用超时、重试、断路器、降级这四种基本模式。不同点有三个。第一，故障模式不同——微服务的故障通常是网络超时或服务不可用，多智能体的故障还包括"LLM 返回格式错误"或"幻觉输出"——这种故障无法用断路器处理，需要输出验证。第二，依赖关系不同——微服务的依赖是静态的（API 网关配置），多智能体的依赖是动态的（LLM 决定调用哪个智能体）。第三，成本影响不同——微服务的重试成本可以忽略（网络请求），多智能体的重试成本很高（每次调用都是 LLM API 调用，按词元计费）。

### Q2：如何设计一个多智能体系统的成本控制系统？（难度：⭐⭐⭐）

**参考答案：**
分三层设计。第一层，**请求级预算**——每个用户请求设置总词元上限（如 10000 词元），超过即终止。第二层，**智能体级配额**——每个智能体有独立的词元配额（如数据分析师 3000、写作智能体 5000），防止单个智能体消耗过多。第三层，**全局预算**——每个用户/租户有月度预算上限，达到后降级或拒绝服务。三层之间是包含关系：请求级 < 智能体级 < 全局级。实现上，用一个中央 CostTracker 在每次 API 调用后检查预算，超限时抛出 BudgetExceededError，由流水线的异常处理器决定降级策略。

### Q3：你的多智能体系统在高峰期响应时间从 5 秒飙升到 30 秒。如何定位瓶颈？（难度：⭐⭐⭐）

**参考答案：**
四步定位。第一，**查看 LangSmith/追踪系统**——看每个智能体的延迟分布，找到最慢的那个。第二，**检查并行度**——本该并行的智能体是否被串行化了？（常见原因：LangGraph 的边配置错误）第三，**检查 API 限流**——是否某个智能体因为 429 错误在反复重试？查看重试日志。第四，**检查上下文膨胀**——智能体的输入是否随对话轮次指数增长？（每轮对话都携带完整历史→词元消耗翻倍→延迟翻倍）。最常见的瓶颈是第四点：解决方案是截断历史——只保留最近 3 轮对话，更早的历史用摘要替代。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 断路器 | "自动跳过坏的" | 连续失败 N 次后暂时跳过调用，定期半开试探。防止故障级联 |
| 降级 | "用差一点的方案" | 智能体失败时用缓存/规则引擎/小模型替代。保证系统可用性 |
| 成本放大 | "多智能体烧钱" | 用户请求的词元消耗被智能体对话放大 N 倍。一次 500 词元的请求可能消耗 5000 词元 |
| 对话膨胀 | "上下文越来越长" | 智能体每轮对话携带完整历史→输入词元指数增长→延迟和成本飙升 |
| 重试风暴 | "大家一起重试" | 多个智能体同时失败→同时重试→同时冲击 API→全部失败的恶性循环 |
| 速率限制 | "API 限流" | 提供商对并发请求数或每分钟请求数的限制。多智能体并行调用容易触发 |
| 请求级预算 | "一次请求最多花多少钱" | 每个用户请求的总词元/费用上限，超过即终止或降级 |

---

## 📚 小结

多智能体系统的生产部署有三个核心挑战：故障隔离（断路器+降级）、成本控制（三级预算体系）、延迟管理（并行化+缓存+流式输出）。每个智能体都是一个潜在的故障点和成本中心。LangGraph 提供了开箱即用的重试策略，LangSmith 提供了全链路追踪。记住：多智能体系统的成本是单智能体的 N 倍——没有预算控制的多智能体系统，在生产环境中就是一颗定时炸弹。

下一课我们将讨论多智能体系统的评估——如何衡量多个智能体协作的效果，而不仅仅是单个智能体的表现。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么多智能体系统的成本是单智能体的 N 倍而不是 1 倍？用一个具体例子说明"成本放大效应"。

2. 【实现】实现一个带断路器的智能体运行器。要求：连续失败 3 次后断路 60 秒，断路期间直接返回降级结果。

3. 【实验】模拟 5 个智能体的流水线，其中 1 个智能体有 50% 的概率超时。分别测试：无重试、固定重试（3 次）、指数退避重试的端到端延迟和成功率。

4. 【思考】如果你的多智能体系统需要处理每秒 100 个并发请求，但 LLM API 的 QPS 限制是 50，你会如何设计架构来应对？画出架构图。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 容错智能体运行器 | `code/resilient_runner.py` | 带超时、重试、断路器的智能体调用封装 |
| 成本追踪器 | `code/cost_tracker.py` | 多维度成本追踪，支持请求级/智能体级/全局预算 |
| 生产流水线示例 | `code/production_pipeline.py` | 完整的生产级多智能体流水线，含故障隔离和成本控制 |

---

## 📖 参考资料

1. [论文] Nygard, M. T. "Release It! Design and Deploy Production-Ready Software". Pragmatic Bookshelf, 2018. — 断路器、舱壁、超时等生产模式的经典著作
2. [官方文档] LangGraph — Persistence and Fault Tolerance. https://langchain-ai.github.io/langgraph/ — LangGraph 的容错机制文档
3. [官方文档] LangSmith. https://docs.smith.langchain.com/ — 全链路追踪和成本监控
4. [官方文档] LiteLLM. https://docs.litellm.ai/ — 统一多提供商的 LLM 代理
5. [论文] Smith, R. G. "The Contract Net Protocol". IEEE, 1980. — 故障隔离在分布式系统中的理论基础

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
