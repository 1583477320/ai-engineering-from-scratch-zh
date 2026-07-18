# 生产扩缩容——队列、检查点与持久化

> 多智能体系统在内存里跑得很好。一旦扩到几千个并发，没有持久化层的架构就像没有备份的服务器——早晚会出事。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 16 · 09（并行群体网络）、阶段 16 · 13（共享记忆与黑板）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 17（基础设施与生产）——持久化执行是多智能体上生产的先决条件

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释持久化执行模式——每个超级步骤（super-step）后写入检查点，崩溃后从最近检查点恢复
- [ ] 比较四种生产架构——LangGraph 运行时、MegaAgent 队列、Temporal 工作流引擎、FastAPI+Postgres 精简方案
- [ ] 实现基于 SQLite 的检查点存储，支持崩溃恢复和 worker 重新调度
- [ ] 理解异步 I/O 对比线程池的 10 倍性能差距——为什么 async 不是优化而是架构选择

---

## 1. 问题

原型阶段的多智能体系统在一个笔记本电脑上跑三个智能体，一切正常。推向生产后：

- 智能体有时需要运行几个小时（深度研究、等待人工审批）
- Worker 崩溃了——重启后状态丢失
- 峰值负载是平均值的 10 倍；需要水平扩展
- 用户按智能体运行次数付费；需要恰好一次语义（exactly-once）

内存事件循环无法满足以上任何一条。你需要一个**持久化执行层**。2026 年的四个可选方案：

1. **具有检查点的工作流引擎**（Temporal、LangGraph 运行时）
2. **消息队列 + 状态存储**（Postgres + SQS/RabbitMQ）
3. **Actor 模型框架**（MegaAgent 的每个智能体一个生产者-消费者队列）
4. **手写的 FastAPI + Postgres**（Bedi 的论证——精简方案能走多远）

---

## 2. 概念

### 2.1 持久化执行模式

持久化执行引擎在每个"步骤"（超级步骤）之后将完整的程序状态持久化。崩溃时：

```
worker 在中间步骤崩溃
  -> 租约（lease）超时
  -> 另一个 worker 接管该 thread_id
  -> 从最后一个检查点恢复
  -> 不产生重复副作用
```

这个模式需要三个条件：

1. **可序列化的状态。** 所有智能体状态必须可持久化。包含活动数据库连接的闭包无法存活。
2. **确定性恢复。** 给定相同的状态和输入，智能体产生相同的动作（或将 LLM 调用委托给外部确定性 oracle）。
3. **幂等的副作用。** 外部调用（工具调用、支付）必须是幂等的，或使用去重键。

LangGraph 在每个超级步骤后写入检查点；Temporal 在每个活动后写入；Restate 使用事件溯源日志。三者实现的是同一个模式。

### 2.2 LangGraph 运行时

每个智能体有一个 `thread_id`；状态是一个类型化字典；每个超级步骤向检查点表写入一行。恢复时，运行时从最后一个检查点重放，而不是从头开始。智能体可以 `interrupt()` 等待人工输入；运行时持久化状态并释放 worker。输入到达时，任意 worker 都可以恢复。

这是 2026 年的参考生产设计。

### 2.3 MegaAgent 的每智能体队列

MegaAgent（arXiv:2408.09955）描述了一个在单个集群中运行数千个并发智能体的架构：

```
智能体 i:
  状态 ∈ {空闲, 处理中, 响应就绪}
  入队   <- 发给智能体 i 的消息
  出队   -> 回复 + 副作用

协调器:
  组内聊天   (同一组的智能体)
  组间管理聊天 (高层路由)
```

两层协调让组内通信密集进行，组间通信保持稀疏——这是在大规模下保持成本线性的模式。

### 2.4 异步 vs 线程

LLM 调用是 I/O 密集型的。一个等待下一个词元的线程 99% 的时间在空转。每个线程占用约 1MB 内存；10,000 个并发调用需要 10GB 内存。

纤程（Python `asyncio`、Go goroutines、Rust `tokio`）在 I/O 上协作式地让出控制权。同样的 10,000 个调用可以轻松放在一个进程中。

例外：CPU 密集型的后处理（嵌入计算、分词器技巧）仍然需要线程或进程。将 I/O 层与 CPU 层分离。

### 2.5 Bedi 的反论

"Scaling Agentic Software"（Ashpreet Bedi, 2026）认为大多数团队在测量负载之前就过度设计了。务实的默认选择：

- FastAPI + Postgres
- 每个智能体运行是一行记录；状态原地更新，使用乐观并发控制
- 通过 `pg_notify` 或简单的 Celery worker 运行后台任务
- 重试策略放在应用代码中

对于负载低于约 100 个并发智能体运行的场景，这通常就足够了。只有在测量到失败时才升级。

### 2.6 恰好一次语义

对于付费的智能体运行，需要"有效恰好一次"（至少一次交付 + 幂等消费）：

- **每运行去重键。** 在每次副作用调用中包含它。
- **发件箱模式（Outbox）。** 副作用先写入表，然后由单独的进程执行。两步都是幂等的。
- **补偿事务。** 当副作用成功但其追踪写入失败时，调度补偿。

### 2.7 彩虹部署

Anthropic 的多智能体研究系统使用"彩虹部署"——多个版本的智能体运行时同时运行，这样长时间运行的智能体不必在每次代码部署时被终止。在流量的一个切片上尝试新版本；当旧版本的智能体完成时退役。

这对于长时间运行的有状态系统来说是标准的；2026 年的适配是智能体可能运行数小时，部署周期必须适应。

### 2.8 生产检查清单

- 持久化状态（检查点、快照、或发件箱 + 可重放日志）
- 幂等的副作用
- LLM 调用的异步 I/O 层
- 至少一次交付 + 去重
- 有状态工作负载的彩虹/金丝雀部署
- 可观测性：每智能体追踪、超级步骤审计、重试计数

---

## 3. 从零实现

### 第 1 步：基于 SQLite 的检查点存储

```python
import sqlite3
import json
import threading


class CheckpointStore:
    """基于 SQLite 的检查点日志，以 thread_id 为键。"""

    def __init__(self, db_path: str = "checkpoints.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT,
                step INTEGER,
                state TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, step)
            )
        """)
        self._conn.commit()

    def save(self, thread_id: str, step: int, state: dict):
        """保存检查点。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, step, state) VALUES (?, ?, ?)",
                (thread_id, step, json.dumps(state)),
            )
            self._conn.commit()

    def load_latest(self, thread_id: str) -> tuple[dict, int]:
        """加载最新检查点。"""
        cursor = self._conn.execute(
            "SELECT state, step FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
            (thread_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return {}, 0
        return json.loads(row[0]), row[1]
```

### 第 2 步：带检查点的运行器——崩溃恢复演示

```python
import time


class CheckpointRunner:
    """带检查点的智能体运行器——支持崩溃恢复。"""

    def __init__(self, store: CheckpointStore):
        self.store = store

    def run(self, thread_id: str, task: list[str]):
        """执行任务序列，每步后写入检查点。"""
        state, step = self.store.load_latest(thread_id)

        print(f"[恢复] thread_id={thread_id}, 从步骤 {step} 恢复")

        for i in range(step, len(task)):
            step_name = task[i]

            # 模拟执行
            state["current_step"] = step_name
            state["progress"] = i + 1

            # 写入检查点
            self.store.save(thread_id, i + 1, state)
            print(f"[步骤 {i+1}/{len(task)}] {step_name} → 检查点已保存")

            # 模拟 crash（在第 2 步后）
            if i == 1 and not state.get("recovered"):
                print("  ✗ CRASH！worker 崩溃")
                state["recovered"] = True
                raise RuntimeError("模拟崩溃")

        print(f"[完成] 全部 {len(task)} 个步骤执行成功")
        return state


# 演示
if __name__ == "__main__":
    store = CheckpointStore(":memory:")
    runner = CheckpointRunner(store)

    task = ["数据采集", "数据清洗", "数据分析", "报告撰写"]

    # 第一次运行——会在步骤 2 后崩溃
    print("=== 第一次运行 ===")
    try:
        runner.run("thread-001", task)
    except RuntimeError as e:
        print(f"  捕获到崩溃: {e}")

    # 第二次运行——从检查点恢复
    print("\n=== 第二次运行 (恢复) ===")
    result = runner.run("thread-001", task)
    print(f"  最终状态: {result}")
```

执行预期输出：

```text
=== 第一次运行 ===
[恢复] thread_id=thread-001, 从步骤 0 恢复
[步骤 1/4] 数据采集 → 检查点已保存
[步骤 2/4] 数据清洗 → 检查点已保存
  ✗ CRASH！worker 崩溃
  捕获到崩溃: 模拟崩溃

=== 第二次运行 (恢复) ===
[恢复] thread_id=thread-001, 从步骤 2 恢复
[步骤 3/4] 数据分析 → 检查点已保存
[步骤 4/4] 报告撰写 → 检查点已保存
[完成] 全部 4 个步骤执行成功
```

### 第 3 步：每智能体工作队列

```python
from enum import Enum
from queue import Queue


class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    RESPONSE = "response"


class AgentQueue:
    """每智能体的工作队列——带状态转换的生产者-消费者模式。"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.in_queue: Queue[dict] = Queue()
        self.out_queue: Queue[dict] = Queue()

    def submit(self, task: dict):
        """提交任务到入队。"""
        self.in_queue.put(task)
        self.state = AgentState.IDLE

    def process_one(self) -> dict:
        """处理一个任务。"""
        if self.in_queue.empty():
            return {"agent": self.agent_id, "status": "no_task"}

        self.state = AgentState.PROCESSING
        task = self.in_queue.get()
        # 模拟处理
        result = {"agent": self.agent_id, "task": task, "output": f"处理结果: {task}"}
        self.out_queue.put(result)
        self.state = AgentState.RESPONSE
        return result
```

### 第 4 步：异步 vs 线程对比演示

```python
import asyncio
import threading
import time


async def mock_llm_call_async(delay: float):
    """模拟异步 LLM 调用。"""
    await asyncio.sleep(delay)
    return delay


def mock_llm_call_sync(delay: float):
    """模拟同步 LLM 调用。"""
    time.sleep(delay)
    return delay


async def demo_async_vs_threads(count: int = 500):
    """对比异步和线程在大量并发 I/O 下的表现。"""
    delay = 0.01  # 10ms 模拟 LLM 延迟

    # 异步
    start = time.time()
    results = await asyncio.gather(*[
        mock_llm_call_async(delay) for _ in range(count)
    ])
    async_time = time.time() - start

    # 线程
    start = time.time()
    threads = []
    for _ in range(count):
        t = threading.Thread(target=mock_llm_call_sync, args=(delay,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    thread_time = time.time() - start

    print(f"异步 {count} 次调用: {async_time:.3f}s")
    print(f"线程 {count} 次调用: {thread_time:.3f}s")
    print(f"差距: {thread_time / async_time:.1f}x")

    return {"async": async_time, "thread": thread_time}


if __name__ == "__main__":
    asyncio.run(demo_async_vs_threads(200))
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 LangGraph 运行时

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint import MemorySaver

graph = StateGraph(MessagesState)
# ... 添加节点和边 ...
app = graph.compile(checkpointer=MemorySaver())

# 持久化运行
result = app.invoke(
    {"messages": [{"role": "user", "content": "分析数据"}]},
    config={"configurable": {"thread_id": "thread-001"}},
)
```

LangGraph 原生支持线程级检查点持久化、崩溃恢复、人工-回环中断、彩虹部署。

### 4.2 架构选型对比

| 方案 | 复杂度 | 容错 | 规模上限 | 适用场景 |
|---|---|---|---|---|
| FastAPI + Postgres | 低 | 中 | ~100 并发 | 起步阶段，负载尚不明确 |
| LangGraph 运行时 | 中 | 高 | 企业级 | 需要持久化 + 人工-回环 |
| Temporal | 高 | 高 | 企业级 | 需要复杂补偿/重试策略 |
| MegaAgent 队列 | 中 | 中 | 数千并发 | 需要两层协调的大规模部署 |
| 自定义 + Redis/RabbitMQ | 中 | 中 | 数千并发 | 已有消息中间件基础设施 |

---

## 5. 工程最佳实践

### 5.1 Bedi 规则——启动前不要过度设计

- FastAPI + Postgres 作为默认起步。
- 只有当你测量到它在某个具体问题上失败时，才升级到持久化执行引擎。
- 过早采用会浪费大量精力在对你并没有好处的仪式上。

### 5.2 每步都做可观测性

部署前的必装设备：

- 每次运行延迟直方图、每步耗时、重试次数、失败分类。
- 每智能体追踪：一个 run_id 下所有消息的完整传播链路。
- 超级步骤审计：谁在什么时候做了什么，状态变化记录。

### 5.3 幂等性是第一优先级

- 每个外部调用（支付、API、数据库写入）必须可重放而不产生重复效果。
- 发件箱模式：先写入表，再执行；执行时标记完成。
- 补偿事务：副作用的追踪写入失败时，调度补偿操作。

### 5.4 中文场景特别建议

- **国内云环境下的 Postgres 性能。** 如果使用阿里云 RDS 或腾讯云 CDB，Postgres 连接池大小需要根据实例规格调整。默认 100 连接在 2C4G 实例上可能撑爆
- **LangGraph 运行时在国内的部署。** LangGraph Cloud 目前不支持中国大陆区域。自托管 LangGraph 时需要注意 Postgres 版本的兼容性（需要 14+）
- **国产队列的替代方案。** RabbitMQ 在国内有阿里云 AMQP 托管服务，也可以考虑 RocketMQ（国内广泛使用，社区活跃）

---

## 6. 常见错误

### 错误 1：没有检查点导致崩溃后全丢

**现象：** Worker 在运行 30 分钟后崩溃。重启后，智能体从第 1 步重新开始——之前的所有工作丢失。

**原因：** 智能体状态保存在内存中，没有持久化检查点。

**修复：** 每执行完一个可重复步骤后，将状态写入检查点存储。

### 错误 2：线程池撑爆内存

**现象：** 部署上线后，服务的内存曲线直线上升。1000 个并发请求时 OOM 崩溃。

**原因：** 每个 LLM 调用使用一个线程，线程栈占用约 1MB。1000 个并发 = 1GB 内存。

**修复：** 使用 asyncio（纤程）替代线程池处理 LLM 调用。

### 错误 3：重试不幂等导致重复收费

**现象：** 网络抖动导致支付调用超时（实际已成功）。重试后扣了两次钱。

**原因：** 支付调用不是幂等的。重试操作重复执行了副作用。

**修复：** 每次调用携带唯一去重键；服务端检查并跳过已处理的请求。

---

## 7. 面试考点

### Q1：持久化执行需要哪三个条件？为什么 LLM 调用的确定性恢复是一个问题？（难度：⭐⭐⭐）

**参考答案：**
三个条件：可序列化的状态、确定性恢复、幂等的副作用。LLM 调用的非确定性（temperature > 0 时每次输出不同）使得"确定性恢复"无法自然实现。解决思路：将 LLM 调用的输入/输出记录到日志中，恢复时不重新调用 LLM 而是回放日志中记录的输出。这需要 LLM 调用的输出是确定性的（temperature = 0），或在调用日志中缓存结果。

### Q2：什么时候应该使用 FastAPI + Postgres 的精简方案，什么时候应该上 Temporal/LangGraph？（难度：⭐⭐）

**参考答案：**
Bedi 的规则：从精简方案起步，测量到失败再升级。具体指标：并发智能体运行数 < 100、单次运行时长 < 5 分钟、不需要人工-回环、重试逻辑简单，就足够用精简方案。反之，当遇到以下问题时升级：需要长时间等待人工审批（数小时）、需要跨区域协调、需要复杂的补偿策略、状态量 > 1GB。

### Q3：什么是彩虹部署？为什么多智能体系统特别需要它？（难度：⭐⭐）

**参考答案：**
彩虹部署是多个版本的智能体运行时同时运行的部署模式。新版本处理新的 thread_id，旧版本继续处理已在运行的任务直到完成。多智能体系统特别需要它，因为智能体任务可能运行数小时——如果每次部署都中断它们，用户会丢失进度和上下文。这是有状态系统的标准实践，在 LLM 智能体场景下尤其重要。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 持久化执行 | "保存程序状态" | 引擎在每个超级步骤后写入状态；崩溃后确定性恢复 |
| 超级步骤 | "事务边界" | 检查点之间的工作单元。LangGraph 术语 |
| 幂等性 | "可安全重试" | 重复执行副作用产生与单次执行相同的结果 |
| 发件箱模式 | "解耦副作用" | 先写入意图表，再执行，完成后标记 |
| 彩虹部署 | "多版本共存" | 长时运行期间多个运行时版本同时工作 |
| 纤程 | "协作式并发" | 用户态并发；比线程轻量得多，适合 I/O 密集型负载 |
| 检查点 | "状态快照" | 在超级步骤边界序列化的状态，用于恢复 |

---

## 📚 小结

多智能体系统的生产扩缩容有三个关键工程决策：**持久化执行**（检查点/快照/发件箱）、**异步 I/O 层**（纤程 vs 线程）、**幂等副作用**（去重键/发件箱模式/补偿事务）。Bedi 的经验法则最有实践价值：用 FastAPI + Postgres 起步，测量到失败再升级——大多数团队过早采用了他们并不需要的复杂度。

下一课我们将讨论多智能体系统的故障模式——系统不是"会不会失败"而是"会以什么方式失败"。

---

## ✏️ 练习

1. 运行 `code/main.py`。确认检查点恢复在模拟崩溃后成功工作；对比异步 vs 线程在高并发下的差异。
2. 实现一个**发件箱表**：每次工具调用先写入发件箱，然后由单独的 task 执行。通过调用两次来验证幂等性。
3. 模拟**彩虹部署**：两个并发的运行时版本；将一半的 thread_id 路由到每个版本；确认旧版本上的运行不会被中断。
4. 阅读 LangGraph 运行时文档。找出哪些特性在一个手写的 FastAPI + Postgres 版本中最难复现。这是采用的充分理由还是可以继续推迟？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 检查点存储 | `code/main.py` | SQLite 检查点存储 + 崩溃恢复演示 |
| 扩缩容选型建议 | `outputs/skill-scaling-advisor.md` | 根据负载和需求推荐架构 |
