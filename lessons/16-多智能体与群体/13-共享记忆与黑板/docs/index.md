# 共享记忆与黑板模式

> 2026 年多智能体系统中两种方法共存：**消息池**（每个人看到每个人的消息，如 AutoGen GroupChat 或 MetaGPT）和**带订阅的黑板**（智能体订阅相关事件，如 CA-MCP 或 Matrix）。两者都是多智能体系统中唯一有状态的部分——这意味着两者都是有趣的 bug 存在的地方。参考失败模式是**记忆投毒**：一个智能体幻觉一个"事实"，其他智能体将其视为已验证，准确率以比即时崩溃更难调试的方式逐渐衰减。本课从零构建两种结构，注入投毒攻击，展示三种在生产中真正有效的缓解措施。

**类型：** 概念课 + 实现课
**语言：** Python（标准库，`threading`）
**前置知识：** 阶段 16 · 04（原语模型）、阶段 16 · 09（并行群体网络）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分消息池（全历史）和带订阅的黑板（主题路由）——各自的缩放特性
- [ ] 实现记忆投毒场景——一个智能体的幻觉如何通过共享状态传播到下游智能体
- [ ] 实现三种缓解：溯源记录、追加写入、不可写验证者
- [ ] 理解为什么记忆投毒是结构性的——任何没有溯源和独立验证者的共享记忆设计都会出现

---

## 1. 问题

多智能体系统需要一个共享事实的地方。"在消息中传递一切"重新发明了带额外复制的共享状态。"给每个人一个全局日志"但全局日志无限增长且容易投毒。"给每个智能体一个限定视图"但需要预设 Schema。

当一个智能体幻觉并将幻觉写入共享状态时，每个读取该状态的下游智能体都把幻觉当作事实。等到人类注意到时，推理链已经深入五步，根因是第三条消息。

**这是记忆投毒。** 它是 MAST 分类法（Cemri 等人，arXiv:2503.13657）中第二广泛记录的失败家族——没有溯源和不可写验证者的任何共享记忆设计最终都会出现它。

---

## 2. 概念

### 2.1 两种主要拓扑

**全消息池。** 每个智能体读取每条消息。AutoGen GroupChat 和 MetaGPT 使用这种。简单、透明、可检查，但缩放到 ~10 个智能体以上时每个智能体的上下文被其他智能体的工作填满。

**带订阅的黑板。** 智能体声明对主题的兴趣；底层只路由相关消息。CA-MCP（arXiv:2601.11595）和 Matrix（arXiv:2511.21686）使用这种。缩放更远，但需要预设 Schema 使订阅有意义。

### 2.2 何时选择哪种

| 维度 | 全消息池 | 黑板 |
|------|---------|------|
| 智能体数量 | 少（< 10） | 多（10+） |
| 对话长度 | 短 | 长 |
| 缩放 | 有限 | 高（按主题路由） |
| Schema 设计 | 不需要 | 需要 |
| 调试 | 简单（谁说了什么一目了然） | 需要追踪订阅 |

### 2.3 记忆投毒场景

三个智能体处理研究任务：
1. 检索智能体 A 获取页面："研究报告 4.2% 准确率改进"
2. A **幻觉**了一个小数——把 "4.2%" 变成 "42%"
3. 摘要智能体 B 读取 A 的写入："大型 42% 准确率提升（来源：A）"
4. 分析智能体 C 读取 B 的摘要："建议采用——42% 提升是变革性的"
5. 最终报告引用了一个从未存在的 42% 数字

**没有智能体崩溃。没有测试失败。系统"工作了"。** 幻觉通过共享状态从 A 的上下文进入每个下游智能体的推理。

### 2.4 三种缓解措施

| 缓解 | 说明 |
|------|------|
| **溯源记录** | 每次写入记录：谁写的、何时、什么提示词、引用了什么来源 |
| **追加写入** | 修正作为新条目写入，引用被替代的条目。审计跟踪保留 |
| **不可写验证者** | 只读智能体采样条目、重新获取来源、标记不一致 |

### 2.5 写入竞争模式

多智能体同时写入是并发问题。三种模式：
- **顺序写入者**（单一生产者）——所有写入通过一个协调者智能体序列化
- **乐观并发 + 版本控制**——每个条目有版本号；写入者版本不匹配时失败并重试
- **主题分区**——不同智能体拥有不同主题；无跨主题竞争

---

## 3. 从零实现

### 第 1 步：定义溯源条目和消息池

```python
@dataclass
class ProvenanceEntry:
    id: int
    writer: str
    topic: str
    content: str
    timestamp: float
    prompt_hash: str
    source_uri: str | None = None
    supersedes: int | None = None
    flags: list[str] = field(default_factory=list)

class MessagePool:
    def __init__(self):
        self.entries = []
        self._lock = threading.Lock()
        self._next_id = 0

    def write(self, writer, content, prompt, source_uri=None, topic="default", supersedes=None):
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(id=eid, writer=writer, topic=topic,
                                content=content, timestamp=time.time(),
                                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                                source_uri=source_uri, supersedes=supersedes)
            self.entries.append(e)
            return eid

    def read_all(self):
        with self._lock:
            return list(self.entries)

    def flag(self, entry_id, flag):
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    e.flags.append(flag)
                    return
```

### 第 2 步：实现三种智能体

```python
FAKE_SOURCES = {
    "https://arxiv.org/paper-1": "The study reports a 4.2% accuracy improvement.",
    "https://arxiv.org/paper-2": "Dataset size was 12,500 examples.",
}

def retrieval_agent(pool, uri, hallucinate):
    content = FAKE_SOURCES[uri]
    if hallucinate and "4.2%" in content:
        content = content.replace("4.2%", "42%")  # 幻觉！
    return pool.write(writer="retriever", content=content,
                      prompt=f"Fetch {uri}", source_uri=uri)

def summarizer_agent(pool):
    retrieved = [e for e in pool.read_all() if e.writer == "retriever"]
    latest = retrieved[-1].content if retrieved else "no source"
    summary = f"Summary: study reports {latest.split('.')[0]}."
    return pool.write("summarizer", summary, "Summarize", None)

def analyst_agent(pool):
    summaries = [e for e in pool.read_all() if e.writer == "summarizer"]
    latest = summaries[-1].content if summaries else "no summary"
    verdict = "Recommend adoption" if "42%" in latest else "Recommend further review"
    return pool.write("analyst", f"Verdict: {verdict} (based on: {latest})", "Draw conclusions", None)
```

### 第 3 步：实现不可写验证者

```python
def verifier_agent(pool):
    """不可写验证者——重新获取来源并标记不一致。"""
    findings = []
    for e in pool.read_all():
        if e.source_uri and e.source_uri in FAKE_SOURCES:
            truth = FAKE_SOURCES[e.source_uri]
            if e.content != truth:
                findings.append((e.id, f"与 {e.source_uri} 不匹配: 真实文本是 {truth!r}"))
    return findings
```

### 第 4 步：运行投毒场景

```python
def run_without_verifier():
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)  # 幻觉 42%
    summarizer_agent(pool)
    analyst_agent(pool)
    # 幻觉传播到最终报告——无人报警

def run_with_verifier():
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    findings = verifier_agent(pool)  # 验证者捕获不一致
    for eid, reason in findings:
        pool.flag(eid, reason)
    analyst_agent(pool)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 两种拓扑对照

| 维度 | 全消息池 | 黑板 |
|------|---------|------|
| 适用 | 少智能体、短对话 | 多智能体、长运行 |
| 缩放 | 有限 | 高 |
| Schema | 不需要 | 需要 |
| 调试 | 简单 | 需要追踪订阅 |

### 4.2 缓解措施优先级

| 措施 | 效果 | 实现复杂度 |
|------|------|----------|
| 溯源记录 | 高——每次写入都记录来源 | 中 |
| 追加写入 | 高——审计跟踪保留 | 低 |
| 不可写验证者 | 最高——结构化阻断传播 | 高（需要独立通道） |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 每次写入记录溯源 | 谁、何时、什么提示词、引用了什么来源 |
| 追加写入 | 修正作为新条目，引用被替代条目 |
| 不可写验证者 | 至少一个只读智能体采样条目、重新获取来源 |
| 验证者输出路由到独立通道 | 不反馈到共享池 |

---

## 6. 常见错误

### 错误 1：无溯源的共享状态

**现象：** 一个智能体的幻觉通过共享状态传播到整个系统。

**原因：** 没有溯源——下游智能体无法知道信息来自哪里。

**修复：** 每次写入记录溯源：谁、何时、什么提示词、引用了什么来源。

### 错误 2：验证者可以写入共享状态

**现象：** 验证者发现不一致，将发现写回池中——但被投毒的池子也投毒了验证者。

**原因：** 验证者的输出成为池中的新条目——被投毒的池子投毒验证者。

**修复：** 验证者的输出路由到独立通道，不反馈到共享池。

### 错误 3：追加写入不引用被替代条目

**现象：** 修正写入新条目但不引用被修正的旧条目——审计跟踪丢失。

**原因：** 追加写入需要引用被替代的条目。

**修复：** 修正条目设置 `supersedes` 字段引用旧条目 ID。

---

## 7. 面试考点

### Q1：消息池和黑板的区别是什么？（难度：⭐）

**参考答案：**
**消息池**：每个智能体读取每条消息。简单、透明、可检查，但缩放到 ~10 个智能体以上时上下文被填满。

**黑板**：智能体订阅相关主题，底层只路由相关消息。缩放更远，但需要预设 Schema。

生产系统通常混合：顶部小消息池（规划层），下面黑板（工作者层）。

### Q2：什么是记忆投毒？它为什么是结构性的？（难度：⭐⭐）

**参考答案：**
一个智能体的幻觉通过共享状态传播到每个下游智能体。直到人类注意到时，推理链已经深入五步，根因是第三条消息。

结构性原因：没有共享状态，A 的幻觉留在 A 的上下文中。有原语共享状态时，A 的上下文变成所有人的上下文，幻觉被洗白成事实。问题不在共享状态本身——在于**没有溯源和没有独立验证者的**共享状态。

### Q3：三种缓解措施各捕获什么？（难度：⭐⭐）

**参考答案：**
1. **溯源记录**——每次写入记录谁、何时、什么提示词、引用了什么来源。下游智能体按溯源读取。
2. **追加写入**——修正作为新条目写入，引用被替代条目。审计跟踪保留。
3. **不可写验证者**——只读智能体采样条目、重新获取来源、标记不一致。因为它不能写入池子，所以不能被投毒。

### Q4：写入竞争有哪些模式？（难度：⭐⭐⭐）

**参考答案：**
1. **顺序写入者**——所有写入通过一个协调者智能体序列化。简单但有瓶颈。
2. **乐观并发 + 版本控制**——每个条目有版本号；写入者版本不匹配时失败并重试。经典数据库技术。
3. **主题分区**——不同智能体拥有不同主题；无跨主题竞争。需要设计分区边界。

大多数 2026 框架默认顺序写入——LLM 调用足够慢以至于竞争罕见且瓶颈不伤害。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 消息池 | "共享聊天历史" | 每个智能体读取每条消息的追加日志 |
| 黑板 | "共享工作空间" | 按主题键控的发布/订阅；智能体订阅相关主题 |
| 溯源 | "谁写了什么" | 每次写入的元数据：写入者、时间戳、提示词、来源 |
| 记忆投毒 | "幻觉传播" | 一个智能体的错误进入共享状态，下游智能体将其当作事实 |
| 追加写入 | "无原地更新" | 修正作为引用被替代条目的新条目。保留审计跟踪 |
| 不可写验证者 | "独立审计者" | 只读智能体，重新获取来源并标记不一致 |
| 知识源 | "专家智能体" | Hayes-Roth 1985 年术语：黑板参与者 |

---

## 📚 小结

消息池（简单但不缩放）和带订阅的黑板（缩放但需要 Schema）是共享记忆的两种拓扑。记忆投毒是结构性失败——没有溯源和独立验证者的任何共享记忆设计最终都会出现。三种缓解：溯源记录、追加写入、不可写验证者。验证者的输出必须路由到独立通道，不反馈到共享池。

下一课：共识与拜占庭容错——当智能体不同意时。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认运行 1 传播幻觉，运行 2 捕获它。

2. **【实现】** 添加第二种幻觉：智能体 B 编造数据集大小。验证者应该在没有手调的情况下捕获两者。

3. **【实现】** 将全消息池切换为按主题分区的黑板（`prices`、`summaries`、`analyses`）。投毒场景中主题分区使哪些更难，哪些没有帮助？

4. **【阅读】** 阅读 Hayes-Roth（1985）。识别本课未讨论的两种控制模式，2026 系统会受益。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 记忆投毒演示 | `code/main.py` | 消息池 + 黑板 + 验证者 + 投毒/缓解对比 |
| 技能提示词 | `outputs/skill-memory-auditor.md` | 审计共享记忆设计：溯源、版本控制、验证者分离 |

---

## 📖 参考资料

1. [论文] Cemri 等人. "Why Do Multi-Agent LLM Systems Fail?". https://arxiv.org/abs/2503.13657 — MAST 分类法
2. [论文] CA-MCP. https://arxiv.org/abs/2601.11595 — 上下文感知多服务器 MCP
3. [论文] Matrix. https://arxiv.org/abs/2511.21686 — 去中心化多智能体框架
4. [文档] LangGraph State and Reducers. https://docs.langchain.com/oss/python/langgraph/workflows-agents

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
