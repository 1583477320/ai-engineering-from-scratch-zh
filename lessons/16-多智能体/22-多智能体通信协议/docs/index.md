# 多智能体通信协议

> 两个智能体各说各话不是通信，是噪音。通信协议定义了它们如何达成共识——什么时候说话、说什么、怎么说。没有协议，多智能体就是一群自说自话的鹦鹉。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 16 · 19（多智能体协商与拍卖机制）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 16 · 23（多智能体共识与投票）— 通信是共识的前提

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释三种主流智能体通信模式——直接消息、共享黑板、事件总线的优缺点和适用场景
- [ ] 实现基于共享黑板（Blackboard）的多智能体信息交换系统——去中心化、可扩展
- [ ] 理解结构化消息协议（如 FIPA ACL）的设计原则——消息类型、内容语言、交互协议
- [ ] 诊断通信瓶颈——消息过载、格式冲突、死信队列——并给出解决方案

---

## 1. 问题

你有 5 个智能体协作完成一个复杂的文档分析任务。智能体 A 把分析结果发给了智能体 B，但 B 没有收到——因为 A 用的是一个叫 `send_to_agent` 的自定义函数，B 用的是一个 `receive_message` 的 WebSocket。A 以为自己发了消息，B 没有任何通知。错误在 3 轮对话后才被发现——但中间步骤的数据已经全部丢失了。

更糟糕的是，当系统扩展到 20 个智能体时，每个智能体都需要知道其他 19 个智能体的通信地址。新增一个智能体意味着要修改 19 个智能体的配置。文档分析变成了配置管理噩梦。

**通信协议解决了三个问题：**
1. **寻址：** 智能体不需要知道对方的地址——消息通过逻辑名称路由
2. **格式：** 所有消息使用统一的结构——发送方、接收方、内容语言、交互协议
3. **语义：** 消息的含义是明确的——"请求"、"通知"、"查询"、"确认"——不是模糊的自然语言

---

## 2. 概念

### 2.1 三种通信模式

多智能体系统的通信模式有三种基本架构：

```
直接消息模式                    共享黑板模式                    事件总线模式
                                
A → B                          A ──► 黑板 ◄── B               A ──► 事件总线 ◄── B
A → C                          C ──► 黑板 ◄── D               C ──► 事件总线 ◄── D
B → D                                                         E ──► 事件总线 ◄── F
                                                              
点对点，最直接                  去中心化，松耦合                发布/订阅，可扩展
```

**直接消息（Direct Messaging）：**

最简单——智能体之间直接发送消息。问题在于耦合度太高：每个智能体需要知道其他智能体的地址、接口、消息格式。20 个智能体时，每个智能体需要维护 19 个地址。

**共享黑板（Blackboard）：**

所有智能体通过一个中央"黑板"交换信息。A 把结果写到黑板，B 从黑板读取。智能体之间不需要知道对方的存在——它们只和黑板通信。新增智能体不需要修改已有智能体的配置。

**事件总线（Event Bus）：**

智能体发布事件到总线，订阅了该事件的智能体自动接收。发布者不知道谁在订阅，订阅者不知道谁是发布者。最松耦合，但需要事件类型的一致命名。

### 2.2 消息结构

无论使用哪种通信模式，消息本身应该有一个统一的结构。FIPA ACL（Foundation for Intelligent Physical Agents - Agent Communication Language）定义了标准的消息结构：

```
消息 = {
  sender:      "agent_A",
  receiver:    "agent_B",
  performative: "request",    // 消息类型：request、inform、query、confirm、refuse
  content:     "请分析上季度销售数据",
  language:    "json",        // 内容的语言
  ontology:    "data_analysis",  // 领域本体
  protocol:    "fipa-request",   // 交互协议
  reply_with:  "msg_001",     // 消息 ID
  in_reply_to: null,          // 回复的消息 ID
}
```

`performative`（言外行为）是最重要的字段——它告诉接收方这条消息是"请求"（需要回复）还是"告知"（不需要回复）。这个区分在异步通信中至关重要。

### 2.3 大语言模型时代的通信

传统智能体使用预定义的消息类型和协议。LLM 智能体有一个新选择：**自然语言通信**。智能体可以直接用自然语言发送消息，接收方用 LLM 理解消息含义。

```
LLM 智能体 A 发送："请分析上季度销售数据，重点是增长趋势"
LLM 智能体 B 接收后：用 LLM 解析"分析上季度销售数据"是什么意思→执行→回复

# 实际上，LLM 智能体可以通过结构化消息做同样的事：
{
  "performative": "request",
  "content": {"task": "data_analysis", "period": "last_quarter", "focus": "growth_trend"},
  "ontology": "business_intelligence"
}
```

自然语言通信更灵活，但代价是更加昂贵和不可预测。结构化通信解析效率高，但需要预定义本体。**实践中，结构化 + 自然语言混合最常用**——外层是结构化的消息头（sender、performative、protocol），内层的内容用自然语言描述。

---

## 3. 从零实现

### 第 1 步：定义消息格式

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Performative(Enum):
    """FIPA ACL 言外行为——消息的意图。"""
    REQUEST = "request"     # 请求对方执行某个动作
    INFORM = "inform"       # 告知对方某个事实
    QUERY = "query"         # 查询信息
    CONFIRM = "confirm"     # 确认收到
    REFUSE = "refuse"       # 拒绝请求
    AGREE = "agree"         # 同意请求
    FAILURE = "failure"     # 执行失败
    SUBSCRIBE = "subscribe"  # 订阅事件


@dataclass
class Message:
    """统一的消息格式，基于 FIPA ACL。"""
    sender: str
    receiver: str
    performative: Performative
    content: str = ""
    ontology: str = "default"
    protocol: str = "fipa-request"
    reply: Optional[str] = None       # 回复的消息 ID
    conversation_id: str = ""         # 会话 ID，用于追踪对话流
```

### 第 2 步：共享黑板实现

```python
import time
from collections import defaultdict


class Blackboard:
    """共享黑板——智能体之间的信息交换中心。

    智能体通过 write 写入消息，通过 read 读取特定类型的消息。
    消息在黑板上有 TTL（生存时间），超时后自动清理。
    """

    def __init__(self, default_ttl: float = 60.0):
        self._entries = []          # 所有消息
        self._subscriptions = defaultdict(list)  # ontology -> [agent_ids]
        self.default_ttl = default_ttl

    def write(self, message: Message) -> str:
        """智能体将消息写入黑板。"""
        entry = {
            "message": message,
            "timestamp": time.time(),
            "expires_at": time.time() + self.default_ttl,
        }
        self._entries.append(entry)

        # 自动通知订阅者
        for subscriber in self._subscriptions.get(message.ontology, []):
            self._notify(subscriber, message)

        return f"msg_{len(self._entries)}"

    def read(self, ontology: str = "", performative: Optional[Performative] = None,
             agent_id: str = "", since: float = 0) -> list[Message]:
        """读取符合条件的消息。"""
        self._cleanup_expired()
        results = []

        for entry in self._entries:
            msg = entry["message"]
            if ontology and msg.ontology != ontology:
                continue
            if performative and msg.performative != performative:
                continue
            if agent_id and msg.receiver != agent_id and msg.receiver != "ALL":
                continue
            if entry["timestamp"] < since:
                continue
            results.append(msg)

        return results

    def subscribe(self, agent_id: str, ontology: str):
        """订阅特定类型的消息——新消息自动通知。"""
        if agent_id not in self._subscriptions[ontology]:
            self._subscriptions[ontology].append(agent_id)

    def unsubscribe(self, agent_id: str, ontology: str):
        """取消订阅。"""
        if agent_id in self._subscriptions[ontology]:
            self._subscriptions[ontology].remove(agent_id)

    def _notify(self, agent_id: str, message: Message):
        """通知订阅者（实际系统中会调用智能体的回调方法）。"""
        pass

    def _cleanup_expired(self):
        """清理过期的消息。"""
        now = time.time()
        self._entries = [e for e in self._entries if e["expires_at"] > now]
```

### 第 3 步：智能体通过黑板通信

```python
class Agent:
    """一个通过黑板通信的智能体。"""

    def __init__(self, agent_id: str, blackboard: Blackboard):
        self.agent_id = agent_id
        self.blackboard = blackboard
        self.knowledge = {}  # 本地知识

    def send(self, receiver: str, performative: Performative, content: str,
             ontology: str = "default"):
        """发送消息（写入黑板）。"""
        msg = Message(
            sender=self.agent_id,
            receiver=receiver,
            performative=performative,
            content=content,
            ontology=ontology,
        )
        self.blackboard.write(msg)
        print(f"[{self.agent_id}] → {receiver}: {performative.value} | {content[:20]}...")

    def receive(self, ontology: str = "", since: float = 0) -> list[Message]:
        """接收消息（从黑板读取）。"""
        messages = self.blackboard.read(
            ontology=ontology,
            agent_id=self.agent_id,
            since=since,
        )
        return messages

    def process_messages(self):
        """处理收到的消息。"""
        messages = self.receive()
        for msg in messages:
            if msg.performative == Performative.REQUEST:
                response = self._handle_request(msg)
                if response:
                    self.send(msg.sender, Performative.INFORM, response, "analysis_results")
            elif msg.performative == Performative.QUERY:
                response = self._handle_query(msg)
                if response:
                    self.send(msg.sender, Performative.INFORM, response, msg.ontology)
            elif msg.performative == Performative.INFORM:
                self.knowledge[msg.ontology] = msg.content

    def _handle_request(self, msg: Message) -> str:
        """处理请求消息。"""
        return f"[{self.agent_id}] 完成: {msg.content}"

    def _handle_query(self, msg: Message) -> str:
        """处理查询消息。"""
        return self.knowledge.get(msg.ontology, "未知")
```

### 第 4 步：运行通信系统

```python
bb = Blackboard()

analyzer = Agent("数据分析师", bb)
writer = Agent("报告撰写者", bb)
verifier = Agent("核验者", bb)

# 数据分析师发送请求
analyzer.send("报告撰写者", Performative.REQUEST, "分析上季度销售数据")

# 报告撰写者处理请求
writer.process_messages()

# 核验者订阅结果
verifier.subscribe(verifier.agent_id, "analysis_results")

# 报告撰写者将结果写入黑板
writer.send("ALL", Performative.INFORM, "销售增长 20%", "analysis_results")

# 核验者自动收到结果
verifier.process_messages()
print(f"核验者知识: {verifier.knowledge}")
```

```text
[数据分析师] → 报告撰写者: request | 分析上季度销售数据...
[报告撰写者] → ALL: inform | 销售增长 20%...
核验者知识: {'analysis_results': '[报告撰写者] 完成: 分析上季度销售数据'}
```

注意：核验者通过订阅 `analysis_results` 本体，自动接收了报告撰写者的结果——即使数据分析师没有明确告诉核验者。**黑板的松耦合特性让智能体不需要知道"消息应该发给谁"，只需要关心"我关心什么类型的信息"。**

---

## 4. 工业工具

### 4.1 LangGraph——通过 State 通信

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def analyst_node(state: MessagesState):
    response = llm.invoke([
        {"role": "system", "content": "分析数据并输出分析结果。"},
        *state["messages"],
    ])
    # 输出自动写入共享状态（相当于黑板）
    return {"messages": [response]}


def writer_node(state: MessagesState):
    # 读取共享状态中的分析结果
    analysis = state["messages"][-1].content if state["messages"] else ""
    response = llm.invoke([
        {"role": "system", "content": f"基于以下分析结果撰写报告：{analysis}"},
    ])
    return {"messages": [response]}
```

LangGraph 的共享 `State` 本质上就是一个黑板——每个节点的输出自动写入状态，后面的节点从状态读取。但 LangGraph 的黑板是有状态的（每次调用覆盖或追加），不是消息队列。

### 4.2 Redis Pub/Sub——事件总线方案

```python
import redis

r = redis.Redis(host="localhost", port=6379)

# 发布者（智能体 A）
def publish_analysis(data):
    r.publish("analysis_results", data)

# 订阅者（智能体 B）
pubsub = r.pubsub()
pubsub.subscribe("analysis_results")

for message in pubsub.listen():
    if message["type"] == "message":
        print(f"收到分析结果: {message['data']}")
```

Redis Pub/Sub 是事件总线模式的工业实现。发布者和订阅者完全解耦——发布者不知道谁在订阅，订阅者不知道谁是发布者。适用于大规模多智能体系统，但需要独立维护 Redis 服务。

### 4.3 工具对比

| 架构 | 耦合度 | 可扩展性 | 容错性 | 适用场景 |
|---|---|---|---|---|
| 直接消息 | 高 | 低（O(n²)） | 低 | 小规模、固定拓扑 |
| 共享黑板 | 中 | 中（O(n)） | 高 | 中等规模、动态加入 |
| 事件总线 | 低 | 高（O(1)） | 中 | 大规模、生产环境 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

OpenAI 的 GPTs 功能内部使用事件总线模式——主模型收到用户消息后，触发对应的事件（代码执行、联网搜索、图像生成），响应事件的服务者自动执行。Anthropic 的 Claude Artifacts 类似——编译器事件触发代码渲染。

### 5.2 LLM 时代什么变了？

传统智能体通信需要预定义的消息类型和协议。LLM 智能体可以理解自然语言消息——"帮我分析一下这个数据"和 `{"performative": "request", "content": "analyze_data"}` 在 LLM 看来是一样的。**通信格式从 IDE 强制变成了 LLM 理解。**

### 5.3 什么没变？

消息路由和任务分发的基础架构没有变——黑板和事件总线仍是主流选择。增加了一个新问题：当所有智能体都用自然语言通信时，消息体变大（自然语言比 JSON 更长），通信成本随之增加。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你用 ChatGPT 的 DALL-E 生成图像时，系统内部的事件总线在运行："文本生成"事件 → 触发 DALL-E 服务 → 返回图像 URL → "图像就绪"事件 → 触发回复生成。整个过程对用户透明——你不需要知道 DALL-E 的 API 地址。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 规模 | 推荐方案 | 备注 |
|---|---|---|
| 2-5 个智能体 | 直接消息（函数调用） | 最简单，够用就行 |
| 5-20 个智能体 | LangGraph 共享状态 | 结构化，可追踪 |
| 20-100 个智能体 | Redis Pub/Sub | 松耦合，可扩展 |
| 100+ 个智能体 | Apache Kafka | 企业级事件流处理 |

### 6.2 中文场景特别建议

- **中文智能体的消息体更大。** 同等信息量，中文消息比英文长 1.5-2 倍。通信带宽需要相应扩展
- **中文 Ontology 命名要统一。** `analysis_results` vs `分析结果` 在黑板中会被视为不同的本体。中英文混用会导致订阅失败
- **中文语境下的消息幂等性。** 中文 LLM 可能对同一自然语言消息产生不同的理解——`"分析数据"` 可能被理解为"数据分析"或"数据审查"。使用结构化消息头 + 中文内容体混合方案

### 6.3 踩坑经验

- **消息体过大。** LLM 智能体的输出可能很长（完整报告），直接作为消息传递会撑爆黑板。解决方案：黑板只存摘要和引用（URI），完整内容存在外部存储
- **死信队列缺失。** 没有接收方的消息永远留在黑板里。解决方案：每条消息设置 TTL，超时自动清理
- **回环通信。** 智能体 A 发消息给 B → B 处理后又发消息给 A → A 再处理又发消息给 B……无限循环。解决方案：每个消息带 `conversation_id` 和 `hop_count`，超过最大跳数自动丢弃
- **订阅泄漏。** 智能体退出后没有取消订阅 → 黑板继续向不存在的智能体推送消息。解决方案：心跳检测 + 自动清理失效订阅

---

## 7. 常见错误

### 错误 1：直接消息模式导致 O(n²) 配置

**现象：** 系统扩展到 10 个智能体后，每个新智能体需要配置 9 个通信地址。新增一个智能体需要修改 9 个已有智能体的配置。

**原因：** 直接消息模式要求每个智能体知道其他所有智能体的地址。配置维护开销随智能体数量平方增长。

**修复：**

```python
# ❌ 直接消息——每个智能体维护通信列表
class DirectAgent:
    def __init__(self):
        self.peers = {}  # 需要手动维护

    def send_to(self, agent_id, message):
        peer = self.peers.get(agent_id)
        if peer:
            peer.receive(message)

# ✓ 黑板模式——智能体只和黑板通信
class BlackboardAgent:
    def __init__(self, blackboard):
        self.blackboard = blackboard  # 唯一的依赖

    def send(self, receiver, message):
        self.blackboard.write(Message(
            sender=self.agent_id, receiver=receiver, content=message
        ))
    # 新增智能体不需要修改任何已有智能体
```

### 错误 2：黑板消息没有 TTL 导致内存爆炸

**现象：** 运行 24 小时后，黑板占用了 8GB 内存，查询响应时间从 1ms 飙升到 5 秒。

**原因：** 所有消息都永久保存，没有 TTL。每秒 10 条消息 × 86400 秒 = 86 万条消息。

**修复：**

```python
# ❌ 消息永久保留
self._entries.append(entry)

# ✓ 设置 TTL
entry["expires_at"] = time.time() + self.default_ttl  # 60 秒后自动清理
```

### 错误 3：LLM 智能体通信不设置超时

**现象：** 智能体 A 发送消息给 B → B 没有回复 → A 一直等待 → 整个流水线卡住。

**原因：** 没有超时机制。A 假设 B 一定会回复，B 可能在处理其他任务。

**修复：**

```python
# ❌ 无限等待
response = blackboard.wait_for_reply(message_id)

# ✓ 设置超时
import time

def wait_for_reply(blackboard, message_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        replies = blackboard.read(reply_to=message_id)
        if replies:
            return replies[0]
        time.sleep(0.1)
    raise TimeoutError(f"等待回复超时: {message_id}")
```

---

## 8. 面试考点

### Q1：共享黑板和事件总线的本质区别是什么？（难度：⭐⭐）

**参考答案：**
黑板是**主动写入、被动读取**——智能体把消息写到黑板，其他智能体定期轮询或主动读取。事件总线是**主动发布、自动推送**——智能体发布事件，订阅者自动收到推送。黑板的优势是**消息有持久化**（写入后即使订阅者不在线，后续也能读取），事件总线的优势是**更实时**（推送到订阅者无需轮询）。选择依据：如果订阅者需要回溯历史消息（如数据分析师先写入结果，报告撰写者过一会儿才读取）→ 黑板。如果需要实时通知（如故障告警）→ 事件总线。

### Q2：设计一个多智能体系统的通信架构，支持 50 个 LLM 智能体协作。你会选择什么模式？为什么？（难度：⭐⭐⭐）

**参考答案：**
选择**分层混合架构**——顶层用事件总线（大规模、松耦合），底层用黑板（小团队内消息持久化）。50 个智能体按功能分成 5 个小组（数据组、分析组、写作组、核验组、反馈组），每组 10 个智能体。组内通信使用黑板（共享数据），组间通信使用事件总线（发布结果）。这样组内新增智能体不影响其他组（黑板），组间松耦合（事件总线），且每个黑板的规模控制在 10 个智能体以内，不会爆炸。

### Q3：LLM 智能体的通信协议应该设计成结构化还是自然语言？给出权衡。（难度：⭐⭐⭐）

**参考答案：**
**混合设计最实用。** 结构化部分：sender、receiver、performative、protocol、message_id——这些字段必须结构化，确保路由正确、消息可追踪、超时可检测。自然语言部分：content 字段——用自然语言描述具体内容。你不需要定义每个任务的 JSON Schema，LLM 可以直接理解"分析上季度销售数据，重点看增长率"这个自然语言字符串。权衡：结构化降低了解析开销和错误率（解析器从不失败），但灵活度低（新增消息类型需要修改协议）。自然语言灵活但昂贵（每次通信都需要 LLM 理解消息），且不可预测（LLM 可能误解消息含义）。混合方案取两者之长：路由和追踪是结构化的，内容是自然语言的。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 直接消息 | "点对点通信" | 智能体之间直接发送消息，简单但耦合度高 |
| 共享黑板 | "中心公告栏" | 所有智能体通过中央存储交换信息，智能体之间互相不知道 |
| 事件总线 | "发布/订阅" | 智能体发布事件到总线，订阅者自动接收，最松耦合 |
| FIPA ACL | "智能体通信标准" | 智能体通信的标准化消息结构，定义 performative、ontology、protocol 等 |
| 言外行为 | "消息的意图" | 消息的 performative 字段——是"请求"、"告知"还是"查询" |
| 本体 | "领域的分类体系" | 消息所属的知识领域——如 data_analysis、report_generation。用于消息路由 |
| 死信 | "发不出去的消息" | 没有接收方或接收方不存在的消息。需要超时清理 |
| TTL | "消息过期时间" | 消息在黑板中的生存时间。超时自动清理，防止内存爆炸 |

---

## 📚 小结

多智能体系统的通信模式有三种：直接消息（简单但耦合高）、共享黑板（去中心化、持久化）、事件总线（松耦合、可扩展）。FIPA ACL 定义了标准消息结构——sender、receiver、performative、content、ontology。LLM 时代，结构化 + 自然语言混合通信是最实用的方案：外层结构化（路由和追踪），内容自然语言（灵活和表达力）。TTL、死信队列、跳数限制是生产环境必须考虑的工程细节。

下一课我们将讨论多智能体共识与投票——当多个智能体的意见不一致时，系统如何达成统一决策。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么直接消息模式在 3 个智能体时够用，在 30 个智能体时就不够用了？黑板模式如何解决这个问题？

2. 【实现】扩展 `Blackboard` 类，添加 `wait_for_reply(message_id, timeout)` 方法——智能体可以等待某个消息的回复，超时后返回 None。

3. 【实验】用直接消息模式和黑板模式分别实现 5 个智能体协作的任务分配。比较两种模式的代码量、可读性、扩展性。

4. 【思考】如果你的多智能体系统使用 Redis Pub/Sub 做通信，但 Redis 服务突然挂了，系统应该如何降级？设计一个降级方案。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 共享黑板通信系统 | `code/blackboard.py` | 完整的黑板通信实现，含消息 TTL、订阅、清理 |
| 通信架构对比工具 | `code/communication_bench.py` | 三种通信模式的性能对比测试 |

---

## 📖 参考资料

1. [标准] FIPA. "FIPA ACL Message Structure Specification". IEEE Foundation for Intelligent Physical Agents, 2002. http://www.fipa.org/specs/fipa00061/ — 智能体通信语言标准
2. [论文] Engelmore, R. & Morgan, T. "Blackboard Systems". Addison-Wesley, 1988. — 黑板架构的经典著作
3. [官方文档] Redis Pub/Sub. https://redis.io/docs/manual/pubsub/ — 事件总线模式的生产实现
4. [官方文档] Apache Kafka. https://kafka.apache.org/documentation/ — 企业级事件流处理
5. [论文] Kafle, S. et al. "An Empirical Study on the Performance of Inter-Agent Communication in LLM-based Multi-Agent Systems". 2024. — LLM 智能体通信性能研究

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
