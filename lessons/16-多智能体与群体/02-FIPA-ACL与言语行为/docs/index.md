# FIPA-ACL 与言语行为的遗产

> 在 MCP 之前、在 A2A 之前，有 FIPA-ACL。2000 年 IEEE 智能物理代理基金会批准了一种智能体通信语言，包含二十种言语行为、两种内容语言和一组交互协议。它因本体论开销对 Web 来说太重而淡出工业界，但 LLM 多智能体系统的复兴正在悄然重新实现相同的想法——没有形式语义：JSON 契约代替言语行为，自然语言代替本体论。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 16 · 01（为什么多智能体）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别 FIPA-ACL 的二十种言语行为中哪些对应 2026 年协议中的原语
- [ ] 理解 FIPA-ACL 的经典信封格式（谁→谁、意图、内容、协议）
- [ ] 实现一个 FIPA-ACL 转换器——将 MCP/A2A 消息转换为 ACL 信封并反向转换
- [ ] 理解为什么 FIPA 失败以及现代协议学到了什么教训
- [ ] 识别三种值得移植到 LLM 多智能体系统的 FIPA 交互协议

---

## 1. 问题

2026 年的智能体协议格局很忙：MCP 用于工具、A2A 用于智能体、ACP 用于企业审计、ANP 用于去中心化信任、NLIP 用于自然语言内容。每个规范都宣称自己是基础性的。

诚实的阅读是：它们大多数都在重新发现一个非常具体的二十年前的决策树。言语行为理论（Austin 1962，Searle 1969）给了我们"话语是行动"。KQML（1993）将其转化为线路协议。FIPA-ACL（2000 年批准）产生了参考标准化：二十种言语行为、内容语言 SL0/SL1、交互协议。

当你查看 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策的更柔和、JSON 原生的重做。了解谱系告诉你两件事：哪些新"创新"实际上是重新发明，以及新规范将重新发现哪些旧失败模式。

---

## 2. 概念

### 2.1 言语行为——一段话总结

Austin 注意到有些句子不描述世界——它们改变世界。"我承诺。""我请求。""我宣布。"他称这些为施为性话语。Searle 形式化了五类：断言、指令、承诺、表达、宣告。KQML（Finin 等人，1993）将其变为软件智能体的可操作标准：消息是言语行为（动作）加内容（动作关于什么）。FIPA-ACL 清理了 KQML 的缺口，围绕二十种言语行为标准化。

### 2.2 二十种 FIPA 言语行为（部分列表）

| 言语行为 | 意图 |
|---------|------|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请你做 X" |
| `query-if` | "P 为真吗？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受提议" |
| `reject-proposal` | "我拒绝提议" |
| `cfp` | "征求 X 的提案" |
| `subscribe` | "X 变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试了 X 并失败了" |

要点不是记住它——要点是每一种都对应 LLM 协议最终重新添加的原语。

### 2.3 FIPA-ACL 信封格式

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段携带协议信封；一个字段（`content`）携带载荷。其余字段正是你每次在 JSON 协议上添加重试、线程和本体论时重新发明的东西。

### 2.4 FIPA-ACL 与现代协议的对比

```
FIPA-ACL:                          MCP/A2A:
(inform                            {
  :sender agent1                    "jsonrpc": "2.0",
  :receiver tool-server             "method": "tools/call",
  :content "(lookup stock IBM)"     "params": {"name":"lookup_stock",
  :ontology finance                       "arguments":{"symbol":"IBM"}},
  :conversation-id c42             "id": 42
)                                  }
```

同样的信封，不同的语法。两者都携带：谁、对谁、意图、载荷、关联 ID。都不是对方的革命——它们是同一设计上的不同权衡。

### 2.5 FIPA 失败的原因

| 原因 | 说明 |
|------|------|
| 本体论开销 | FIPA 要求共享本体论来解析 `content`。达成共识需要数年标准流程 |
| 无人使用的形式语义 | SL 给出了严格的真值条件，但大多生产系统使用自由形式内容 |
| 工具锁定 | JADE 是 Java 专用；JACK 是商业的 |
| 互联网赢得了栈 | REST、JSON-RPC、gRPC 替代了 ACL 的传输 |

### 2.6 LLM 复兴是 FIPA-lite

| 现代规范 | FIPA 类比 | 保留了什么 | 丢弃了什么 |
|---------|----------|----------|----------|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体论 |
| A2A 任务生命周期 | 合同网 + 请求-何时 | 异步生命周期、状态转换 | 形式完备性保证 |
| ACP 轨迹元数据 | 通告 | 推理链审计 | 形式语义 |
| ANP 元协议 | —（新） | 去中心化身份、E2EE | — |

从上到下读：模式是保留结构原语，丢弃形式主义，让 LLM 覆盖歧义。

### 2.7 值得移植的三个 FIPA 交互协议

1. **合同网协议 (CNP)**：管理者发 `cfp`；投标者回 `propose`；管理者 `accept/reject`。标准任务市场模式
2. **订阅/通知**：订阅者发 `subscribe`；发布者在主题变化时发 `inform`。2026 年的每个事件总线
3. **请求-何时**："当条件 Y 成立时做 X。"延迟动作加前置条件。2026 年的等价物是持久工作流引擎中的延迟任务

### 2.8 丢弃本体论时什么会坏

没有共享本体论，智能体从自然语言内容推断含义。已记录的 2026 年失败模式是**语义漂移**：两个智能体用同一个词（"客户"）指代微妙不同的概念，接收方智能体按错误解释行动，没有 Schema 验证器捕获它。

缓解：
- `content` 上的 JSON Schema——在线路处拒绝结构错误
- 类型化工件（A2A）——拒绝错误模态
- 信封中的显式言语行为——即使内容是自然语言也使意图明确

---

## 3. 从零实现

### 第 1 步：定义 ACL 信封和转换器

```python
from dataclasses import dataclass
from typing import Any, Optional

PERFORMATIVES = {
    "inform", "request", "query-if", "query-ref", "propose",
    "accept-proposal", "reject-proposal", "agree", "refuse",
    "confirm", "disconfirm", "not-understood", "cfp",
    "subscribe", "cancel", "failure",
}

@dataclass
class ACLMessage:
    performative: str
    sender: str
    receiver: str
    content: Any
    language: str = "SL0"
    ontology: str = "default"
    protocol: Optional[str] = None
    conversation_id: Optional[str] = None
    reply_with: Optional[str] = None

    def render(self) -> str:
        fields = [
            f":sender       {self.sender}",
            f":receiver     {self.receiver}",
            f":content      {self.content!r}",
            f":language     {self.language}",
            f":ontology     {self.ontology}",
        ]
        if self.protocol:
            fields.append(f":protocol     {self.protocol}")
        if self.conversation_id:
            fields.append(f":conversation-id {self.conversation_id}")
        if self.reply_with:
            fields.append(f":reply-with   {self.reply_with}")
        inner = "\n  ".join(fields)
        return f"({self.performative}\n  {inner}\n)"
```

### 第 2 步：实现 MCP/A2A 转换器

```python
def mcp_tools_call_to_acl(req: dict) -> ACLMessage:
    """MCP tools/call → FIPA-ACL request。"""
    return ACLMessage(
        performative="request",
        sender="host",
        receiver=req["params"]["name"],
        content=req["params"].get("arguments", {}),
        language="JSON",
        protocol="fipa-request",
        conversation_id=f"jsonrpc-{req['id']}",
    )

def a2a_task_create_to_acl(task: dict) -> ACLMessage:
    """A2A POST /tasks → FIPA-ACL request。"""
    return ACLMessage(
        performative="request",
        sender=task.get("client", "client"),
        receiver=task.get("agent", "agent"),
        content=task["input"],
        language="JSON",
        protocol="a2a-task",
        conversation_id=task.get("task_id", "t-0"),
    )
```

### 第 3 步：实现合同网演示

```python
@dataclass
class Bid:
    bidder: str
    price: int
    eta_minutes: int

@dataclass
class ContractNet:
    manager: str
    bidders: list[str]
    log: list = field(default_factory=list)

    def cfp(self, task, conv):
        for b in self.bidders:
            self.log.append(ACLMessage(
                performative="cfp", sender=self.manager,
                receiver=b, content=task,
                ontology="contract-net", conversation_id=conv,
            ))

    def propose(self, bidder, bid, conv):
        self.log.append(ACLMessage(
            performative="propose", sender=bidder,
            receiver=self.manager, content={"price": bid[0], "eta": bid[1]},
            ontology="contract-net", conversation_id=conv,
        ))

    def award(self, winner, losers, conv):
        self.log.append(ACLMessage(
            performative="accept-proposal", sender=self.manager,
            receiver=winner, content="awarded", conversation_id=conv,
        ))
        for loser in losers:
            self.log.append(ACLMessage(
                performative="reject-proposal", sender=self.manager,
                receiver=loser, content="not awarded", conversation_id=conv,
            ))
```

### 第 4 步：运行演示

```python
def main():
    # MCP tools/call → FIPA-ACL
    mcp_call = {"jsonrpc": "2.0", "method": "tools/call",
                "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}}, "id": 42}
    acl = mcp_tools_call_to_acl(mcp_call)
    print("MCP tools/call → FIPA-ACL:")
    print(acl.render())

    # 合同网演示
    cn = ContractNet(manager="scheduler", bidders=["worker-a", "worker-b", "worker-c"])
    cn.cfp("compress 10GB log", "cn-1")
    cn.propose("worker-a", Bid("worker-a", 3, 18), "cn-1")
    cn.propose("worker-b", Bid("worker-b", 2, 25), "cn-1")
    cn.propose("worker-c", Bid("worker-c", 4, 10), "cn-1")
    cn.award("worker-b", ["worker-a", "worker-c"], "cn-1")

    for msg in cn.log:
        print(msg.render())
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 现代协议与 FIPA 谱系对照

| 现代规范 | FIPA 等价 | 保留 | 丢弃 |
|---------|----------|------|------|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体论 |
| A2A 任务生命周期 | 合同网 + 请求-何时 | 异步生命周期、状态转换 | 形式完备性保证 |
| ACP 轨迹元数据 | 通告 | 推理链审计 | 形式语义 |
| ANP 元协议 | —（新） | 去中心化身份 | — |

### 4.2 本体论问题的缓解

| 缓解 | 说明 | 限制 |
|------|------|------|
| JSON Schema | 在线路处拒绝结构错误 | 不捕获语义歧义 |
| 类型化工件（A2A） | 拒绝错误模态 | 只覆盖类型 |
| 显式言语行为 | 使意图明确 | 不消除内容歧义 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 保留结构原语 | 谁→谁、意图、内容、关联 ID |
| 丢弃形式主义 | 不要求共享本体论 |
| 显式言语行为 | 每条消息都有意图标签 |
| JSON Schema 覆盖结构 | 在线路处拒绝结构错误 |
| 考虑语义漂移 | 同一个词可能指代不同概念 |

---

## 6. 常见错误

### 错误 1：认为新协议是全新发明

**修复：** 了解谱系。新协议保留结构原语，丢弃形式主义。

### 错误 2：不做语义漂移检查

**修复：** `content` 上的 JSON Schema + 显式言语行为 + 类型化工件。

### 错误 3：忽视合同网模式

**修复：** 使用标准合同网模式（`cfp` → `propose` → `accept/reject`），而不是自定义委派。

---

## 7. 面试考点

### Q1：FIPA-ACL 的二十种言语行为中哪些对应 2026 年的协议？（难度：⭐）

**参考答案：**
`request` → MCP `tools/call`；`query-ref` → MCP `resources/read`；`cfp` → 合同网；`propose/accept/reject` → A2A 协商；`subscribe` → 事件总线。

### Q2：为什么 FIPA 失败了而现代协议成功了？（难度：⭐⭐）

**参考答案：**
本体论开销太重、形式语义无人使用、工具锁定、互联网赢得了栈。现代协议保留结构原语，丢弃形式主义，让 LLM 覆盖歧义。

### Q3：丢弃本体论后什么会坏？（难度：⭐⭐⭐）

**参考答案：**
语义漂移——两个智能体用同一个词指代不同概念。缓解：JSON Schema + 类型化工件 + 显式言语行为。但不能完全消除语义歧义。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 言语行为 | "话语是行动" | Austin/Searle：施为性话语改变世界 |
| FIPA-ACL | "智能体通信语言" | 2000 年批准的信封格式：言语行为 + 内容 + 元数据 |
| 合同网 | "任务市场" | 管理者发 cfp；投标者 propose；管理者 accept。标准交互协议 |
| 语义漂移 | "同一词不同含义" | 两个智能体用同一个词指代不同概念 |

---

## 📚 小结

FIPA-ACL 的二十种言语行为和信封格式是 2026 年协议（MCP、A2A、ACP）的祖先。每个现代协议保留结构原语（谁→谁、意图、内容、关联 ID），丢弃形式主义（本体论、形式语义），让 LLM 覆盖歧义。FIPA 失败是因为本体论开销太重；LLM 复兴是 FIPA-lite。值得移植的三个 FIPA 交互协议：合同网、订阅/通知、请求-何时。

下一课：通信协议——MCP、A2A、ACP、ANP 四协议深度解析。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。观察往返编码。识别 `tools/call`、`resources/read`、A2A 任务创建对应的 FIPA 言语行为。

2. **【实现】** 扩展合同网演示，添加 `cancel` 言语行为——让管理者在投标中途撤回任务。

3. **【阅读】** 阅读 FIPA ACL 消息结构规范（fipa00037）第 4.1-4.3 节。选一个本课未涵盖的言语行为，描述其现代 JSON-RPC 等价物。

4. **【设计】** 为 `request` 言语行为的 `content` 字段设计一个最小 JSON Schema。纯自然语言不提供什么而 Schema 提供什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| FIPA-ACL 转换器 | `code/main.py` | MCP/A2A 消息 ↔ ACL 信封往返 + 合同网 |
| 技能提示词 | `outputs/skill-fipa-mapper.md` | 读取任何协议规范并生成 FIPA-ACL 映射 |

---

## 📖 参考资料

1. [论文] Liu et al. "A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP". https://arxiv.org/html/2505.02279v1
2. [规范] FIPA ACL 消息结构规范 (fipa00037). http://www.fipa.org/specs/fipa00037/
3. [规范] MCP 2025-11-25. https://modelcontextprotocol.io/specification/2025-11-25
4. [规范] A2A. https://a2a-protocol.org/latest/specification/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
