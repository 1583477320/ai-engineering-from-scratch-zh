"""FIPA-ACL 转换器和迷你合同网演示——纯标准库。

展示每个 2026 年智能体协议消息（MCP tools/call、MCP resources/read、A2A 任务创建）
都归约为具有不同语法的 FIPA-ACL 信封。
然后使用言语行为运行三投标者合同网协商。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        if self.performative not in PERFORMATIVES:
            raise ValueError(f"unknown performative: {self.performative}")

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


# ── 转换器 ────────────────────────────────────────────────

def mcp_tools_call_to_acl(req: dict) -> ACLMessage:
    return ACLMessage(
        performative="request", sender="host",
        receiver=req["params"]["name"],
        content=req["params"].get("arguments", {}),
        language="JSON", ontology=req["params"]["name"],
        protocol="fipa-request",
        conversation_id=f"jsonrpc-{req['id']}",
    )

def mcp_resources_read_to_acl(req: dict) -> ACLMessage:
    return ACLMessage(
        performative="query-ref", sender="host",
        receiver="resource-server",
        content=req["params"]["uri"],
        language="URI", ontology="mcp-resource",
        protocol="fipa-query",
        conversation_id=f"jsonrpc-{req['id']}",
    )

def a2a_task_create_to_acl(task: dict) -> ACLMessage:
    return ACLMessage(
        performative="request",
        sender=task.get("client", "client"),
        receiver=task.get("agent", "agent"),
        content=task["input"],
        language="JSON", ontology=task.get("skill", "default"),
        protocol="a2a-task",
        conversation_id=task.get("task_id", "t-0"),
    )


# ── 合同网 ────────────────────────────────────────────────

@dataclass
class Bid:
    bidder: str
    price: int
    eta_minutes: int


@dataclass
class ContractNet:
    manager: str
    bidders: list[str]
    log: list[ACLMessage] = field(default_factory=list)

    def cfp(self, task: str, conv: str) -> None:
        for b in self.bidders:
            self.log.append(ACLMessage(
                performative="cfp", sender=self.manager, receiver=b,
                content=task, ontology="contract-net",
                protocol="fipa-contract-net", conversation_id=conv,
                reply_with=f"cfp-{b}",
            ))

    def propose(self, bidder: str, bid: Bid, conv: str) -> None:
        self.log.append(ACLMessage(
            performative="propose", sender=bidder, receiver=self.manager,
            content={"price": bid.price, "eta_minutes": bid.eta_minutes},
            ontology="contract-net", protocol="fipa-contract-net",
            conversation_id=conv, reply_with=f"propose-{bidder}",
        ))

    def award(self, winner: str, losers: list[str], conv: str) -> None:
        self.log.append(ACLMessage(
            performative="accept-proposal", sender=self.manager,
            receiver=winner, content="awarded",
            ontology="contract-net", protocol="fipa-contract-net",
            conversation_id=conv,
        ))
        for loser in losers:
            self.log.append(ACLMessage(
                performative="reject-proposal", sender=self.manager,
                receiver=loser, content="not awarded",
                ontology="contract-net", protocol="fipa-contract-net",
                conversation_id=conv,
            ))


# ── 主函数 ────────────────────────────────────────────────

def demo_round_trip() -> None:
    print("往返: 2026 JSON-RPC / REST <-> FIPA-ACL 信封")
    print("=" * 72)

    mcp_call = {"jsonrpc": "2.0", "method": "tools/call",
                "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}}, "id": 42}
    print("\n-- MCP tools/call --")
    print(mcp_call)
    print("as ACL:")
    print(mcp_tools_call_to_acl(mcp_call).render())

    a2a_task = {"client": "research-host", "agent": "code-review-agent",
                "skill": "review-python", "input": "def f(x): return x", "task_id": "t-12"}
    print("\n-- A2A POST /tasks --")
    print(a2a_task)
    print("as ACL:")
    print(a2a_task_create_to_acl(a2a_task).render())


def demo_contract_net() -> None:
    cn = ContractNet(manager="scheduler", bidders=["worker-a", "worker-b", "worker-c"])
    conv = "cn-1"

    cn.cfp(task="compress 10GB log bundle", conv=conv)
    cn.propose("worker-a", Bid("worker-a", 3, 18), conv)
    cn.propose("worker-b", Bid("worker-b", 2, 25), conv)
    cn.propose("worker-c", Bid("worker-c", 4, 10), conv)

    proposes = [m for m in cn.log if m.performative == "propose"]
    winner = min(proposes, key=lambda m: m.content["price"] + m.content["eta_minutes"] / 10)
    losers = [m.sender for m in proposes if m.sender != winner.sender]
    cn.award(winner.sender, losers, conv)

    print("\n合同网协商")
    print("=" * 72)
    for msg in cn.log:
        print(msg.render())

    print(f"\n赢家: {winner.sender} (价格 {winner.content['price']}, "
          f"交付 {winner.content['eta_minutes']} 分钟)")


def main() -> None:
    print("=" * 72)
    print("FIPA-ACL 转换器与合同网演示（阶段 16，第 2 课）")
    print("=" * 72)
    demo_round_trip()
    demo_contract_net()
    print("\n要点: MCP/A2A 消息是具有 JSON 语法的 FIPA-ACL 信封。")
    print("结构原语存活；本体论和形式语义没有。")


if __name__ == "__main__":
    main()
