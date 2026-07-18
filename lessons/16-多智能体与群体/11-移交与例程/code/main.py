"""移交驱动编排——OpenAI Swarm 微型实现。

两个原语：
  - Agent(name, instructions, functions)
  - handoff = 一个返回 Agent 的工具

运行循环检测 Agent 类型的返回并切换活跃智能体。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass
class Agent:
    name: str
    instructions: str
    functions: list[Callable] = field(default_factory=list)


@dataclass
class Msg:
    role: str
    content: str
    sender: Optional[str] = None


# ── 智能体工厂 ──────────────────────────────────────────

def triage_agent_factory():
    def transfer_to_refunds(): return refund_agent
    def transfer_to_sales(): return sales_agent
    def transfer_to_support(): return support_agent
    return Agent("triage", "路由用户到：退款、销售或支持。",
                 [transfer_to_refunds, transfer_to_sales, transfer_to_support])

def refund_agent_factory():
    def process_refund(order_id: str) -> str:
        return f"退款已处理，订单 {order_id}。"
    return Agent("refund", "处理退款请求。", [process_refund])

def sales_agent_factory():
    def quote_product(product: str) -> str:
        return f"{product} 报价：$99/月。"
    return Agent("sales", "处理销售咨询。", [quote_product])

def support_agent_factory():
    def open_ticket(issue: str) -> str:
        return f"工单已创建：{issue}"
    return Agent("support", "处理技术支持。", [open_ticket])


triage_agent = triage_agent_factory()
refund_agent = refund_agent_factory()
sales_agent = sales_agent_factory()
support_agent = support_agent_factory()


# ── 路由器（LLM 替身） ──────────────────────────────────

def scripted_router(current: Agent, user_msg: str) -> Union[str, Agent]:
    """模拟 LLM 路由——读取用户消息并调用移交工具。"""
    text = user_msg.lower()
    if current.name == "triage":
        if "refund" in text or "money back" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_refunds")()
        if "buy" in text or "price" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_sales")()
        if "broken" in text or "bug" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_support")()
        return "您需要什么帮助？"
    if current.name == "refund":
        order = next((w for w in user_msg.split() if w.isdigit()), "42")
        return next(f for f in current.functions if f.__name__ == "process_refund")(order)
    if current.name == "sales":
        return next(f for f in current.functions if f.__name__ == "quote_product")("enterprise plan")
    if current.name == "support":
        return next(f for f in current.functions if f.__name__ == "open_ticket")(user_msg)
    return "[no response]"


# ── 运行循环 ──────────────────────────────────────────────

def run_swarm(start_agent: Agent, user_messages: list[str]) -> list[Msg]:
    history: list[Msg] = []
    active = start_agent
    for user in user_messages:
        history.append(Msg(role="user", content=user))
        out = scripted_router(active, user)
        if isinstance(out, Agent):
            history.append(Msg(role="assistant", content=f"(移交到 {out.name})", sender=active.name))
            active = out
            out = scripted_router(active, user)
        history.append(Msg(role="assistant", content=str(out), sender=active.name))
    return history


def render(history: list[Msg]) -> None:
    for m in history:
        tag = m.sender if m.sender else m.role
        print(f"  [{tag:>8s}]: {m.content}")


# ── 主函数 ────────────────────────────────────────────────

def main():
    print("移交驱动编排——OpenAI Swarm 形状")
    print("-" * 54)

    scenarios = [
        ("退款流程", ["I need a refund on order 77"]),
        ("销售流程", ["I want to buy the enterprise plan. what's the price?"]),
        ("支持流程", ["my dashboard is broken"]),
        ("模糊", ["hello"]),
    ]
    for label, msgs in scenarios:
        print(f"\n=== {label} ===")
        history = run_swarm(triage_agent, msgs)
        render(history)

    print("\n要点: 每次移交都是一个返回 Agent 的工具调用。")
    print("框架的唯一工作是检测 Agent 类型的返回并切换活跃智能体。")
    print("没有状态机。没有 DSL。智能体提示词就是路由逻辑。")


if __name__ == "__main__":
    main()
