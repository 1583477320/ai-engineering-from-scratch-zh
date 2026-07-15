# LangGraph 状态机模拟

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class State:
    """对话状态。"""
    messages: list = field(default_factory=list)
    confidence: float = 0.0
    step: str = "start"


class SimpleStateGraph:
    """简化版状态图——演示 LangGraph 核心概念。"""
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.conditional_edges = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))

    def add_conditional_edge(self, from_node, condition_fn, routes):
        self.conditional_edges[from_node] = (condition_fn, routes)

    def invoke(self, state):
        current = "start"
        for _ in range(20):  # 防止无限循环
            if current not in self.nodes:
                break
            result = self.nodes[current](state)
            state.update(result)
            # 检查条件边
            if current in self.conditional_edges:
                cond_fn, routes = self.conditional_edges[current]
                current = routes[cond_fn(state)]
            else:
                # 无条件边：找下一个
                next_node = None
                for from_n, to_n in self.edges:
                    if from_n == current:
                        next_node = to_n
                        break
                if next_node is None:
                    break
                current = next_node
        return state


def build_customer_service_graph():
    """构建客服状态机。"""
    graph = SimpleStateGraph()

    def understand(state):
        query = state["messages"][-1] if state["messages"] else ""
        return {"step": "understood", "messages": state["messages"] + [f"[理解] {query[:20]}..."]}

    def retrieve(state):
        return {"step": "retrieved", "messages": state["messages"] + ["检索到相关文档"], "confidence": 0.7}

    def respond(state):
        return {"step": "responded", "messages": state["messages"] + ["基于文档的回答"]}

    def transfer(state):
        return {"step": "transferred", "messages": state["messages"] + ["转接人工客服"]}

    def route(state):
        if state.get("confidence", 0) < 0.5:
            return "transfer"
        return "respond"

    graph.add_node("start", lambda s: {"messages": s.get("messages", [])})
    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_node("transfer", transfer)

    graph.add_edge("start", "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_conditional_edge("retrieve", route, {"respond": "respond", "transfer": "transfer"})

    return graph


if __name__ == "__main__":
    print("LangGraph 状态机演示\n")

    graph = build_customer_service_graph()
    app = graph.compile()

    # 测试高置信度路径
    result = app.invoke({"messages": ["退货政策是什么？"], "confidence": 0.8})
    print("路径 1 (高置信度):")
    for msg in result["messages"]:
        print(f"  {msg}")

    # 测试低置信度路径
    result = app.invoke({"messages": ["复杂的法律问题"], "confidence": 0.3})
    print("\n路径 2 (低置信度):")
    for msg in result["messages"]:
        print(f"  {msg}")
