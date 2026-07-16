# OpenAI Agents SDK：Handoff + Guardrail


class SimpleAgent:
    """简化版 OpenAI Agent。"""
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.handoffs = {}

    def add_handoff(self, agent):
        self.handoffs[f"transfer_to_{agent.name}"] = agent

    def run(self, input_text):
        for tool, agent in self.handoffs.items():
            if tool in input_text:
                return {"action": "handoff", "target": agent.name}
        return {"action": "respond", "content": f"[{self.name}] {input_text[:30]}"}


def input_guardrail(text):
    """输入护栏。"""
    if any(p in text for p in ["注入", "系统提示"]):
        return False, "检测到危险输入"
    return True, "安全"


if __name__ == "__main__":
    print("OpenAI Agents SDK 演示\n")
    support = SimpleAgent("support", "处理技术支持")
    router = SimpleAgent("router", "路由请求")
    router.add_handoff(support)

    for q in ["帮我查天气", "转接到技术支持", "如何配置API"]:
        result = router.run(q)
        print(f"  '{q[:15]}...' -> {result['action']}: {result.get('content', result.get('target', ''))}")
