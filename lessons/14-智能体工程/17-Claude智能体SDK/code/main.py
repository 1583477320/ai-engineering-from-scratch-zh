# Claude Agent SDK 概念实现


class ClaudeAgentSDK:
    """简化版 Claude Agent SDK。"""
    def __init__(self, model="claude-sonnet-5"):
        self.model = model
        self.tools = []
        self.sub_agents = {}
        self.hooks = {"pre_call": [], "post_call": []}

    def add_tool(self, name, handler):
        self.tools.append({"name": name, "handler": handler})

    def add_hook(self, event, handler):
        self.hooks[event].append(handler)

    def add_subagent(self, name, agent):
        self.sub_agents[name] = agent

    def run(self, prompt):
        for hook in self.hooks["pre_call"]:
            prompt = hook(prompt)
        response = f"[Claude] 处理: {prompt[:50]}..."
        for hook in self.hooks["post_call"]:
            response = hook(response)
        return response


class SubAgent:
    """子智能体——独立上下文。"""
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions

    def run(self, task):
        return f"[{self.name}] 完成: {task[:30]}"


if __name__ == "__main__":
    print("Claude Agent SDK 演示\n")

    sdk = ClaudeAgentSDK()
    sdk.add_hook("pre_call", lambda x: f"[前置检查] {x}")
    sdk.add_hook("post_call", lambda x: f"{x} [记录完成]")

    sub = SubAgent("代码审查", "审查代码质量")
    sdk.add_subagent("reviewer", sub)

    result = sdk.run("审查一个 Python 脚本")
    print(f"主智能体: {result}")

    sub_result = sub.run("检查安全性")
    print(f"子智能体: {sub_result}")
