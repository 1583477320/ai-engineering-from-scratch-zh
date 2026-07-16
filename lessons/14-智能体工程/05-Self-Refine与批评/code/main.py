# Self-Refine Agent 实现


class SelfRefineAgent:
    """Self-Refine——生成→批评→改进。"""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def generate(self, task):
        return self.llm_fn(f"写一个关于'{task}'的回复")

    def feedback(self, task, response):
        return self.llm_fn(f"评价这个回复: {response[:50]}...")

    def refine(self, task, response, feedback):
        return self.llm_fn(f"根据批评改进: {feedback[:30]}... 原始: {response[:30]}")

    def self_refine_loop(self, task, max_iterations=3):
        response = self.generate(task)
        print(f"  生成: {response[:60]}...")

        for i in range(max_iterations):
            feedback = self.feedback(task, response)
            print(f"  批评 ({i+1}): {feedback[:50]}...")
            response = self.refine(task, response, feedback)
            print(f"  改进 ({i+1}): {response[:60]}...")

        return response


if __name__ == "__main__":
    print("Self-Refine Agent 演示\n")

    def mock_llm(prompt):
        if "写" in prompt:
            return "初版: Python函数没有文档字符串"
        elif "评价" in prompt:
            return "缺少文档字符串和类型标注"
        elif "改进" in prompt:
            return "改进版: def add(a:int, b:int) -> int: '''两数相加'''\n  return a + b"
        return "回复"

    agent = SelfRefineAgent(mock_llm)
    result = agent.self_refine_loop("写一个加法函数", max_iterations=2)
    print(f"\n最终: {result[:80]}...")
