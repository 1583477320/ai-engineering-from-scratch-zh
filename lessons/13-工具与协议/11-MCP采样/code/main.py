# MCP 采样：Server 请求的 LLM 补全


class SamplingHandler:
    """MCP 采样请求处理器。"""
    def __init__(self):
        self.llm_fn = None

    def set_llm(self, llm_fn):
        self.llm_fn = llm_fn

    def create_message(self, messages, model=None, max_tokens=1024):
        if not self.llm_fn:
            return {"error": "未配置 LLM"}
        response = self.llm_fn(messages, model=model, max_tokens=max_tokens)
        return {"role": "assistant", "content": response, "model": model}


class ServerHostedLoop:
    """Server 主持的智能体循环。"""
    def __init__(self, sampler):
        self.sampler = sampler

    def analyze_code(self, code):
        """Server 调用 Client 的 LLM 分析代码。"""
        analysis = self.sampler.create_message([
            {"role": "user", "content": f"分析以下代码并给出改进建议：\n{code}"}
        ])
        return analysis


if __name__ == "__main__":
    print("MCP 采样演示\n")

    sampler = SamplingHandler()
    sampler.set_llm(lambda msgs, **kw: f"分析结果：{msgs[0]['content'][:30]}... 的改进：建议使用更清晰的变量名。")

    loop = ServerHostedLoop(sampler)
    result = loop.analyze_code("def foo(): pass")
    print(f"代码分析: {result['content'][:60]}...")

    print("\n采样流程:")
    print("  1. 用户请求 → Server")
    print("  2. Server 需要 LLM 推理 → sampling/createMessage")
    print("  3. Client 使用自己的 LLM 生成 → 返回 Server")
    print("  4. Server 继续处理 → 完成响应")
