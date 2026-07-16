# GenAI Span 发射器


import time

class GenAISpanEmitter:
    def __init__(self):
        self.spans = []

    def start_agent_span(self, agent_id, query):
        span = {"name": f"agent.{agent_id}", "start": time.time(),
                "attrs": {"agent.id": agent_id, "input": query[:50]}}
        return span

    def start_llm_span(self, model, input_tokens, output_tokens):
        return {"name": "llm.generate", "start": time.time(),
                "attrs": {"model": model, "tokens": {"in": input_tokens, "out": output_tokens}}}

    def start_tool_span(self, tool_name, arguments):
        return {"name": f"tool.{tool_name}", "start": time.time(), "attrs": {"tool": tool_name}}

    def end(self, span):
        span["duration_ms"] = (time.time() - span["start"]) * 1000
        self.spans.append(span)
        return span


if __name__ == "__main__":
    print("GenAI 追踪演示\n")
    emitter = GenAISpanEmitter()
    s1 = emitter.start_agent_span("router", "查天气")
    s2 = emitter.start_llm_span("gpt-4o", 100, 50)
    emitter.end(s2)
    s3 = emitter.start_tool_span("get_weather", {"city": "北京"})
    emitter.end(s3)
    emitter.end(s1)
    for s in emitter.spans:
        print(f"  {s['name']:25s} {s.get('duration_ms', 0):.1f}ms")
