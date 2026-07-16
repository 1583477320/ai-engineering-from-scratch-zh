# OpenTelemetry GenAI Span 发射器

import time


class SpanEmitter:
    """简化版 OTel Span 发射器。"""
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None):
        return {"name": name, "start_time": time.time(), "end_time": None,
                "attributes": attributes or {}, "status": "ok"}

    def end_span(self, span, status="ok", error=None):
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        span["status"] = status
        if error:
            span["error"] = str(error)
        self.spans.append(span)
        return span

    def emit_llm_span(self, model, input_tokens, output_tokens, duration_ms):
        span = self.start_span("llm.generate", {
            "gen_ai.system": "openai", "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        })
        span["duration_ms"] = duration_ms
        return self.end_span(span)

    def emit_tool_span(self, tool_name, duration_ms):
        return self.end_span(
            self.start_span(f"tool.{tool_name}", {"tool.name": tool_name}),
            status="ok",
        )


if __name__ == "__main__":
    print("OpenTelemetry GenAI 追踪演示\n")
    emitter = SpanEmitter()

    s1 = emitter.emit_llm_span("gpt-4o", 150, 50, 1200)
    s2 = emitter.emit_tool_span("get_weather", 200)
    s3 = emitter.emit_llm_span("gpt-4o", 200, 150, 2500)

    for s in emitter.spans:
        print(f"  {s['name']:20s} {s['duration_ms']:.0f}ms {s['status']}")
    total = sum(s["duration_ms"] for s in emitter.spans)
    print(f"  总耗时: {total:.0f}ms")
