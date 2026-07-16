# 智能体追踪器


import time

class SimpleTracer:
    def __init__(self):
        self.traces = []

    def trace(self, name, fn, **kwargs):
        start = time.time()
        try:
            result = fn(**kwargs)
            latency = (time.time() - start) * 1000
            self.traces.append({"name": name, "latency_ms": latency, "status": "ok"})
            return result
        except Exception as e:
            latency = (time.time() - start) * 1000
            self.traces.append({"name": name, "latency_ms": latency, "status": "error", "error": str(e)})
            raise

    def summary(self):
        ok = sum(1 for t in self.traces if t["status"] == "ok")
        total = len(self.traces)
        avg = sum(t["latency_ms"] for t in self.traces) / max(total, 1)
        return {"总 span": total, "成功": ok, "错误率": f"{(total-ok)/max(total,1):.1%}", "平均延迟": f"{avg:.1f}ms"}


if __name__ == "__main__":
    print("智能体追踪器演示\n")
    tracer = SimpleTracer()
    tracer.trace("llm.generate", lambda: time.sleep(0.01))
    tracer.trace("tool.get_weather", lambda: time.sleep(0.005))
    tracer.trace("llm.generate", lambda: time.sleep(0.008))
    print(f"追踪摘要: {tracer.summary()}")
