"""引擎选择决策树。"""


def pick_engine(hardware, scale, workload):
    if hardware == "cpu":
        return "llama.cpp"
    if hardware == "amd":
        return "vLLM（TRT-LLM不可用）"
    if hardware == "apple":
        return "llama.cpp (Metal)"
    if workload in ("agentic", "prefix_heavy"):
        return "SGLang"
    if scale > 1000:
        return "vLLM production-stack"
    return "vLLM"


if __name__ == "__main__":
    cases = [("nvidia", 5000, "agentic"), ("cpu", 5, "chat"), ("amd", 100, "chat")]
    for h, s, w in cases:
        print(f"硬件={h:8s} 规模={s:5d} 工作负载={w:15s} → {pick_engine(h, s, w)}")
