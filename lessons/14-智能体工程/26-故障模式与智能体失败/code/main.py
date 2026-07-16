# 智能体故障检测器


class FaultDetector:
    def __init__(self):
        self.tool_calls = []
        self.step_results = []

    def check_tool_abuse(self, tool_calls, max_per_minute=10):
        recent = tool_calls[-max_per_minute:]
        if len(recent) > max_per_minute:
            return True, "工具调用频率异常"
        return False, None

    def check_cascading_error(self, step_results):
        errors = sum(1 for r in step_results if r.get("error"))
        if errors > len(step_results) * 0.5:
            return True, "超过50%步骤出错——级联错误"
        return False, None

    def check_scope_creep(self, completed_steps):
        if len(completed_steps) > 20:
            return True, f"已完成{len(completed_steps)}步——可能存在范围蔓延"
        return False, None


FAULT_CATEGORIES = {
    "planning": ["规划错误", "目标偏移", "步骤遗漏"],
    "execution": ["工具调用错误", "参数错误", "超时"],
    "coordination": ["通信失败", "状态不一致", "死锁"],
}

def classify_fault(error_msg):
    for cat, patterns in FAULT_CATEGORIES.items():
        if any(p in error_msg for p in patterns):
            return cat
    return "unknown"


if __name__ == "__main__":
    print("故障检测器演示\n")
    detector = FaultDetector()

    # 工具滥用检测
    tool_calls = [{"tool": "search"} for _ in range(15)]
    ok, msg = detector.check_tool_abuse(tool_calls)
    print(f"工具滥用: {ok}, {msg}")

    # 级联错误检测
    step_results = [{"error": True} for _ in range(10)] + [{"error": False}]
    ok, msg = detector.check_cascading_error(step_results)
    print(f"级联错误: {ok}, {msg}")

    # 故障分类
    print(f"分类 '工具调用超时': {classify_fault('工具调用超时')}")
    print(f"分类 '状态不一致': {classify_fault('状态不一致')}")
