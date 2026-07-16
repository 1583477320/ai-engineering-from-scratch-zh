# 运行时选择器


def select_runtime(task_type, latency_requirement="realtime"):
    if latency_requirement == "realtime":
        if task_type in ["chat", "qa"]:
            return "request-response"
        elif task_type in ["generation"]:
            return "streaming"
    if latency_requirement == "batch":
        if task_type in ["data_processing"]:
            return "queue-based"
        elif task_type in ["maintenance"]:
            return "scheduled"
    if task_type in ["workflow"]:
        return "event-driven"
    return "durable-execution"


if __name__ == "__main__":
    print("运行时选择器演示\n")
    tasks = [("chat", "realtime"), ("generation", "realtime"), ("data_processing", "batch"), ("maintenance", "batch")]
    for task in tasks:
        runtime = select_runtime(*task)
        print(f"  {task[0]}({task[1]}) -> {runtime}")
