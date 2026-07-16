# 多模态智能体：感知-推理-行动循环


class MultimodalAgent:
    """简化版多模态智能体。"""
    def __init__(self):
        self.memory = []
        self.step_count = 0

    def step(self, observation):
        # 1. 感知（简化：直接解析观察）
        understanding = f"屏幕内容: {observation[:50]}..."

        # 2. 推理（简化：规则决策）
        if "登录" in observation:
            action = {"type": "click", "x": 300, "y": 150}
        elif "搜索" in observation:
            action = {"type": "type", "text": "搜索内容"}
        else:
            action = {"type": "wait", "duration": 1}

        # 3. 执行（简化：打印操作）
        result = f"执行了 {action['type']} 操作"

        # 4. 记忆
        self.memory.append({"observation": understanding, "action": action})
        self.step_count += 1

        return {"action": action, "result": result}

    def reset(self):
        self.memory = []
        self.step_count = 0


def execute_workflow(agent, task_steps, max_steps=10):
    """执行多步骤工作流。"""
    agent.reset()
    for i, step in enumerate(task_steps[:max_steps]):
        result = agent.step(step)
        print(f"  步骤 {i+1}: {result['result']}")
    return {"completed": len(task_steps) <= max_steps, "steps": len(task_steps)}


if __name__ == "__main__":
    print("多模态智能体演示\n")
    agent = MultimodalAgent()
    steps = ["打开浏览器", "搜索天气", "点击登录按钮", "填写表单", "提交报告"]
    result = execute_workflow(agent, steps)
    print(f"完成: {result['completed']}, 步数: {result['steps']}")
