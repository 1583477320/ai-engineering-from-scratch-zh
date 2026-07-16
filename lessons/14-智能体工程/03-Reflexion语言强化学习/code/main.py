# Reflexion Agent 实现


class ReflexionAgent:
    """Reflexion 智能体——自我反思学习。"""
    def __init__(self, actor_fn, evaluator_fn, reflector_fn):
        self.actor_fn = actor_fn
        self.evaluator_fn = evaluator_fn
        self.reflector_fn = reflector_fn
        self.memory = []

    def run(self, task, max_trials=3):
        for trial in range(max_trials):
            result = self.actor_fn(task, context=self.memory)
            score, feedback = self.evaluator_fn(result)

            if score >= 0.8:
                return result, "成功"

            reflection = self.reflector_fn(task, result, feedback)
            self.memory.append(reflection)
            print(f"  试验 {trial+1}: 评分={score:.2f}, 反思={reflection[:60]}...")

        return None, "达到最大试验次数"


if __name__ == "__main__":
    print("Reflexion Agent 演示\n")

    def mock_actor(task, context=None):
        if len(context) < 1:
            return "初版代码：def add(a,b): return a+b"
        return "改进版：def add(a,b): return a+b  # 已添加类型检查"

    def mock_evaluator(result):
        if "类型检查" in result:
            return 0.9, "代码质量良好"
        return 0.3, "缺少类型检查和文档"

    def mock_reflector(task, result, feedback):
        return f"任务'{task[:10]}': {feedback} → 需要添加错误处理"

    agent = ReflexionAgent(mock_actor, mock_evaluator, mock_reflector)
    result, status = agent.run("写一个加法函数", max_trials=3)
    print(f"\n最终结果: {result}")
    print(f"状态: {status}")
    print(f"记忆: {len(agent.memory)} 条反思")
