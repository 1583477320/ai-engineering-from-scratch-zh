# 智能体框架选型分析


FRAMEWORKS = {
    "LangGraph": {
        "type": "编排层", "language": "Python",
        "strength": "有状态图、可调试、检查点",
        "best_for": "复杂多步流水线、需要人工审批",
        "learning_curve": "中等",
    },
    "CrewAI": {
        "type": "多智能体", "language": "Python",
        "strength": "角色分工、团队协作",
        "best_for": "多 Agent 协作任务",
        "learning_curve": "低",
    },
    "Dify": {
        "type": "低代码平台", "language": "无代码/Python",
        "strength": "可视化编排、无代码",
        "best_for": "快速原型、非工程师",
        "learning_curve": "低",
    },
    "Coze": {
        "type": "Agent 平台", "language": "中文优先",
        "strength": "中文支持好、插件生态",
        "best_for": "中文 Agent 场景",
        "learning_curve": "低",
    },
    "AutoGen": {
        "type": "多智能体", "language": "Python",
        "strength": "对话式协作、灵活",
        "best_for": "研究探索、多 Agent 对话",
        "learning_curve": "高",
    },
}


def recommend_framework(requirements):
    """根据需求推荐框架。"""
    scores = {}
    for name, info in FRAMEWORKS.items():
        score = 0
        if requirements.get("chinese") and "中文" in info.get("strength", ""):
            score += 3
        if requirements.get("multi_agent") and "多智能体" in info["type"]:
            score += 3
        if requirements.get("complex_flow") and "编排" in info["type"]:
            score += 3
        if requirements.get("no_code"):
            score += 2 if info["learning_curve"] == "低" else 0
        if requirements.get("production") and "可调试" in info.get("strength", ""):
            score += 2
        scores[name] = score

    best = max(scores, key=scores.get)
    return best, FRAMEWORKS[best]


if __name__ == "__main__":
    print("智能体框架选型分析\n")

    print("框架对比:")
    for name, info in FRAMEWORKS.items():
        print(f"  {name}: {info['type']} | {info['best_for']} | 学习曲线: {info['learning_curve']}")

    print("\n推荐:")
    scenarios = [
        {"chinese": True, "no_code": True},
        {"multi_agent": True},
        {"complex_flow": True, "production": True},
    ]
    labels = ["中文低代码", "多 Agent", "生产级流水线"]
    for req, label in zip(scenarios, labels):
        name, info = recommend_framework(req)
        print(f"  {label}: {name} ({info['best_for']})")
