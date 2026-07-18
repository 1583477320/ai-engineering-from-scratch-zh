"""案例映射器：为设计方案推荐最近的 2026 年参考案例。"""

from dataclasses import dataclass


@dataclass
class Design:
    name: str
    task_type: str           # "research" | "engineering" | "population" | "automation"
    n_agents_expected: int
    verification_required: bool
    runtime_duration_hours: float
    roles_distinct: bool
    user_facing_network: bool


CASES = {
    "anthropic_research": {
        "name": "Anthropic Research (supervisor-worker)",
        "patterns": ["子智能体新上下文窗口", "编排者综合", "彩虹部署", "验证者角色"],
        "framework": "Anthropic Claude Agent SDK 或 LangGraph",
        "citation": "https://www.anthropic.com/engineering/multi-agent-research-system",
    },
    "metagpt_chatdev": {
        "name": "MetaGPT / ChatDev (SOP 角色分解)",
        "patterns": ["角色提示词编码 SOP", "结构化制品移交", "沟通式去幻觉", "DAG 路由扩展"],
        "framework": "CrewAI 或 MetaGPT 参考实现",
        "citation": "arXiv:2308.00352 / arXiv:2307.07924 / arXiv:2406.07155",
    },
    "openclaw_moltbook": {
        "name": "OpenClaw / Moltbook (群体规模基底)",
        "patterns": ["本地 ReAct 循环", "智能体间社交网络", "涌现经济", "提示词注入威胁模型"],
        "framework": "自定义基底 + MCP + A2A",
        "citation": "https://en.wikipedia.org/wiki/OpenClaw",
    },
}

FRAMEWORK_LANDSCAPE = [
    ("LangGraph", "生产", "结构化图 + 检查点 + 人工回环"),
    ("CrewAI", "生产", "基于角色的顺序/层级流程"),
    ("AG2", "社区维护", "GroupChat + 说话者选择"),
    ("Microsoft Agent Framework", "RC (2026.02)", "编排模式 + 企业集成"),
    ("OpenAI Agents SDK", "生产", "Swarm 继任者; tool-return handoff"),
    ("Google ADK", "生产 (2025.04)", "A2A 原生; Google Cloud"),
    ("Anthropic Claude Agent SDK", "生产", "单智能体 + Research 扩展"),
]


def map_to_case(d: Design) -> str:
    if d.task_type == "population" or d.user_facing_network:
        return "openclaw_moltbook"
    if d.task_type == "engineering" or d.roles_distinct:
        return "metagpt_chatdev"
    if d.task_type == "research":
        return "anthropic_research"
    if d.verification_required and d.runtime_duration_hours >= 1:
        return "anthropic_research"
    return "anthropic_research"


def main():
    designs = [
        Design("研究助手", "research", 6, True, 2.0, False, False),
        Design("代码生成团队", "engineering", 5, True, 1.0, True, False),
        Design("智能体市场", "population", 1000, False, 24.0, False, True),
        Design("内部自动化", "automation", 3, True, 0.5, True, False),
    ]

    print("=" * 70)
    print("案例映射器 — 设计方案 → 最接近的 2026 年参考案例")
    print("=" * 70)

    for d in designs:
        print(f"\n设计方案: {d.name}")
        print(f"  类型={d.task_type}  智能体数={d.n_agents_expected}  "
              f"验证={d.verification_required}  时长={d.runtime_duration_hours}h")
        case = CASES[map_to_case(d)]
        print(f"  推荐案例: {case['name']}")
        print(f"  可复用的模式:")
        for p in case["patterns"]:
            print(f"    - {p}")
        print(f"  推荐框架: {case['framework']}")

    print("\n" + "=" * 70)
    print("框架图景 — 2026 年 4 月")
    print("=" * 70)
    print(f"  {'框架':25s} {'状态':15s} {'最适合':30s}")
    for name, status, best_for in FRAMEWORK_LANDSCAPE:
        print(f"  {name:25s} {status:15s} {best_for:30s}")

    print("\n要点:")
    print("  从案例开始设计，不要从零开始。")
    print("  所有 2026 框架都支持 MCP 和 A2A。")
    print("  生产级多智能体需要：验证、成本核算、彩虹部署。")


if __name__ == "__main__":
    main()
