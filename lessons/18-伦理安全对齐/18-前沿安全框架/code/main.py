"""前沿安全框架对比——三大实验室结构映射。"""


def compare_frameworks():
    """对比三大安全框架的结构。"""
    frameworks = {
        "Anthropic RSP": {
            "version": "v3.0 (2026.02)",
            "tiers": "ASL-1 到 ASL-5+",
            "asr3": "CBRN 相关能力，2025.05 激活",
            "evaluation": "阈值触发式评估",
            "adjustment": "是",
        },
        "OpenAI PF": {
            "version": "v2 (2025.04)",
            "tiers": "五标准追踪",
            "asr3": "合理+可测量+严重+新出现+瞬时",
            "evaluation": "能力报告+保障报告分离",
            "adjustment": "是",
        },
        "DeepMind FSF": {
            "version": "v3.0 (2025.09)",
            "tiers": "CCL（按领域）",
            "asr3": "生物/网络/ML研发/有害操纵",
            "evaluation": "领域特定评估",
            "adjustment": "是",
        },
    }

    print(f"{'框架':20s} {'版本':20s} {'分层':25s} {'调整条款':>6}")
    print("-" * 75)
    for name, f in frameworks.items():
        print(f"{name:20s} {f['version']:20s} {f['tiers']:25s} {f['adjustment']:>6}")


if __name__ == "__main__":
    compare_frameworks()
