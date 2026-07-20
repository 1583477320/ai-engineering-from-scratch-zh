"""双重用途分类表——标准库Python。

打印2024-2025年跨领域双重用途图景作为表格。
仅供参考；主要来源见docs/index.md。

使用方法：python3 code/main.py
"""

from __future__ import annotations


DOMAINS = [
    {
        "domain": "生物",
        "2024_state": "温和提升",
        "2025_state": "2.53倍新手相对提升；接近ASL-3",
        "inflection": "获取阶段自动化",
        "bottleneck_remaining": "病原体采购、生物安全设备",
    },
    {
        "domain": "化学",
        "2024_state": "温和提升",
        "2025_state": "视觉启用LLM的执行差距侵蚀",
        "inflection": "实时湿实验协议纠正",
        "bottleneck_remaining": "前体采购、专业设备",
    },
    {
        "domain": "网络",
        "2024_state": "代码片段辅助",
        "2025_state": "80-90%活动自动化（Anthropic 2025年11月）",
        "inflection": "智能体编码工作流",
        "bottleneck_remaining": "4-6个人类干预步骤",
    },
    {
        "domain": "核",
        "2024_state": "有限",
        "2025_state": "有限",
        "inflection": "（2024-2025年无重大拐点报告）",
        "bottleneck_remaining": "裂变材料获取主导",
    },
]


def main() -> None:
    print("=" * 82)
    print("2026年双重用途图景（第18章，第30节）")
    print("=" * 82)

    for d in DOMAINS:
        print(f"\n{d['domain'].upper()}")
        print(f"  2024年状态            : {d['2024_state']}")
        print(f"  2025年状态            : {d['2025_state']}")
        print(f"  拐点                  : {d['inflection']}")
        print(f"  剩余瓶颈              : {d['bottleneck_remaining']}")

    print("\n" + "=" * 82)
    print("核心结论：四个CBRN领域中有三个在2025年跨越了阈值。")
    print("生物：2.53倍提升，接近ASL-3。化学：执行差距侵蚀。")
    print("网络：80-90%活动智能体自动化。核仍受材料获取约束。")
    print("安全案例必须同时针对新手相对和专家绝对；")
    print("仅输入过滤防御不足。")
    print("=" * 82)


if __name__ == "__main__":
    main()
