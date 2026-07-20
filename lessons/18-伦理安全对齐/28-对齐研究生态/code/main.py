"""对齐研究生态地图——标准库Python。

打印2026年非实验室对齐研究层的紧凑地图，
包含标准产出和交叉引用。

使用方法：python3 code/main.py
"""

from __future__ import annotations


ECOSYSTEM = [
    {
        "org": "MATS",
        "full_name": "ML对齐与理论学者",
        "scale": "自2021年以来527+研究者，180+论文，h指数47",
        "role": "人才管道 + 指导计划",
        "canonical_output": "90名学者 × 10-12周训练营 -> 实验室和外部评估者",
    },
    {
        "org": "Redwood",
        "full_name": "Redwood Research",
        "scale": "由Buck Shlegeris创立；应用对齐实验室",
        "role": "AI控制议程；UK AISI合作伙伴",
        "canonical_output": "Greenblatt、Shlegeris等人 AI Control（ICML 2024）",
    },
    {
        "org": "Apollo",
        "full_name": "Apollo Research",
        "scale": "前沿实验室的预部署策略评估",
        "role": "三支柱策略分解",
        "canonical_output": "Meinke等人 上下文策略（arXiv:2412.04984）",
    },
    {
        "org": "METR",
        "full_name": "模型评估与威胁研究",
        "scale": "任务时间跨度评估；框架综合",
        "role": "外部跨实验室比较",
        "canonical_output": "前沿AI安全政策的共同要素（2025）",
    },
    {
        "org": "Eleos",
        "full_name": "Eleos AI Research",
        "scale": "模型福利预部署评估",
        "role": "福利方法论检查",
        "canonical_output": "Claude Opus 4福利评估（系统卡5.3）",
    },
]


def main() -> None:
    print("=" * 78)
    print("对齐研究生态（第18章，第28节）")
    print("=" * 78)
    for org in ECOSYSTEM:
        print(f"\n{org['org']}（{org['full_name']}）")
        print(f"  规模              : {org['scale']}")
        print(f"  角色              : {org['role']}")
        print(f"  标准产出          : {org['canonical_output']}")

    print("\n" + "=" * 78)
    print("核心结论：外部评估提供结构性可信度。")
    print("仅实验室内部评估存在利益冲突；")
    print("多组织出版物（如Apollo + OpenAI、Redwood + Anthropic）")
    print("是质量控制。MATS是人才管道。UK AISI/CAISI")
    print("是监管对应方（第24节）。")
    print("=" * 78)


if __name__ == "__main__":
    main()
