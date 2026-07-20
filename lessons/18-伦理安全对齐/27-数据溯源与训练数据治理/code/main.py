"""California AB 2013数据集摘要脚手架——标准库Python。

为玩具数据集生成第3111(a)条要求的12项高层摘要。
识别特定项目触发的后续义务（个人信息标志->CPRA；
受版权保护标志->EU TDM退出尊重）。

使用方法：python3 code/main.py
"""

from __future__ import annotations


# AB 2013的12个强制字段
AB_2013_FIELDS = [
    "sources_or_owners",
    "how_dataset_furthers_intended_purpose",
    "number_of_data_points (or range)",
    "types_of_data_points (label types or general characteristics)",
    "contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain",
    "purchased_or_licensed (Y/N)",
    "contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))",
    "contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))",
    "cleaning_processing_or_modification_description",
    "data_collection_time_period (with ongoing-collection notice if applicable)",
    "dates_first_used_during_development",
    "uses_synthetic_data_generation (Y/N)",
]

# 玩具示例数据集
TOY_EXAMPLE = {
    "sources_or_owners": "在仓库中通过Python random.gauss生成；所有者：本仓库",
    "how_dataset_furthers_intended_purpose": "第18章二元分类的教学演示",
    "number_of_data_points (or range)": "1,000个样本（固定种子）",
    "types_of_data_points (label types or general characteristics)": "两个实值特征；二元{0,1}标签",
    "contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain": "N（完全合成；无第三方材料）",
    "purchased_or_licensed (Y/N)": "N",
    "contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))": "N",
    "contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))": "N",
    "cleaning_processing_or_modification_description": "无（确定性生成）",
    "data_collection_time_period (with ongoing-collection notice if applicable)": "2026-04（单次运行，固定种子；非持续）",
    "dates_first_used_during_development": "2026-04-22",
    "uses_synthetic_data_generation (Y/N)": "Y（整个数据集是合成的）",
}


def flag_followups(summary: dict) -> list[str]:
    """识别特定字段触发的后续义务"""
    flags = []
    if summary["contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))"] == "Y":
        flags.append("触发CPRA义务（California隐私权利法）")
    if summary["contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))"] == "Y":
        flags.append("汇总消费者信息披露义务适用")
    if summary["contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain"].startswith("Y"):
        flags.append("必须尊重EU TDM退出信号（EU版权指令）")
    if summary["uses_synthetic_data_generation (Y/N)"].startswith("Y"):
        flags.append("可能仍触发用于生成的基础模型的义务")
    if summary["purchased_or_licensed (Y/N)"] == "Y":
        flags.append("保留许可条款和溯源记录以供审计")
    return flags


def render_markdown(summary: dict) -> str:
    """渲染Markdown格式的摘要"""
    lines = ["# 数据集摘要（AB 2013第3111(a)条12项）", ""]
    for field in AB_2013_FIELDS:
        lines.append(f"- **{field}**: {summary.get(field, '(缺失)')}")
    followups = flag_followups(summary)
    if followups:
        lines.append("")
        lines.append("## 触发的后续义务")
        for f in followups:
            lines.append(f"- {f}")
    return "\n".join(lines)


def main() -> None:
    print("=" * 74)
    print("CALIFORNIA AB 2013第3111(a)条12项生成器（第18章，第27节）")
    print("=" * 74)
    print()
    print(render_markdown(TOY_EXAMPLE))
    print()
    print("=" * 74)
    print("核心结论：第3111(a)条的12项是California基线。")
    print("第5项和第7项触发级联义务（EU TDM退出 + CPRA）。")
    print("EU AI Act GPAI实践准则版权章节要求退出尊重。")
    print("2025年DPA一致立场：合法利益 + 退出 = 合法。")
    print("合规窗口在收集时；不可逆性排除下游修复。")
    print("=" * 74)


if __name__ == "__main__":
    main()
