# 数据溯源与训练数据治理

> EU AI Act要求GPAI在2025年8月前实现机器可读的退出标准（通过EU版权指令TDM例外）。California AB 2013（2024年签署）——生成式AI训练数据透明性要求开发者发布包含12个强制字段的数据集摘要。2025年DPA对合法利益的一致立场：爱尔兰DPC（2025年5月21日）接受Meta在EU/EEA成人内容上训练LLM，需有保障措施。关键不可逆性问题：一旦数据进入模型权重，手术式擦除不可能——训练神经网络没有实用的GDPR删除权。

**类型：** 学习
**编程语言：** Python（标准库）
**前置知识：** 第18章 · 第24节（监管框架），第18章 · 第26节（卡片）
**预计时间：** 约60分钟

---

## 学习目标

- 描述California AB 2013的12个生成式AI训练数据透明性强制字段
- 阐述2025年DPA对合法利益LLM训练的立场（爱尔兰DPC、UK ICO、汉堡、科隆）
- 描述不可逆性问题：为什么GDPR删除权对训练神经网络没有实用等价物
- 阐述Data Provenance Initiative的"同意危机"发现

---

## 1. 问题

训练数据治理是每个模型卡（第26节）和监管义务（第24节）的上游。2024-2025年，监管格局在三个原则上巩固：退出基础设施、每数据集披露和公开可用数据的合法利益。不在收集时合规的提供者无法在下游补救。

---

## 2. 核心概念

### 2.1 California AB 2013

2024年签署。文档必须在2026年1月1日前发布，适用于2022年1月1日后发布的系统。

第3111(a)条要求开发者发布训练数据集的高层摘要，包含12个法定项目：
1. 数据集的来源或所有者
2. 数据集如何促进AI系统的预期目的
3. 数据集中的数据点数量（接受大致范围；动态数据集接受估计）
4. 数据点类型的描述（标注数据集的标签类型；未标注数据集的一般特征）
5. 数据集是否包含受版权、商标或专利保护的数据，或完全属于公共领域
6. 数据集是否被购买或许可
7. 数据集是否包含个人信息（根据Cal. Civ. Code §1798.140(v)）
8. 数据集是否包含汇总消费者信息（根据Cal. Civ. Code §1798.140(b)）
9. 开发者的清理、处理或其他修改，及预期目的
10. 数据收集的时间段，如收集仍在进行需注明
11. 数据集在开发期间首次使用的日期
12. 系统是否使用或持续使用合成数据生成

第12项（合成数据）相对于Gebru等人2018年数据表是新增的。第7项（个人信息）触发隐私权利法（CPRA）义务。该法规豁免安全/完整性、飞机运行和联邦仅国家安全系统（第3111(b)条）。

### 2.2 EU AI Act（第24节）和TDM退出

EU版权指令文本和数据挖掘例外允许训练公开可用内容，除非权利人退出。EU AI Act GPAI实践准则版权章节要求GPAI提供者尊重机器可读的退出信号（robots.txt、C2PA"无AI训练"声明等）。

### 2.3 2025年DPA对合法利益的一致立场

**爱尔兰DPC（2025年5月21日）**：Meta在EU/EEA第一方公开成人用户内容上训练的计划在EDPB意见后被接受，需有保障措施。

**科隆高等地区法院（2025年5月23日）**：驳回对Meta的禁令：退出即足够。

**汉堡DPA**：为EU范围一致性放弃紧急程序。

**UK ICO（2025年9月23日）**：对LinkedIn恢复AI训练发出积极监管响应——非正式许可——需类似保障措施和持续监控。

一致原则：合法利益可以证明公开可用第一方内容的训练是合理的。不需要同意。

### 2.4 巴西ANPD（2024年6月）

因信息透明性不足暂停Meta使用巴西用户数据进行AI训练。与EU DPA不同的结果——ANPD优先考虑透明性而非合法利益的可接受性。

### 2.5 不可逆性问题

Cookie同意是为实时、可逆跟踪设计的。训练数据不同：一旦数据进入模型权重，手术式擦除不可能。从头重新训练是唯一的完全补救，但成本过高。

部分补救措施：
- **遗忘**：近似删除；通过MIA（第22节）测量
- **影响函数定位**：识别受数据影响最大的权重；选择性更新
- **微调抑制**：训练模型拒绝源自该数据的输出

没有一种完全解决问题。合规窗口在收集时。

### 2.6 Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee等人"同意危机"（2024年7月）：AI训练数据公地的大规模审计。发现：发布者以加速率添加robots.txt限制。公开可训练的公地正在快速收缩。2023年至2024年，约25%的顶级训练源添加了某种限制。

含义：未来的训练数据可用性取决于新的获取范式（许可、合成生成、激励参与）。

---

## 3. 从零实现

`code/main.py`为玩具数据集生成California AB 2013合规的12字段数据集摘要脚手架。

```python
"""California AB 2013数据集摘要脚手架——标准库Python。

为玩具数据集生成第3111(a)条要求的12项高层摘要。
识别特定项目触发的后续义务（个人信息标志->CPRA；
受版权保护标志->EU TDM退出尊重）。

使用方法：python3 code/main.py
"""


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
```

运行结果：

```
==========================================================================
CALIFORNIA AB 2013第3111(a)条12项生成器（第18章，第27节）
==========================================================================

# 数据集摘要（AB 2013第3111(a)条12项）

- **sources_or_owners**: 在仓库中通过Python random.gauss生成；所有者：本仓库
- **how_dataset_furthers_intended_purpose**: 第18章二元分类的教学演示
- **number_of_data_points (or range)**: 1,000个样本（固定种子）
- **types_of_data_points (label types or general characteristics)**: 两个实值特征；二元{0,1}标签
- **contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain**: N（完全合成；无第三方材料）
- **purchased_or_licensed (Y/N)**: N
- **contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))**: N
- **contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))**: N
- **cleaning_processing_or_modification_description**: 无（确定性生成）
- **data_collection_time_period (with ongoing-collection notice if applicable)**: 2026-04（单次运行，固定种子；非持续）
- **dates_first_used_during_development**: 2026-04-22
- **uses_synthetic_data_generation (Y/N)**: Y（整个数据集是合成的）

## 触发的后续义务
- 可能仍触发用于生成的基础模型的义务

==========================================================================
核心结论：第3111(a)条的12项是California基线。
第5项和第7项触发级联义务（EU TDM退出 + CPRA）。
EU AI Act GPAI实践准则版权章节要求退出尊重。
2025年DPA一致立场：合法利益 + 退出 = 合法。
合规窗口在收集时；不可逆性排除下游修复。
==========================================================================
```

---

## 4. 工具实践

**数据溯源工具：**
- Data Provenance Initiative工具包
- C2PA元数据工具
- robots.txt解析器

**合规检查：**
- AB 2013字段验证器
- EU TDM退出信号检测
- DPA合规检查清单

---

## 5. LLM视角

**不可逆性视角：**
训练数据一旦进入模型权重，无法手术式移除。这与传统的数据隐私权（GDPR删除权）形成根本冲突。

**合规视角：**
合规窗口在收集时。事后无法修复。这要求在数据管道中嵌入合规检查。

**公地视角：**
Data Provenance Initiative发现训练数据公地正在收缩。未来训练数据可用性取决于新的获取范式。

---

## 6. 工程最佳实践

**数据收集：**
- 在收集时嵌入合规检查
- 实现机器可读的退出信号
- 记录数据来源和许可

**文档：**
- 自动生成AB 2013摘要
- 维护数据溯源链
- 版本控制数据集

**监控：**
- 跟踪robots.txt变化
- 监控DPA立场变化
- 审计数据使用合规性

---

## 7. 常见错误

**错误1：假设事后可以合规**
症状：收集数据后才考虑AB 2013
修复：在数据管道中嵌入合规检查

**错误2：忽略退出信号**
症状：不尊重robots.txt或C2PA退出
修复：实现机器可读退出检测

**错误3：低估不可逆性**
症状：假设可以"删除"训练数据
修复：在收集时决定是否使用数据

---

## 8. 面试考点

**Q1：California AB 2013的12个强制字段是什么？**
考察：对监管要求的理解

**Q2：为什么GDPR删除权对训练神经网络没有实用等价物？**
考察：对技术限制的理解

**Q3：2025年DPA对合法利益的立场是什么？**
考察：对监管趋势的理解

**Q4：Data Provenance Initiative的"同意危机"发现是什么？**
考察：对数据公地问题的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AB 2013 | "California法律" | 生成式AI训练数据透明性；12个强制字段 |
| TDM例外 | "文本和数据挖掘" | EU版权指令训练数据例外，可退出 |
| 合法利益 | "EU基础" | GDPR第6条，可能证明公开内容训练合理 |
| 退出信号 | "机器可读的无训练" | robots.txt、C2PA"无AI训练"、TDM.Reservation |
| 不可逆性 | "无法取消训练" | 模型权重中的数据无法手术式移除 |
| 遗忘 | "近似删除" | 训练后干预，减少模型对特定数据的依赖 |
| 同意危机 | "DPI审计" | 2024年7月发现的robots.txt限制加速 |

---

## 参考文献

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013)
- [EU AI Act + GPAI实践准则（第24节）](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [Longpre、Mahari、Lee等人 — 同意危机（dataprovenance.org，2024年7月）](https://www.dataprovenance.org/consent-in-crisis-paper)
- [IAPP — EU数字综合GDPR修正案（2025）](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark)
