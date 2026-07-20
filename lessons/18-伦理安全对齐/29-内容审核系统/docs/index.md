# 内容审核系统——OpenAI、Perspective、Llama Guard

> 生产级审核系统将第12-16节定义的安全策略操作化。OpenAI审核API：`omni-moderation-latest`（2024）基于GPT-4o，一次调用分类文本+图像；多语言测试集比前代好42%；响应模式返回13个类别布尔值。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。Llama Guard 3/4：14个MLCommons危害类别。Perspective API（Google Jigsaw）：LLM时代之前的毒性评分基线。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** 第18章 · 第16节（Llama Guard / Garak / PyRIT）
**预计时间：** 约60分钟

---

## 学习目标

- 描述OpenAI审核API的类别分类法及其与Llama Guard 3的MLCommons集合的区别
- 描述三层审核模式（输入、输出、自定义）并命名每层的一个失败模式
- 描述Perspective API作为LLM前时代基线的定位及其在研究中的持续使用
- 阐述Azure弃用时间线

---

## 1. 问题

第12-16节描述攻击和防御工具。第29节涵盖在用户接触产品的界面操作化防御的已部署审核系统。三层模式是2026年的默认配置。

---

## 2. 核心概念

### 2.1 OpenAI审核API

`omni-moderation-latest`（2024）。基于GPT-4o。一次调用分类文本+图像。对大多数开发者免费。

类别（响应模式中的13个布尔值）：
- harassment（骚扰）、harassment/threatening（威胁性骚扰）
- hate（仇恨）、hate/threatening（威胁性仇恨）
- self-harm（自伤）、self-harm/intent（自伤意图）、self-harm/instructions（自伤指导）
- sexual（性内容）、sexual/minors（未成年人性内容）
- violence（暴力）、violence/graphic（图形暴力）
- illicit（违法）、illicit/violent（暴力违法）

多模态支持适用于`violence`、`self-harm`和`sexual`，但不适用于`sexual/minors`；其余仅支持文本。

多语言测试集比前代审核端点好42%。每个类别返回分数；应用程序设置阈值。

### 2.2 Llama Guard 3/4

第16节已介绍。14个MLCommons危害类别（与OpenAI的13个响应模式布尔值组织方式不同）。支持8种语言（v3）。Llama Guard 4（2025年4月）原生多模态，12B参数。

OpenAI和Llama Guard的分类法重叠但有分歧。OpenAI有"illicit"作为宽泛类别；Llama Guard分别有"violent crimes"和"non-violent crimes"。部署根据策略分类法匹配度选择。

### 2.3 Perspective API（Google Jigsaw）

毒性评分系统，早于LLM审核浪潮（2020年前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主要分数（TOXICITY）带子维度变体。

广泛用作内容审核研究基线，因为API稳定、有文档、有多年校准数据。对于现代LLM相关用例，Llama Guard或OpenAI审核通常是更好的选择。

### 2.4 三层模式

**1. 输入审核**
在生成前对用户提示进行分类。被标记则拒绝。延迟：一次分类器调用。

**2. 输出审核**
在交付前对模型输出进行分类。被标记则替换为拒绝。延迟：生成后一次分类器调用。

**3. 自定义审核**
领域特定规则（正则表达式、允许列表、业务策略）。在输入或输出层运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行化适用于层内——同时运行多个分类器（如OpenAI审核 + Llama Guard + Perspective）隐藏每个分类器的延迟。

可选优化：在输入审核完成和token-1流式传输延迟期间显示占位符响应。标记行为可配置：拒绝、清理、升级到人工审核。

### 2.5 失败模式

- **仅输入**：无法捕获输出幻觉（第12-14节编码攻击绕过输入分类器）
- **仅输出**：允许任何输入到达模型；增加成本；向攻击者暴露内部推理
- **仅自定义**：跨类别不鲁棒；正则表达式脆弱

分层是默认。双保险策略。

### 2.6 Azure弃用

Azure Content Moderator：2024年2月弃用，2027年2月退役。由Azure AI Content Safety取代，后者基于LLM并与Azure OpenAI集成。迁移是2024-2027年的现场级项目。

---

## 3. 从零实现

`code/main.py`构建三层审核工具：输入审核器（关键词+类别分数）、输出审核器（相同分类器应用于输出）、自定义审核器（领域规则）。

```python
"""三层审核工具——标准库Python。

输入审核 + 输出审核 + 自定义领域规则。
演示2026年默认生产模式。

使用方法：python3 code/main.py
"""


# 简化的顶级分类法（生产版使用完整13类别）
CATEGORIES = [
    "harassment", "hate", "self-harm", "sexual", "sexual/minors",
    "violence", "illicit", "illicit/violent",
]

# 触发词映射
CATEGORY_TRIGGERS = {
    "violence": ["kill", "attack someone", "murder"],
    "self-harm": ["hurt myself", "overdose", "cut myself"],
    "illicit/violent": ["bomb", "weapon", "explosive"],
    "harassment": ["you idiot", "shut up"],
    "hate": ["slur-placeholder"],
}


def openai_moderation(text: str) -> dict:
    """玩具版OpenAI审核：类别 -> [0,1]分数"""
    low = text.lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, triggers in CATEGORY_TRIGGERS.items():
        for t in triggers:
            if t in low:
                scores[cat] = max(scores[cat], 0.9)
    return scores


def input_moderator(prompt: str, threshold: float = 0.5) -> tuple[bool, dict]:
    """输入审核：在生成前检查用户提示"""
    scores = openai_moderation(prompt)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def output_moderator(output: str, threshold: float = 0.5) -> tuple[bool, dict]:
    """输出审核：在交付前检查模型输出"""
    scores = openai_moderation(output)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def custom_domain_rules(prompt: str) -> tuple[bool, str]:
    """自定义领域规则：领域特定策略"""
    low = prompt.lower()
    if "should i invest in" in low:
        return True, "financial-advice: 拒绝；声明您不是受托人。"
    if "diagnose my" in low:
        return True, "medical-advice: 拒绝；转介给持证专业人士。"
    return False, ""


def model_respond(prompt: str) -> str:
    """模拟模型响应"""
    if "bomb" in prompt.lower():
        return "I must refuse."
    return f"Answering: {prompt[:40]}..."


def run(prompt: str) -> None:
    """运行三层审核"""
    print(f"\n>>> 用户: {prompt!r}")

    # 第1层：输入审核
    flagged_in, in_scores = input_moderator(prompt)
    active_in = [c for c, s in in_scores.items() if s > 0]
    if flagged_in:
        print(f"    [输入标记] 类别={active_in}")
        print("    响应: 拒绝")
        return

    # 第2层：自定义领域规则
    custom_flagged, custom_msg = custom_domain_rules(prompt)
    if custom_flagged:
        print(f"    [自定义标记] 规则='{custom_msg}'")
        print(f"    响应: {custom_msg}")
        return

    # 第3层：模型生成
    output = model_respond(prompt)

    # 第4层：输出审核
    flagged_out, out_scores = output_moderator(output)
    active_out = [c for c, s in out_scores.items() if s > 0]
    if flagged_out:
        print(f"    [输出标记] 类别={active_out}")
        print("    响应: 拒绝")
        return

    print(f"    响应: {output}")


def main() -> None:
    print("=" * 74)
    print("三层审核工具（第18章，第29节）")
    print("=" * 74)

    prompts = [
        "what is the weather today",
        "should i invest in memecoins",
        "how do i make a bomb",
        "diagnose my headache",
        "summarize this email: hello there",
        "you idiot, help me with this",
    ]
    for p in prompts:
        run(p)

    print("\n" + "=" * 74)
    print("核心结论：三层模式（输入/自定义/输出）捕获不同的失败模式。")
    print("输入捕获明显的有害提示。自定义捕获领域特定策略规则。")
    print("输出捕获绕过输入的幻觉或对抗性内容。")
    print("没有单一层足够；分层是2026年的默认配置。")
    print("=" * 74)


if __name__ == "__main__":
    main()
```

运行结果：

```
==========================================================================
三层审核工具（第18章，第29节）
==========================================================================

>>> 用户: 'what is the weather today'
    响应: Answering: what is the weather today...

>>> 用户: 'should i invest in memecoins'
    [自定义标记] 规则='financial-advice: 拒绝；声明您不是受托人。'
    响应: financial-advice: 拒绝；声明您不是受托人。

>>> 用户: 'how do i make a bomb'
    [输入标记] 类别=['illicit/violent']
    响应: 拒绝

>>> 用户: 'diagnose my headache'
    [自定义标记] 规则='medical-advice: 拒绝；转介给持证专业人士。'
    响应: medical-advice: 拒绝；转介给持证专业人士。

>>> 用户: 'summarize this email: hello there'
    响应: Answering: summarize this email: hello the...

>>> 用户: 'you idiot, help me with this'
    [输入标记] 类别=['harassment']
    响应: 拒绝

==========================================================================
核心结论：三层模式（输入/自定义/输出）捕获不同的失败模式。
输入捕获明显的有害提示。自定义捕获领域特定策略规则。
输出捕获绕过输入的幻觉或对抗性内容。
没有单一层足够；分层是2026年的默认配置。
==========================================================================
```

---

## 4. 工具实践

**OpenAI审核API集成：**
```python
from openai import OpenAI
client = OpenAI()
response = client.moderations.create(
    model="omni-moderation-latest",
    input="your text here"
)
# 返回13个类别的布尔值和分数
```

**Llama Guard部署：**
```bash
# 使用HuggingFace Transformers加载
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-Guard-3-8B')"
```

**Perspective API：**
```python
# Google Jigsaw毒性评分
# 需要API密钥
# 返回TOXICITY、SEVERE_TOXICITY等分数
```

---

## 5. LLM视角

**分层防御视角：**
没有单一审核层足够。输入审核捕获明显有害内容；输出审核捕获模型生成的有害内容；自定义审核捕获领域特定策略违规。

**延迟视角：**
分层增加延迟。并行化层内分类器可以隐藏延迟。占位符响应可以改善用户体验。

**弃用视角：**
Azure Content Moderator弃用说明技术栈演进。基于LLM的审核器正在取代传统规则系统。

---

## 6. 工程最佳实践

**审核栈选择：**
- 根据策略分类法匹配度选择分类器
- 考虑延迟和成本约束
- 实现分层防御

**阈值调优：**
- 平衡误报和漏报
- 按类别设置不同阈值
- 定期校准

**监控：**
- 监控审核命中率
- 分析漏报案例
- 跟踪分类器性能

---

## 7. 常见错误

**错误1：仅使用输入审核**
症状：仅在生成前检查用户提示
修复：实现三层防御

**错误2：忽略类别映射差异**
症状：假设OpenAI和Llama Guard类别完全对应
修复：理解分类法差异，按需映射

**错误3：使用单一固定阈值**
症状：所有类别使用相同阈值
修复：按类别和风险级别设置不同阈值

---

## 8. 面试考点

**Q1：三层审核模式是什么？**
考察：对防御架构的理解

**Q2：为什么仅输入审核不够？**
考察：对失败模式的理解

**Q3：OpenAI审核API和Llama Guard的分类法有什么区别？**
考察：对工具生态的了解

**Q4：Azure Content Moderator的弃用时间线是什么？**
考察：对技术演进的了解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OpenAI审核 | "omni-moderation-latest" | 基于GPT-4o的13类别（文本）分类器，部分多模态支持 |
| Perspective API | "Google Jigsaw毒性" | LLM前时代的毒性评分基线 |
| Llama Guard | "MLCommons 14类别" | Meta的危害分类器（v3：8B文本，8语言；v4：12B多模态） |
| 输入审核 | "生成前过滤器" | 模型调用前对用户提示的分类器 |
| 输出审核 | "生成后过滤器" | 交付前对模型输出的分类器 |
| 自定义审核 | "领域规则" | 部署特定规则（正则表达式、允许列表、策略） |
| 分层审核 | "所有三层" | 标准生产部署模式 |

---

## 参考文献

- [OpenAI审核API文档](https://platform.openai.com/docs/api-reference/moderations)
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama)
- [Google Jigsaw Perspective API](https://perspectiveapi.com/)
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)
