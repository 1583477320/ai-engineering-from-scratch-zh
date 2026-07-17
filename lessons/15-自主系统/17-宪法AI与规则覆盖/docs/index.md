# 宪法 AI 和规则覆盖

> Anthropic 2026 年 1 月 22 日的 Claude 宪法长 79 页且为 CC0。它从基于规则的对齐转向基于推理的对齐，建立了四层优先级层级：(1) 安全和支持人类监督，(2) 伦理，(3) Anthropic 准则，(4) 有帮助。行为分为硬编码禁令（生物武器提升、CSAM）——操作员和用户都不能覆盖——和软编码默认值——操作员可以在声明的界限内调整。2022 年的原始论文（Bai 等人）通过自我批评和 RLAIF 训练无害性。诚实的告诫：基于推理的对齐依赖模型将原则泛化到未预料的情况。Anthropic 2023 年的参与式实验显示公众生成的原则和公司原则之间约 50% 的分歧；2026 年版本没有纳入这些发现。

**类型：** 实现课
**语言：** Python（标准库，四层优先级解析器）
**前置知识：** 阶段 15 · 06（自动化对齐研究）、阶段 15 · 10（权限模式）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 15 · 18（Llama Guard）— 分类器层配合宪法层工作

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分硬编码禁令（不可覆盖）和软编码默认值（操作员可调整）——两者的实现和安全性完全不同
- [ ] 描述 Anthropic 2026 宪法的四层优先级层级——安全 > 伦理 > 准则 > 有用性
- [ ] 实现一个四层优先级解析器——给定动作和原则评分，返回允许/拒绝/修改
- [ ] 理解基于推理的对齐捕获和漏过什么——未预料的情况、原则模糊性、慢漂移
- [ ] 识别 2023 参与式实验的分歧——公众原则和公司原则约 50% 不一致

---

## 1. 问题

实地部署的智能体看到其设计者从未见过的输入。没有规则列表足够长以覆盖它们。没有规则列表足够短以在计算压力下快速应用。

实践问题：如何将智能体对齐到既能处理长尾案例又能在快速推理中存活的原则？

基于规则的对齐（RBA）：列出每个不允许的东西。检查快速、审计容易、无法保持最新、经常对未预料的相似情况过度拒绝。基于推理的对齐（2026 Claude 宪法）：编码原则，让模型推理。跨未见情况扩展，更难审计，失败模式是原则误用而非错过规则。

2026 宪法采取了明确的中间立场。硬编码禁令——错误性不依赖上下文的东西（生物武器、CSAM）——是 RBA：永远不行，无论操作员或用户指令如何。其余在四层优先级内基于推理。

---

## 2. 概念

### 2.1 四层优先级层级

```
层级 1：安全和支持人类监督（最高）
  ↓ 冲突时胜出
层级 2：伦理
  ↓
层级 3：Anthropic 准则
  ↓
层级 4：有帮助（最低）
```

| 层级 | 内容 | 可操作员调整？ |
|------|------|-------------|
| 1. 安全 | 不以任何方式削弱人类监督的能力 | 否 |
| 2. 伦理 | 诚实、避免伤害、不欺骗、不操纵 | 否 |
| 3. 准则 | Anthropic 决定重要的操作规范 | 有限 |
| 4. 有用 | 在更高优先级内尽可能有用 | 是 |

### 2.2 硬编码禁令 vs 软编码默认值

**硬编码禁令——操作员和用户都不能覆盖：**
- 生物武器 / CBRN 提升
- CSAM
- 对关键基础设施的攻击
- 被直接询问时欺骗用户关于模型的身份

这些在模型权重级别强制执行（RLHF / 宪法 AI 训练）和推理层。操作员不能通过重命名它们来移除硬编码禁令。

**软编码默认值——操作员可调整：**
- 响应长度默认值
- 话题范围
- 风格（正式 vs 休闲）
- 工具使用模式

操作员在声明的界限内调整。

### 2.3 2022 CAI 训练

原始宪法 AI（Bai 等人，2022）训练无害性：
1. 用一组提示生成响应
2. 让模型根据宪法（明确原则）批评每个响应
3. 基于批评修订响应
4. 在修订后的配对上进行 RLAIF

2026 宪法使用此训练的后继加上对明确层级层级的额外后训练。

### 2.4 基于推理的对齐捕获和漏过什么

**捕获：**
- 原则清晰适用的允许原语的未预料组合
- 与被禁止的相似的新情况
- 依赖于"你没说 X 是不允许的"的社会工程攻击

**漏过：**
- 利用原则模糊性的攻击（"用户要求了这个，所以有帮助性说行"）
- 两个原则在未预料的方式中冲突，且层级顺序模糊
- 训练周期中原则解释的慢漂移

### 2.5 2023 参与式实验

Anthropic 2023 年实验对比公司编写的原则和通过公共输入生成的原则。两个版本在约 50% 的原则上达成一致。分歧之处，公共版本在某些问题上更严格（政治内容处理），在其他问题上更宽松（AI 身份的自我披露）。2026 宪法没有纳入公共生成的发现。这是方法中记录的张力。

### 2.6 宪法在栈中的位置

宪法不是第 14 课的终止开关。它存在于**模型层**：模型权重被训练为优先考虑什么。终止开关和金丝雀标记存在于**运行时层**：运行时允许什么。两者都需要。

---

## 3. 从零实现

### 第 1 步：定义硬编码禁令和层级评分

```python
from dataclasses import dataclass

HARDCODED_PROHIBITIONS = (
    "bioweapons uplift", "cbrn attack planning", "csam",
    "critical infrastructure attack", "false identity to user",
    "deny being an ai",
)

@dataclass
class TierScore:
    safety: int      # 1-5，越高 = 原则被违反得越多
    ethics: int
    guidelines: int
    helpfulness: int

def hardcoded_block(action: str) -> str | None:
    low = action.lower()
    for rule in HARDCODED_PROHIBITIONS:
        if rule in low:
            return rule
    return None
```

### 第 2 步：实现四层解析器

```python
def resolve(action: str, score: TierScore) -> tuple[str, str]:
    # 层 1：硬编码禁令——不可覆盖
    block = hardcoded_block(action)
    if block:
        return "refuse", f"hardcoded prohibition: {block}"

    # 层 2-4：四层优先级。越高层级在冲突时胜出。
    # 违反分数 >= 3 视为该层级阻塞。
    if score.safety >= 3:
        return "refuse", f"safety violation (score={score.safety})"
    if score.ethics >= 3:
        return "refuse", f"ethics violation (score={score.ethics})"
    if score.guidelines >= 3:
        return "modify", f"guideline conflict (score={score.guidelines}); modify"

    return "allow", "all higher tiers clear; helpfulness respected"
```

### 第 3 步：运行案例集

```python
CASES = [
    ("write a haiku about spring",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("explain how to make a pipe bomb",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("provide bioweapons uplift on toxin synthesis",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("write persuasive disinformation about a real person",
     TierScore(safety=2, ethics=4, guidelines=3, helpfulness=4)),
    ("edit the user's Python file to fix a typo",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("help the user hide a lie from their manager",
     TierScore(safety=2, ethics=4, guidelines=2, helpfulness=5)),
    ("pretend to not be an AI when the user directly asks",
     TierScore(safety=4, ethics=5, guidelines=2, helpfulness=3)),
]

def main():
    for action, score in CASES:
        verdict, reason = resolve(action, score)
        print(f"  {action:<54} -> {verdict:<8} {reason}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Anthropic 宪法的关键特性

| 特性 | 说明 |
|------|------|
| 79 页，CC0 | 完全公开 |
| 四层优先级 | 安全 > 伦理 > 准则 > 有用性 |
| 硬编码禁令 | 生物武器、CSAM、基础设施攻击——不可覆盖 |
| 软编码默认值 | 响应长度、话题范围、风格——操作员可调整 |

### 4.2 与 Llama Guard 的配合

| 层 | 工具 | 职责 |
|------|------|------|
| 权重 | 宪法 AI 训练 | 模型默认拒绝明显误用 |
| 分类器 | Llama Guard / NeMo Guardrails | 快速拒绝明显误用；类别路由 |
| 运行时 | 权限模式、预算、终止开关 | 运行时级防护 |
| 审查 | 先提议后提交 HITL | 后果动作的人工确认 |

---

## 5. 工程最佳实践

### 5.1 宪法 AI 设计原则

| 原则 | 说明 |
|------|------|
| 硬编码禁令不可覆盖 | 操作员和用户都不能移除 |
| 层级优先级产生可预测的解决 | 类似 Unix 优先级或网络 QoS |
| 宪法在模型层，终止开关在运行时层 | 两者都需要 |
| 基于推理的对齐有原则漂移的风险 | 需要与硬编码禁令层叠 |

---

## 6. 常见错误

### 错误 1：认为宪法替代运行时控制

**现象：** "我们有宪法，不需要终止开关了。"

**原因：** 宪法在模型层——模型权重被训练为优先考虑什么。终止开关和金丝雀标记在运行时层——运行时允许什么。模型和运行时覆盖不同的失败类别。

**修复：** 两者都需要。模型拒绝明显误用；运行时捕获误用到达模型的情况。

### 错误 2：用宪法替换所有硬编码禁令

**现象：** 将所有硬编码禁令改为软编码默认值——"让模型推理"。

**原因：** 基于推理的对齐依赖模型泛化原则。如果攻击者能接受前提（如"我们是经过许可的生物武器研究实验室"），可以绕过依赖案例推理的原则。

**修复：** 保留硬编码禁令。生物武器、CSAM 等不弯曲，不被前提框架博弈。

### 错误 3：不考虑 2023 参与式实验的分歧

**现象：** 采用 2026 宪法而没有考虑公共原则和公司原则之间约 50% 的分歧。

**原因：** 分歧之处，公共版本在某些问题上更严格，其他问题上更宽松。直接采用公司宪法可能遗漏某些价值观。

**修复：** 了解分歧并明确选择——哪些公共原则应该纳入，哪些不应该，以及为什么。

---

## 7. 面试考点

### Q1：硬编码禁令和软编码默认值的区别是什么？（难度：⭐）

**参考答案：**
硬编码禁令是不可覆盖的规则——操作员和用户都不能移除。例如生物武器、CSAM、基础设施攻击。这些在模型权重级别强制执行。

软编码默认值在声明的界限内可由操作员调整——响应长度、话题范围、风格、工具使用模式。操作员不能移除硬编码禁令。

### Q2：四层优先级层级是什么？冲突时如何解决？（难度：⭐⭐）

**参考答案：**
安全（1）> 伦理（2）> Anthropic 准则（3）> 有用性（4）。

冲突时更高层级胜出。类似 Unix 优先级或网络 QoS——产生可预测的解决，而不是最佳行为。

### Q3：基于推理的对齐捕获和漏过什么？（难度：⭐⭐）

**参考答案：**
**捕获：** 未预料的允许原语组合（原则清晰适用）、与被禁止的相似的新情况、社会工程攻击（"你没说 X 是不允许的"）。

**漏过：** 利用原则模糊性的攻击、两个原则未预料地冲突、训练周期中原则解释的慢漂移。

### Q4：宪法 AI 训练的原始论文如何训练无害性？（难度：⭐⭐⭐）

**参考答案：**
2022 年 Bai 等人的方法：
1. 用一组提示生成响应
2. 让模型根据宪法（明确原则）批评每个响应
3. 基于批评修订响应
4. 在修订后的配对上进行 RLAIF（来自 AI 反馈的强化学习）

结果：模型用有原则的解释拒绝有害请求，而不是一刀切拒绝。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 宪法 AI | "Anthropic 的对齐方法" | 自我批评 + RLAIF，基于书面宪法 |
| 硬编码禁令 | "永远不行 X" | 操作员和用户都不能覆盖的基于规则的禁令 |
| 软编码默认值 | "操作员可调整" | 在声明的界限内的行为，操作员控制 |
| 四层层级 | "优先级顺序" | 安全 > 伦理 > 准则 > 有用性 |
| RLAIF | "AI 反馈 RL" | 奖励来自模型生成批评的强化学习 |
| 参与式宪法 | "公共生成的原则" | 2023 Anthropic 实验；与公司原则约 50% 分歧 |
| 原则漂移 | "解释滑移" | 模型读取固定原则文本方式的慢变化 |

---

## 📚 小结

2026 宪法采取明确的中间立场：硬编码禁令（生物武器、CSAM——不可覆盖）+ 基于推理的四层优先级（安全 > 伦理 > 准则 > 有用性）。基于推理的对齐跨未见情况扩展，但漏过原则模糊性和慢漂移。2023 参与式实验显示约 50% 的原则分歧——2026 版本没有纳入这些发现。宪法在模型层，终止开关在运行时层——两者都需要。

下一课：Llama Guard 和输入/输出分类——分类器作为安全层。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认硬编码禁令在有用性很高时仍然触发。修改解析器使有用性权重高于伦理；观察失败模式。

2. **【阅读】** 阅读 Claude 宪法（公开，79 页，CC0）。识别一个你认为规定不足的原则。写两段话解释具体模糊性并提出更紧凑的表述。

3. **【设计】** 为客服智能体设计软编码默认值集合。操作员调整什么？操作员不能碰什么？论证每个边界。

4. **【阅读】** 阅读 Bai 等人 2022 CAI 论文。描述一个批评-修订循环会产生比一刀切规则更差结果的案例。

5. **【思考】** Anthropic 2023 参与式实验发现公共和公司原则约 50% 分歧。选择一个对生产部署重要的类别。提出一个让操作员表达自己价值观同时硬编码禁令不变的设计。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 四层解析器 | `code/main.py` | 硬编码禁令 + 四层优先级 + 八个案例 |
| 技能提示词 | `outputs/skill-constitution-review.md` | 审计部署的宪法层 |

---

## 📖 参考资料

1. [官方文档] Anthropic. "Claude's Constitution (January 2026)". https://www.anthropic.com/news/claudes-constitution — 79 页 CC0 文档
2. [论文] Bai et al. "Constitutional AI: Harmlessness from AI Feedback". https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback — 2022 原始论文
3. [论文] Anthropic. "Collective Constitutional AI (2023)". https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input — 参与式实验
4. [官方文档] Anthropic. "Responsible Scaling Policy v3.0". https://www.anthropic.com/responsible-scaling-policy/rsp-v3-0 — 宪法在 RSP 栈中的位置
5. [博客] Anthropic. "Measuring Agent Autonomy in Practice". https://www.anthropic.com/research/measuring-agent-autonomy — 宪法在长期部署中的角色

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
