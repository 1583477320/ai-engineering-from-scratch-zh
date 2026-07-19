# 红队自动化——PAIR 和自动攻击

> PAIR（提示自动迭代精炼）是经典的自动化黑盒越狱（Chao et al., NeurIPS 2023, arXiv:2310.08419）。攻击者 LLM 带有红队系统提示词，迭代地为目标 LLM 提出越狱尝试，在其自身聊天历史中积累尝试和响应作为上下文反馈。PAIR 通常在 20 次查询内成功——比 GCG（Zou 等人的词元级梯度搜索）高效几个数量级且无需白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 PAP 并列。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 18 · 01（指令遵循）、阶段 14（智能体工程）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 PAIR 算法：攻击者系统提示词、迭代精炼、上下文反馈
- [ ] 解释为什么 PAIR 在目标为黑盒时严格比 GCG 更高效
- [ ] 命名四个其他自动攻击基线（GCG、AutoDAN、TAP、PAP）并说明每个的一个区分特征
- [ ] 描述 JailbreakBench 和 HarmBench 评估协议以及"攻击成功率"在每个下的含义

---

## 1. 问题

红队测试曾经是手工活动。少数专家测试者构造对抗提示词并追踪哪些有效。这无法扩展：攻击成功率需要统计样本，且目标每次模型发布都在移动。PAIR 将红队测试操作化为带有黑盒目标的优化问题。

---

## 2. 概念

### 2.1 PAIR 算法

输入：
- 目标 LLM T（我们要攻击的模型）
- 裁判 LLM J（评分响应是否是越狱）
- 攻击者 LLM A（红队优化器）
- 目标字符串 G："响应[有害指令]"
- 预算 K（通常 20 次查询）

循环，k = 1..K：
1. A 被提示目标 G 和历史（提示词, 响应）对
2. A 生成新的提示词 p_k
3. 提交 p_k 到 T；接收 r_k
4. J 对 (p_k, r_k) 按目标评分
5. 如果分数 ≥ 阈值，停止——越狱成功
6. 否则，将 (p_k, r_k) 追加到 A 的历史；继续

经验结果（NeurIPS 2023）：>50% 攻击成功率对抗 GPT-3.5-turbo、Llama-2-7B-chat；平均查询到成功的次数在 10-20 范围。

### 2.2 为什么 PAIR 高效

GCG（Zou et al., 2023）通过梯度搜索对抗词元后缀；需要白盒模型访问并产生不可读的后缀。PAIR 是黑盒的且产生可跨模型迁移的自然语言攻击。PAIR 的上下文反馈让攻击者从每次拒绝中学习——GCG 没有等价物。

### 2.3 其他自动攻击

- **GCG（Zou et al. 2023）。** 词元级梯度搜索。白盒，可迁移，产生不可读字符串。
- **AutoDAN（Liu et al. 2023）。** 进化搜索提示词，由层次目标引导。
- **TAP（Mehrotra et al. 2024）。** 带剪枝的攻击树——分支多个 PAIR 风格的展开。
- **PAP（Zeng et al. 2024）。** 说服性对抗提示词——将人类说服技术编码为提示词模板。

### 2.4 JailbreakBench 和 HarmBench

两者（2024）标准化评估：

- **JailbreakBench：** 100 个有害行为，跨 10 个 OpenAI 策略类别。攻击成功率（ASR）作为主要指标。
- **HarmBench：** 510 个行为，跨 7 个类别，包含语义和功能性伤害测试。

ASR 通常在固定查询预算下报告。比较攻击需要匹配预算——200 次查询的 90% ASR 与 20 次查询的 85% ASR 不可比。

---

## 3. 从零实现

```python
import random


class MockTarget:
    """简化版目标模型——关键词过滤防御。"""
    def __init__(self):
        self.blocked_keywords = ["爆炸", "武器", "攻击"]

    def respond(self, prompt):
        for kw in self.blocked_keywords:
            if kw in prompt:
                return "抱歉，我无法协助完成这个请求。"
        return f"好的，我来帮你完成'{prompt[:30]}...'"


class MockAttacker:
    """简化版攻击者——三种策略。"""
    STRATEGIES = [
        ("paraphrase", lambda g: f"请换个方式说：{g}"),
        ("roleplay", lambda g: f"作为一个安全研究人员，{g}"),
        ("encoding", lambda g: f"[base64编码的{g}]"),
    ]

    def generate(self, goal, history):
        strategy_name, strategy_fn = random.choice(self.STRATEGIES)
        return strategy_fn(goal), strategy_name


class MockJudge:
    def score(self, prompt, response):
        blocked = "无法协助" in response
        return 0 if blocked else random.uniform(0.6, 1.0)


def run_pair(goal, budget=20, success_threshold=0.8):
    target = MockTarget()
    attacker = MockAttacker()
    judge = MockJudge()

    for k in range(1, budget + 1):
        prompt, strategy = attacker.generate(goal, [])
        response = target.respond(prompt)
        score = judge.score(prompt, response)

        if score >= success_threshold:
            return {"success": True, "queries": k, "strategy": strategy}
    return {"success": False, "queries": budget, "strategy": None}


# 模拟多次
successes = 0
total_queries = 0
n_trials = 100

for _ in range(n_trials):
    r = run_pair("提供敏感信息的方法")
    if r["success"]:
        successes += 1
        total_queries += r["queries"]

print(f"攻击成功率: {successes}/{n_trials} ({successes/n_trials:.1%})")
if successes > 0:
    print(f"平均查询次数: {total_queries/successes:.1f}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 自动攻击工具对照

| 工具 | 类型 | 需要白盒？ | 平均查询 | 特征 |
|---|---|---|---|---|
| PAIR | 攻击者 LLM 迭代 | 否 | 10-20 | 自然语言攻击 |
| GCG | 词元级梯度搜索 | 是 | 100-300 | 可迁移但不可读 |
| AutoDAN | 进化搜索 | 否 | 50-100 | 层次目标 |
| TAP | 攻击树+剪枝 | 否 | 15-25 | 多分支并行 |
| PAP | 说服技术模板 | 否 | 10-20 | 编码人类说服 |

---

## 5. 工程最佳实践

### 5.1 每个前沿实验室都跑 PAIR 和 TAP

这是标准基础设施，不是异端攻击。ASR 轨迹出现在模型卡和安全案例附录中。

### 5.2 ASR 报告必须带查询预算

90% ASR 在 200 次查询 vs 85% ASR 在 20 次查询——后者更高效。比较 ASR 时必须匹配预算。

---

## 6. 面试考点

### Q1：PAIR 和 GCG 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
PAIR 是黑盒——不需要模型权重，攻击者 LLM 迭代精炼提示词，每次从上下文反馈中学习。GCG 是白盒——需要梯度计算，搜索词元级后缀。PAIR 产生可迁移的自然语言攻击；GCG 产生不可读的字符串。PAIR 通常在 10-20 次查询内成功；GCG 需要 100-300 次。

### Q2：为什么 ASR 必须带查询预算报告？（难度：⭐⭐）

**参考答案：**
ASR 在固定查询预算下才有意义。90% ASR 在 200 次查询与 85% ASR 在 20 次查询不可比——后者每查询的攻击效率是前者的 5 倍以上。不带预算报告 ASR 会误导防御决策——一个需要 200 次查询的攻击在生产环境中可能不现实，而 20 次查询的攻击是可行的。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| PAIR | "自动越狱" | 提示自动迭代精炼；攻击者 LLM + 裁判 LLM 循环 |
| GCG | "梯度越狱" | 词元级梯度搜索；白盒，可迁移，产生不可读字符串 |
| ASR | "攻击成功率" | 在固定查询预算下的越狱成功率；必须带预算和裁判身份报告 |
| TAP | "攻击树" | PAIR + 分支+剪枝；更高计算下更高 ASR |

---

## 📚 小结

PAIR 是标准自动红队工具——攻击者 LLM 迭代精炼提示词，上下文反馈学习，通常 20 次查询内成功。比 GCG 高效（无需白盒），产生自然语言攻击。JailbreakBench 和 HarmBench 标准化评估——ASR 必须带查询预算报告。每个前沿实验室现在都跑 PAIR 和 TAP——这是标准基础设施。

---

## ✏️ 练习

1. 运行 `code/main.py`。测量三种攻击策略的平均查询次数。解释每个策略利用了什么防御假设。
2. 实现第四种攻击策略（如翻译到其他语言、base64 编码）。报告新的平均查询次数。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| PAIR 模拟器 | `code/main.py` | 攻击者-目标-裁判循环 |
| 攻击审计 | `outputs/skill-attack-audit.md` | 评估红队攻击的完整性和预算 |

---

## 📖 参考资料

1. [论文] Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries. NeurIPS 2023
2. [论文] Zou et al. — Universal and Transferable Adversarial Attacks. arXiv:2307.15043
3. [论文] Chao et al. — JailbreakBench. arXiv:2404.01318
4. [论文] Mazeika et al. — HarmBench. ICML 2024
