# 多智能体评估与基准测试

> 单个智能体的准确率是 90%，五个智能体协作后准确率是多少？不是 90%。多智能体系统的评估必须回答一个单智能体评估无法回答的问题：协作是否比单干更好？

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 16 · 22（多智能体协商与拍卖机制）、阶段 16 · 23（多智能体系统生产部署）、阶段 10 · 10（LLM 评估）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 10（LLM 评估框架）— 单智能体评估是多智能体评估的基础，本课在此之上扩展协作维度

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分单智能体评估和多智能体评估的核心差异——协作质量、通信开销、故障恢复
- [ ] 设计端到端评估指标——任务完成率、平均词元消耗、端到端延迟、协作效率
- [ ] 构建多智能体评估流水线——自动化测试多个智能体协作的正确性和效率
- [ ] 诊断"协作不如单干"问题——定位协作开销超过收益的场景

---

## 1. 问题

你的多智能体系统上线一个月了。老板问："效果怎么样？"你打开 LangSmith，看到每个智能体的单独调用成功率都在 85% 以上。但用户投诉说"生成的报告经常前后矛盾"。

问题出在哪里？单个智能体的评估指标全部达标，但**协作质量**没有人评估。数据分析师说"销售额增长 20%"，报告撰写者引用了另一个数字"增长 15%"——两个智能体用了不同的数据源，但没有人在意这个差异。单智能体评估看不到这个问题。

**多智能体评估必须回答四个单智能体评估无法回答的问题：**

1. **协作质量：** 多个智能体的输出是否一致、是否互补？
2. **通信效率：** 智能体之间传递了多少信息？有多少是冗余的？
3. **故障恢复：** 一个智能体失败后，系统是否优雅降级？
4. **协作增益：** 多智能体方案是否真的比单智能体方案更好？

---

## 2. 概念

### 2.1 评估维度

多智能体系统的评估从三个层面展开：

```
┌─────────────────────────────────────────────┐
│          系统级评估（端到端）                   │
│  任务完成率 · 总延迟 · 总成本 · 用户满意度     │
├─────────────────────────────────────────────┤
│          协作级评估（智能体间）                 │
│  一致性 · 互补性 · 通信开销 · 冲突解决          │
├─────────────────────────────────────────────┤
│          单智能体评估（每个智能体）              │
│  准确率 · 延迟 · 成本 · 错误类型               │
└─────────────────────────────────────────────┘
```

**系统级评估**衡量最终结果：用户拿到的东西好不好、快不快、贵不贵。

**协作级评估**衡量智能体之间的交互：信息传递是否高效、输出是否一致、冲突是否妥善解决。

**单智能体评估**衡量每个智能体的个体能力：准确率、延迟、成本。

三层评估缺一不可。只看系统级——不知道问题出在哪个环节。只看单智能体——看不到协作问题。只看协作级——不知道是个体能力不足还是协作机制有问题。

### 2.2 核心指标

| 指标 | 计算方式 | 含义 |
|---|---|---|
| **任务完成率** | 成功任务数 / 总任务数 | 最终输出是否满足用户需求 |
| **端到端延迟** | 从请求到最终响应的总时间 | 用户体验 |
| **总词元消耗** | 所有智能体的词元之和 | 成本 |
| **协作效率** | 最终输出词元 / 所有智能体总词元 | 信息传递的效率——越高说明冗余越少 |
| **一致性分数** | 多个智能体输出的语义一致性 | 输出是否自相矛盾 |
| **故障恢复率** | 故障后成功降级的比例 | 系统的鲁棒性 |
| **协作增益** | 多智能体得分 / 单智能体得分 | 多智能体是否真的更好 |

### 2.3 协作增益：多智能体是否值得？

这是最重要的评估问题。多智能体系统的开销是单智能体的 N 倍——更多的 API 调用、更多的词元、更高的延迟。如果最终质量没有显著提升，多智能体就是过度工程。

```
协作增益 = 多智能体方案的质量 / 单智能体方案的质量

协作增益 > 1.0：多智能体更好
协作增益 = 1.0：效果相同（但成本更高——多智能体失败）
协作增益 < 1.0：多智能体更差（协作开销超过了收益）
```

**什么场景下多智能体有正增益？**

- **任务需要不同专业能力。** 数据分析 + 文本写作 + 图表生成——每个子任务需要不同的 LLM 提示词和工具
- **任务需要多视角验证。** 事实核查——一个智能体生成，另一个智能体验证，减少幻觉
- **任务规模超出单个上下文窗口。** 100 页文档的摘要——单个智能体放不下，需要分块处理后合并

**什么场景下多智能体是负增益？**

- **任务本身简单。** "翻译一句话"——一个智能体就够了，三个智能体只会增加延迟
- **智能体之间需要频繁协调。** 协调开销 > 并行收益
- **输出不需要一致性。** 多个智能体各自独立工作，不需要互相参考

### 2.4 评估方法

**离线评估（开发阶段）：**

用标注数据集评估多智能体系统的输出质量。需要构建"多智能体评估数据集"——不仅标注最终答案，还标注中间步骤的正确性（每个智能体的输出是否合理）。

**在线评估（生产阶段）：**

用真实用户流量评估。关注：任务完成率、用户满意度（点赞/点踩）、成本趋势、延迟分布。A/B 测试：多智能体方案 vs 单智能体方案，对比核心指标。

**对抗评估（鲁棒性测试）：**

注入故障（模拟智能体超时、返回错误结果），观察系统的降级行为。是否优雅降级？是否输出了部分结果？是否给出了有意义的错误信息？

---

## 3. 从零实现

### 第 1 步：定义评估指标

```python
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """多智能体系统的评估指标集合。"""

    # 系统级指标
    task_success: bool = False          # 任务是否成功完成
    total_latency: float = 0.0         # 端到端延迟（秒）
    total_tokens: int = 0              # 总词元消耗
    total_cost: float = 0.0            # 总费用（美元）

    # 协作级指标
    agent_outputs: list = field(default_factory=list)  # 每个智能体的输出
    communication_rounds: int = 0      # 智能体间通信轮次
    conflicts_detected: int = 0        # 检测到的输出冲突数

    # 单智能体指标
    per_agent_tokens: dict = field(default_factory=dict)  # 每个智能体的词元消耗
    per_agent_latency: dict = field(default_factory=dict)  # 每个智能体的延迟

    def compute_efficiency(self) -> float:
        """协作效率 = 最终输出词元 / 所有智能体总词元。"""
        if self.total_tokens == 0:
            return 0.0
        final_output_tokens = len(self.agent_outputs[-1]) if self.agent_outputs else 0
        return final_output_tokens / self.total_tokens

    def compute_consistency(self) -> float:
        """一致性分数：多个智能体输出的语义一致性（简化版）。"""
        if len(self.agent_outputs) < 2:
            return 1.0  # 只有一个智能体，天然一致

        # 简化：检查关键词重叠率
        all_keywords = []
        for output in self.agent_outputs:
            keywords = set(output.split())
            all_keywords.append(keywords)

        overlaps = 0
        pairs = 0
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                intersection = all_keywords[i] & all_keywords[j]
                union = all_keywords[i] | all_keywords[j]
                if union:
                    overlaps += len(intersection) / len(union)
                pairs += 1

        return overlaps / pairs if pairs > 0 else 1.0

    def summary(self) -> dict:
        """返回评估摘要。"""
        return {
            "task_success": self.task_success,
            "total_latency": f"{self.total_latency:.2f}s",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "efficiency": f"{self.compute_efficiency():.2%}",
            "consistency": f"{self.compute_consistency():.2%}",
            "communication_rounds": self.communication_rounds,
            "conflicts_detected": self.conflicts_detected,
        }
```

### 第 2 步：构建评估流水线

```python
import time


class MultiAgentEvaluator:
    """多智能体系统的自动化评估器。"""

    def __init__(self, single_agent_fn, multi_agent_fn):
        """
        Args:
            single_agent_fn: 单智能体方案，输入任务描述，输出结果
            multi_agent_fn: 多智能体方案，输入任务描述，输出结果和指标
        """
        self.single_agent_fn = single_agent_fn
        self.multi_agent_fn = multi_agent_fn

    def evaluate(self, tasks: list[str]) -> dict:
        """在任务集上评估两个方案。"""
        single_results = []
        multi_results = []

        for task in tasks:
            # 单智能体方案
            start = time.time()
            single_output = self.single_agent_fn(task)
            single_latency = time.time() - start
            single_results.append({
                "task": task,
                "output": single_output,
                "latency": single_latency,
            })

            # 多智能体方案
            start = time.time()
            multi_output, metrics = self.multi_agent_fn(task)
            multi_latency = time.time() - start
            multi_results.append({
                "task": task,
                "output": multi_output,
                "latency": multi_latency,
                "metrics": metrics,
            })

        return self._compare(single_results, multi_results)

    def _compare(self, single_results, multi_results) -> dict:
        """对比两个方案的核心指标。"""
        avg_single_latency = sum(r["latency"] for r in single_results) / len(single_results)
        avg_multi_latency = sum(r["latency"] for r in multi_results) / len(multi_results)
        avg_multi_tokens = sum(
            r["metrics"].total_tokens for r in multi_results
        ) / len(multi_results)

        # 协作增益（简化：用延迟和词元的综合评分）
        # 实际应用中应该用人工评估或 LLM-as-Judge
        speed_ratio = avg_single_latency / avg_multi_latency if avg_multi_latency > 0 else 1.0

        return {
            "avg_single_latency": f"{avg_single_latency:.2f}s",
            "avg_multi_latency": f"{avg_multi_latency:.2f}s",
            "avg_multi_tokens": f"{avg_multi_tokens:.0f}",
            "speed_ratio": f"{speed_ratio:.2f}x",
            "recommendation": "使用多智能体" if speed_ratio > 1.0 else "使用单智能体",
        }
```

### 第 3 步：运行评估

```python
# 模拟单智能体和多智能体方案
def single_agent(task: str) -> str:
    """单智能体：一个模型处理所有工作。"""
    time.sleep(0.5)  # 模拟 API 调用
    return f"单智能体完成: {task}"

def multi_agent(task: str) -> tuple[str, EvaluationMetrics]:
    """多智能体：多个模型协作。"""
    metrics = EvaluationMetrics()
    start = time.time()

    # 模拟数据分析师
    time.sleep(0.3)
    analysis = f"数据分析: {task}"
    metrics.per_agent_tokens["分析师"] = 800
    metrics.agent_outputs.append(analysis)

    # 模拟报告撰写者
    time.sleep(0.4)
    report = f"报告撰写: {analysis}"
    metrics.per_agent_tokens["撰写者"] = 1200
    metrics.agent_outputs.append(report)

    metrics.total_tokens = sum(metrics.per_agent_tokens.values())
    metrics.total_latency = time.time() - start
    metrics.task_success = True
    metrics.communication_rounds = 1

    return report, metrics


# 运行评估
evaluator = MultiAgentEvaluator(single_agent, multi_agent)

tasks = [
    "分析上季度销售数据",
    "生成市场趋势报告",
    "对比竞品优劣势",
]

results = evaluator.evaluate(tasks)

print("=== 评估结果 ===")
for key, value in results.items():
    print(f"  {key}: {value}")
```

```text
=== 评估结果 ===
  avg_single_latency: 0.50s
  avg_multi_latency: 0.70s
  avg_multi_tokens: 2000
  speed_ratio: 0.71x
  recommendation: 使用单智能体
```

这个结果说明：对于简单的任务，多智能体方案更慢（0.70s vs 0.50s）且消耗更多词元。**协作增益为负——单智能体就够了。** 只有当任务足够复杂、需要不同专业能力时，多智能体才值得。

---

## 4. 工业工具

### 4.1 LangSmith 评估

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 定义评估函数
def correctness(run, example):
    """评估输出的正确性。"""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    # 简化的匹配评估
    score = 1.0 if reference in prediction else 0.0
    return {"key": "correctness", "score": score}


def consistency(run, example):
    """评估多智能体输出的一致性。"""
    agent_outputs = run.outputs.get("agent_outputs", [])
    if len(agent_outputs) < 2:
        return {"key": "consistency", "score": 1.0}
    # 检查关键词重叠
    all_words = [set(o.split()) for o in agent_outputs]
    overlaps = []
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            if all_words[i] | all_words[j]:
                overlaps.append(
                    len(all_words[i] & all_words[j]) / len(all_words[i] | all_words[j])
                )
    return {"key": "consistency", "score": sum(overlaps) / len(overlaps) if overlaps else 1.0}


# 运行评估
results = evaluate(
    multi_agent_app,       # 要评估的应用
    data="multi-agent-test-dataset",  # 评估数据集
    evaluators=[correctness, consistency],
)
```

### 4.2 LLM-as-Judge 评估

```python
from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(model="gpt-4o")

def llm_judge(task: str, single_output: str, multi_output: str) -> dict:
    """用 LLM 作为裁判，对比两个方案的输出质量。"""
    prompt = f"""你是评估专家。请对比以下两个方案的输出质量。

任务：{task}

方案 A（单智能体）：
{single_output}

方案 B（多智能体）：
{multi_output}

请从以下维度评分（1-5 分）：
1. 准确性：信息是否正确
2. 完整性：是否覆盖了所有要点
3. 一致性：内部是否自相矛盾
4. 可读性：是否易于理解

返回 JSON 格式：
{{"single": {{"accuracy": N, "completeness": N, "consistency": N, "readability": N}},
  "multi": {{"accuracy": N, "completeness": N, "consistency": N, "readability": N}},
  "winner": "single" 或 "multi",
  "reason": "一句话理由"}}"""

    response = judge_llm.invoke(prompt)
    return response.content
```

### 4.3 工具对比

| 工具 | 评估方式 | 适用场景 | 成本 |
|---|---|---|---|
| LangSmith | 追踪 + 自定义评估函数 | 开发阶段的系统评估 | 低 |
| LLM-as-Judge | 用另一个 LLM 评估输出质量 | 质量评估、A/B 对比 | 中 |
| 人工评估 | 人类标注者打分 | 最终验收、高风险场景 | 高 |
| 自定义流水线 | 自动化测试 | CI/CD 集成、回归测试 | 低 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

OpenAI 的 Evals 框架支持自定义评估——你可以定义"什么算成功"，然后在大量测试用例上运行。Anthropic 的 Claude 评估指南推荐用 LLM-as-Judge 做自动化评估。Google 的 Vertex AI Evaluation 提供了端到端的评估流水线。

### 5.2 LLM 时代什么变了？

传统多智能体评估关注"消息传递的正确性"——智能体 A 发送的消息是否被智能体 B 正确接收和处理。LLM 多智能体评估增加了新维度：**语义一致性**——多个 LLM 智能体的输出在语义上是否一致？这比传统评估复杂得多，因为 LLM 的输出是自然语言，不是结构化数据。

### 5.3 什么没变？

评估的基本原则没变：你需要定义"成功"，然后测量"离成功有多远"。多智能体评估只是把"成功"的定义从"单个输出正确"扩展到了"多个输出一致且互补"。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的 GPTs 功能时，系统可能在内部运行多智能体协作——主模型调用代码解释器和联网搜索。如果代码解释器返回了错误结果，主模型是否会发现并纠正？这就是多智能体评估要回答的问题。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 阶段 | 推荐方案 | 备注 |
|---|---|---|
| 开发阶段 | 自定义评估流水线 + LLM-as-Judge | 快速迭代 |
| 测试阶段 | LangSmith 评估 + 人工抽样 | 系统评估 |
| 生产阶段 | 在线指标追踪 + A/B 测试 | 持续监控 |
| 回归测试 | CI/CD 集成的自动化测试 | 每次部署前运行 |

### 6.2 中文场景特别建议

- **中文多智能体评估需要中文评估数据集。** 英文的评估框架和数据集不能直接套用——中文的语言特性（无空格、歧义多、同音词）需要专门的评估维度
- **中文一致性评估更难。** 英文的关键词匹配在中文上不适用——"机器学习"和"机器 学习"在中文中是同一个概念但字面不同。需要用语义相似度（如 BERTScore）替代字面匹配
- **中文 LLM-as-Judge 需要中文提示词。** 用英文提示词让 GPT-4o 评估中文输出，质量会下降。确保评估提示词也是中文

### 6.3 踩坑经验

- **只评估最终输出，忽略中间步骤。** 最终报告看起来不错，但中间的数据分析步骤是错的——只是报告撰写者"修正"了数字。这种"偶然正确"在评估中必须被捕获
- **评估数据集太小。** 10 个测试用例的评估结果不可靠——可能恰好避开了系统的弱点。至少 50 个测试用例，覆盖不同难度和类型
- **忽略评估本身的成本。** LLM-as-Judge 的每次评估都是一次 API 调用。100 个测试用例 × 2 个方案 × 1 个裁判 = 200 次 API 调用。预算要包含评估成本
- **A/B 测试时间不够。** 多智能体方案可能在某些任务上更好，某些任务上更差。需要足够长的测试时间和足够多的样本来得出统计显著的结论

---

## 7. 常见错误

### 错误 1：用单智能体指标评估多智能体系统

**现象：** 每个智能体的准确率都是 90%，但最终输出的准确率只有 60%。

**原因：** 单智能体指标只衡量个体能力，不衡量协作质量。两个智能体各自正确但互相矛盾——合并后的输出可能是错的。

**修复：**

```python
# ❌ 只看单智能体指标
for agent in agents:
    accuracy = evaluate_single(agent)
    print(f"{agent.name}: {accuracy}")

# ✓ 同时评估协作质量
system_accuracy = evaluate_end_to_end(system)
consistency = evaluate_consistency(agent_outputs)
print(f"系统准确率: {system_accuracy}")
print(f"一致性: {consistency}")
# 如果系统准确率 < 单智能体准确率，说明协作有问题
```

### 错误 2：评估数据集不代表真实分布

**现象：** 在评估数据集上多智能体方案得分为 95%，上线后用户满意度只有 70%。

**原因：** 评估数据集是开发者自己构造的——覆盖了"理想情况"但没有覆盖"边界情况"。真实用户会输入各种奇怪的请求。

**修复：**

```python
# ❌ 只用开发者构造的数据集
test_cases = ["分析销售数据", "生成报告", "对比竞品"]

# ✓ 混合真实用户数据
test_cases = developer_cases + sampled_production_cases + adversarial_cases
# 开发者案例（理想情况）+ 真实用户案例（实际分布）+ 对抗案例（边界情况）
```

### 错误 3：评估结果不可复现

**现象：** 同一个评估跑两次，结果不同（一次 85%，一次 78%）。

**原因：** LLM 输出有随机性（temperature > 0）。评估结果依赖于具体的采样。

**修复：**

```python
# ❌ 只跑一次评估
result = evaluate(system, test_cases)

# ✓ 多次评估取平均 + 置信区间
results = []
for seed in range(5):
    set_seed(seed)
    results.append(evaluate(system, test_cases))

avg = sum(results) / len(results)
std = (sum((r - avg) ** 2 for r in results) / len(results)) ** 0.5
print(f"准确率: {avg:.2%} ± {std:.2%}")
```

---

## 8. 面试考点

### Q1：多智能体系统的评估和单智能体系统的评估有什么本质区别？（难度：⭐⭐）

**参考答案：**
本质区别是**评估维度增加了"协作"层**。单智能体评估只看"输入→输出"的质量。多智能体评估需要同时看三层：单智能体的个体能力、智能体间的协作质量、系统的端到端表现。最关键的新指标是**协作增益**——多智能体方案是否真的比单智能体方案更好。如果协作开销（更多的词元、更高的延迟、更复杂的故障处理）超过了质量提升，多智能体就是过度工程。

### Q2：如何设计一个多智能体评估数据集？（难度：⭐⭐⭐）

**参考答案：**
评估数据集需要覆盖三个维度。第一，**任务复杂度梯度**——从简单（单智能体就够了）到复杂（必须多智能体协作）。这样才能测量协作增益随任务复杂度的变化。第二，**故障场景**——包含智能体超时、返回错误结果、输出格式异常的场景。评估系统在故障下的降级行为。第三，**协作冲突场景**——包含多个智能体输出互相矛盾的场景。评估系统的一致性检测和冲突解决能力。数据集规模建议 50-100 个用例，每个用例包含：任务描述、预期输出、中间步骤的预期结果、故障注入点。

### Q3：你的多智能体系统在评估时表现很好，但上线后用户满意度很低。可能的原因有哪些？（难度：⭐⭐⭐）

**参考答案：**
五个可能的原因。第一，**评估数据集偏差**——评估用例是开发者构造的，不代表真实用户输入。第二，**评估指标遗漏**——只评估了正确性，没有评估"有用性"——输出正确但不够详细或不够实用。第三，**延迟未评估**——多智能体方案更慢，用户等不及。第四，**边界情况未覆盖**——评估数据集没有包含长文本、多语言、格式异常的输入。第五，**成本未评估**——用户因为成本太高而减少使用，但满意度调查没有捕捉到这一点。解决方案：在评估中加入"延迟分布"、"成本分布"、"真实用户采样"三个维度。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 协作增益 | "多智能体值不值" | 多智能体方案的质量 / 单智能体方案的质量。大于 1 才值得 |
| 协作效率 | "信息传递的浪费" | 最终输出词元 / 所有智能体总词元。越高说明冗余越少 |
| 一致性分数 | "智能体是否自相矛盾" | 多个智能体输出的语义一致性。低于阈值说明协作有问题 |
| LLM-as-Judge | "用 AI 评估 AI" | 用另一个 LLM 作为裁判评估输出质量。自动化但有偏差 |
| 对抗评估 | "故意搞破坏" | 注入故障观察系统降级行为。评估鲁棒性而非正常性能 |
| 协作级评估 | "智能体之间好不好" | 衡量智能体间的信息传递、一致性、冲突解决 |
| 端到端评估 | "最终结果好不好" | 从用户请求到最终响应的完整链路评估 |

---

## 📚 小结

多智能体评估必须回答单智能体评估无法回答的问题：协作是否比单干更好。评估从三个层面展开——系统级（最终结果）、协作级（智能体间交互）、单智能体级（个体能力）。核心指标是**协作增益**——如果多智能体方案的质量没有显著优于单智能体方案，那么多智能体就是过度工程。评估方法包括离线评估（标注数据集）、在线评估（A/B 测试）、对抗评估（故障注入）。记住：评估本身也有成本——LLM-as-Judge 的每次评估都是一次 API 调用。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么"每个智能体准确率 90%，系统准确率也是 90%"是不成立的？用一个具体例子说明协作如何降低整体准确率。

2. 【实现】扩展 `EvaluationMetrics` 类，添加一个 `compute_cost_per_task` 方法——计算单个任务的平均成本。在 100 个任务上运行，报告成本分布（P50、P90、P99）。

3. 【实验】用 3 个不同的 temperature（0.0、0.5、1.0）运行多智能体系统 10 次。报告一致性分数的变化。temperature 越高，一致性是否越低？

4. 【思考】如果你的多智能体系统需要通过合规审计（如金融、医疗领域），评估需要增加哪些维度？如何设计审计友好的评估报告？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 多智能体评估指标 | `code/eval_metrics.py` | 端到端评估指标定义，含协作效率和一致性计算 |
| 评估流水线 | `code/eval_pipeline.py` | 自动化评估框架，支持单智能体 vs 多智能体对比 |
| LLM-as-Judge 模板 | `outputs/llm-judge-prompt.md` | 可复用的 LLM 裁判提示词模板（中英文） |

---

## 📖 参考资料

1. [论文] Wu, J. et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation". 2023. https://arxiv.org/abs/2308.08155 — AutoGen 多智能体评估框架
2. [官方文档] LangSmith Evaluation. https://docs.smith.langchain.com/evaluation — LangSmith 评估指南
3. [论文] Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". NeurIPS, 2023. https://arxiv.org/abs/2306.05685 — LLM-as-Judge 方法论
4. [官方文档] OpenAI Evals. https://github.com/openai/evals — OpenAI 的评估框架
5. [论文] Park, J. S. et al. "Generative Agents: Interactive Simulacra of Human Behavior". UIST, 2023. https://arxiv.org/abs/2304.03442 — 多智能体社会模拟的评估方法

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
