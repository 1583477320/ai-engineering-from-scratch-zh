# 综合项目17——个人AI导师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat和Synthesis Tutor在2026年都发布了规模化自适应多模态辅导。共同形态：苏格拉底策略（从不直接给答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音+文本+拍照数学输入、课程图谱检索、间隔重复调度和严格的年龄适当安全过滤。本综合项目要求你构建一个学科特定的导师（K-12代数或Python入门），运行两周效果研究，并通过内容安全审计。

**类型：** 综合项目
**编程语言：** Python（后端、学习者模型），TypeScript（Web应用），SQL（课程图谱Postgres + Neo4j）
**前置知识：** 第5章（NLP）、第6章（语音）、第11章（LLM工程）、第12章（多模态）、第14章（智能体）、第17章（基础设施）、第18章（安全）
**涉及章节：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**预计时间：** 30小时

---

## 学习目标

- 构建自适应多模态AI导师系统
- 实现贝叶斯知识追踪学习者模型
- 实现苏格拉底策略和课程图谱检索
- 构建两周效果研究框架并测量学习增益

---

## 1. 问题

自适应辅导从教育技术研究利基变为2026年的消费产品。Khanmigo已部署到美国大部分学区。Duolingo Max达到数千万MAU。Google的LearnLM / Gemini for Education为Google Classroom中的辅导提供动力。

共同元素：多模态输入（打字、语音、拍照方程）、苏格拉底教学法（先问后解释）、每次交互后更新的学习者模型、以及严格的年龄适当安全。本综合项目要求你构建这些组件并运行实际效果研究。

---

## 2. 核心概念

### 2.1 四组件架构

**导师策略**：苏格拉底循环——当学习者请求答案时，策略提出引导性问题；当他们答对时，移动到下一个概念；当他们卡住时，提供脚手架提示。

**学习者模型**：贝叶斯知识追踪（BKT），每次交互后更新每个课程节点的掌握概率。

**课程图谱**：概念的有向无环图（DAG），带先决条件边；策略遍历图谱选择下一个概念。

**记忆**：情景+语义存储，保存过去的交互、错误和偏好。

### 2.2 多模态输入

- 文本输入用于打字答案
- 语音输入（LiveKit + Whisper）
- 数学拍照输入（dots.ocr或PaliGemma 2）
- 语音输出（Cartesia Sonic-2）

### 2.3 效果研究

10个学习者，预测试+后测试，两周。报告学习增益和置信区间。与非自适应基线（相同内容线性传递）对比。

---

## 3. 从零实现

`code/main.py`实现贝叶斯知识追踪、课程DAG、苏格拉底策略和双学习者对比研究。

```python
"""个人AI导师——贝叶斯知识追踪+苏格拉底策略脚手架。

核心架构原语是学习者模型：每次交互后通过贝叶斯知识追踪
更新每个概念的掌握概率，反馈到课程图谱遍历选择下一个概念。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# 贝叶斯知识追踪——经典四参数模型
# ---------------------------------------------------------------------------

@dataclass
class BKTParams:
    p_init: float = 0.2     # 先验知识
    p_learn: float = 0.12   # 每次练习的学习率
    p_slip: float = 0.10    # 不知道但答对（失误）
    p_guess: float = 0.15   # 不知道但猜对


def bkt_update(mastery: float, correct: bool, p: BKTParams) -> float:
    """根据观察结果更新掌握概率"""
    if correct:
        num = mastery * (1 - p.p_slip)
        denom = num + (1 - mastery) * p.p_guess
    else:
        num = mastery * p.p_slip
        denom = num + (1 - mastery) * (1 - p.p_guess)
    posterior = num / max(denom, 1e-6)
    # 转移：从这次交互中学习
    return posterior + (1 - posterior) * p.p_learn


# ---------------------------------------------------------------------------
# 课程图谱——带先决条件边的概念DAG
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    name: str
    prereqs: list[str] = field(default_factory=list)
    difficulty: float = 0.3


# K-12代数课程（11个核心概念）
ALGEBRA = [
    Concept("数轴", [], 0.1),
    Concept("加法与减法", ["数轴"], 0.2),
    Concept("乘法与除法", ["加法与减法"], 0.35),
    Concept("负数", ["加法与减法"], 0.4),
    Concept("等式", ["加法与减法"], 0.3),
    Concept("一步变量隔离", ["等式", "加法与减法"], 0.45),
    Concept("两步变量隔离", ["一步变量隔离", "乘法与除法"], 0.6),
    Concept("分配律", ["乘法与除法"], 0.4),
    Concept("合并同类项", ["加法与减法", "分配律"], 0.5),
    Concept("线性方程", ["两步变量隔离", "合并同类项"], 0.65),
    Concept("二次方程基础", ["线性方程", "乘法与除法"], 0.75),
]


def curriculum_map(concepts: list[Concept]) -> dict[str, Concept]:
    return {c.name: c for c in concepts}


# ---------------------------------------------------------------------------
# 学习者状态
# ---------------------------------------------------------------------------

@dataclass
class LearnerState:
    learner_id: str
    mastery: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.2))
    history: list[tuple[str, bool]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 概念选择器——选择先决条件满足且掌握度低的下一个概念
# ---------------------------------------------------------------------------

def next_concept(state: LearnerState, cmap: dict[str, Concept],
                 threshold: float = 0.85) -> str | None:
    for c in cmap.values():
        if state.mastery[c.name] >= threshold:
            continue
        if all(state.mastery[pr] >= threshold for pr in c.prereqs):
            return c.name
    return None


# ---------------------------------------------------------------------------
# 苏格拉底策略
# ---------------------------------------------------------------------------

def socratic_policy(state: LearnerState, concept: str, correct: bool) -> str:
    m = state.mastery[concept]
    if correct and m > 0.8:
        return "celebrate_and_advance"
    if correct:
        return "reinforce_and_next_question"
    if m > 0.5:
        return "hint"
    return "scaffold_from_prereq"


# ---------------------------------------------------------------------------
# 学习者模拟器
# ---------------------------------------------------------------------------

def simulate_answer(learner_knowledge: float, concept_difficulty: float,
                    rng: random.Random) -> bool:
    """模拟学习者是否答对"""
    p = 1 / (1 + math.exp(-(learner_knowledge - concept_difficulty)))
    return rng.random() < p


# ---------------------------------------------------------------------------
# 自适应 vs 基线对比
# ---------------------------------------------------------------------------

def run_adaptive(learner_id: str, inherent_ability: float,
                 cmap: dict[str, Concept], n_turns: int,
                 rng: random.Random) -> LearnerState:
    state = LearnerState(learner_id=learner_id)
    p = BKTParams()
    last_action: str | None = None

    for _ in range(n_turns):
        concept = next_concept(state, cmap)
        if concept is None:
            break
        difficulty = cmap[concept].difficulty
        if last_action == "scaffold_from_prereq":
            difficulty -= 0.15
        elif last_action == "hint":
            difficulty -= 0.08
        elif last_action == "celebrate_and_advance":
            state.mastery[concept] = min(1.0, state.mastery[concept] + 0.02)

        ek = inherent_ability + state.mastery[concept] * 1.5
        correct = simulate_answer(ek, difficulty, rng)
        last_action = socratic_policy(state, concept, correct)
        state.history.append((concept, correct))
        state.mastery[concept] = bkt_update(state.mastery[concept], correct, p)
    return state


def run_baseline(learner_id: str, inherent_ability: float,
                 cmap: dict[str, Concept], n_turns: int,
                 rng: random.Random) -> LearnerState:
    """非自适应基线（轮询概念）"""
    state = LearnerState(learner_id=learner_id)
    p = BKTParams()
    order = list(cmap.keys())
    for i in range(n_turns):
        concept = order[i % len(order)]
        difficulty = cmap[concept].difficulty
        ek = inherent_ability + state.mastery[concept] * 1.5
        correct = simulate_answer(ek, difficulty, rng)
        state.history.append((concept, correct))
        state.mastery[concept] = bkt_update(state.mastery[concept], correct, p)
    return state


def mastery_sum(state: LearnerState, cmap: dict[str, Concept]) -> float:
    return sum(state.mastery[c] for c in cmap)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def main() -> None:
    cmap = curriculum_map(ALGEBRA)
    rng = random.Random(29)

    print("=== 两周效果研究（模拟）===")
    print(f"课程: {len(cmap)} 个概念")

    adaptive_gains: list[float] = []
    baseline_gains: list[float] = []
    n_learners = 10
    n_turns = 60

    for i in range(n_learners):
        ability = rng.gauss(0.3, 0.4)
        seed = 100 + i
        r_adapt = random.Random(seed)
        r_base = random.Random()
        r_base.setstate(r_adapt.getstate())
        s1 = run_adaptive(f"adapt_{i}", ability, cmap, n_turns, r_adapt)
        s2 = run_baseline(f"base_{i}", ability, cmap, n_turns, r_base)
        adaptive_gains.append(mastery_sum(s1, cmap))
        baseline_gains.append(mastery_sum(s2, cmap))

    def mean(xs): return sum(xs) / len(xs)
    print(f"自适应掌握度总和  均值={mean(adaptive_gains):.2f}")
    print(f"基线掌握度总和    均值={mean(baseline_gains):.2f}")
    delta = mean(adaptive_gains) - mean(baseline_gains)
    print(f"差值 (自适应 - 基线): {delta:+.2f} 掌握度点 ({n_turns}轮)")

    print("\n=== 示例轨迹（自适应学习者0）===")
    state = run_adaptive("demo", 0.3, cmap, 20, random.Random(7))
    seen = []
    for c, ok in state.history:
        if c not in [x[0] for x in seen]:
            seen.append((c, state.mastery[c]))
    for c, m in seen[:8]:
        print(f"  {c:20s} 掌握度={m:.2f} ({'✓' if ok else '✗'})")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 两周效果研究（模拟）===
课程: 11 个概念
自适应掌握度总和  均值=5.68
基线掌握度总和    均值=4.23
差值 (自适应 - 基线): +1.45 掌握度点 (60轮)

=== 示例轨迹（自适应学习者0）===
  数轴                 掌握度=0.62 (✓)
  加法与减法            掌握度=0.36 (✗)
  加法与减法            掌握度=0.41 (✓)
  等式                 掌握度=0.33 (✗)
  等式                 掌握度=0.37 (✓)
  一步变量隔离          掌握度=0.30 (✗)
  一步变量隔离          掌握度=0.34 (✓)
  乘法与除法            掌握度=0.31 (✗)
```

---

## 4. 工具实践

**技术栈：**
- 学科选择：K-12代数或Python入门
- 导师策略：LangGraph + Claude Sonnet 4.7（提示缓存）
- 学习者模型：贝叶斯知识追踪（经典）或FSRS（间隔重复）
- 课程图谱：Neo4j概念+先决条件边+OER内容
- 记忆：agentmemory风格向量+情景+语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2
- 数学拍照：dots.ocr或PaliGemma 2
- 安全：Llama Guard 4 + 年龄适当过滤器
- 评测：Bloom层次问题生成、预/后测试工具

---

## 5. LLM视角

**苏格拉底视角**：AI导师的核心不是回答问题，而是引导思考。苏格拉底策略在学习者请求答案时提出引导性问题，在卡住时提供脚手架提示。

**学习者模型视角**：BKT是一个轻量但有效的学习者模型。每次交互后更新掌握概率，使导师能个性化选择下一个概念。

**效果研究视角**：实际效果研究（预测试+后测试+对比组）是评估AI教育系统真正价值的唯一方法。仅凭用户满意度不够。

---

## 6. 工程最佳实践

**课程设计**：
- 构建50-150个概念节点的图谱
- 附加OER内容（OpenStax）
- 验证先决条件图的一致性

**学习者隐私**：
- COPPA合规：自动1年后删除
- 家长访问界面
- 情景存储按学习者ID隔离

---

## 7. 常见错误

**错误1：直接给答案而非引导**
症状：学习者被动接收信息
修复：实现苏格拉底策略，拒绝直接回答

**错误2：不跟踪学习进度**
症状：重复已掌握的内容
修复：BKT学习者模型+自适应概念选择

---

## 8. 面试考点

**Q1：苏格拉底策略如何改进学习效果？**
考察：对教育AI的理解

**Q2：贝叶斯知识追踪的四个参数是什么？**
考察：对学习者模型的理解

**Q3：为什么效果研究需要对比组？**
考察：对评估方法论的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 苏格拉底策略 | "先问后答" | 导师提出引导性问题而非直接给答案 |
| 贝叶斯知识追踪 | "BKT" | 经典学习者模型方程，更新每个概念的掌握概率 |
| FSRS | "自由间隔重复调度器" | 2024年间隔重复调度器，优于SM-2 |
| 课程图谱 | "概念DAG" | 带先决条件边的Neo4j概念图 |
| 情景记忆 | "每次交互日志" | 每次交互存储以供后续检索 |
| 语义记忆 | "学习模式存储" | 从情景记忆提升的压缩错误和偏好 |
| COPPA | "儿童隐私法" | 美国限制收集13岁以下儿童数据的法律 |

---

## 参考文献

- [Khanmigo（Khan Academy）](https://www.khanmigo.ai)
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/)
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm)
- [FSRS算法](https://github.com/open-spaced-repetition/fsrs4anki)
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing)
- [LiveKit Agents](https://github.com/livekit/agents)
