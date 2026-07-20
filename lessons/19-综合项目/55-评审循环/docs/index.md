# 综合项目55——评审循环（Critic Loop）

> 第一次就返回"看起来不错"的评审器是坏的。总是返回"需要改进"的评审器也是坏的。有趣的评审器是会收敛的那个——你必须工程化收敛。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 在五个固定维度上评分论文草稿：清晰度、新颖性、证据、方法论、相关工作
- 将每轮评审应用为结构化修订差异而非自由格式重写
- 通过跨轮次比较分数检测收敛
- 使用最大迭代预算上限，使不收敛的评审不会永远运行
- 发出每轮迹线，供仪表板或下一阶段渲染分数轨迹

---

## 1. 问题

自由格式的评审器返回一段建议文本。下一轮的修订将这段文本视为环境上下文。重写是否解决了批评是无法验证的，因为批评从未有过结构。

五个维度给了编排器一个合约。分数是一个向量。编排器跨轮次观察每个维度。提升清晰度但降低证据的修订是证据的回归——收敛检查能看到它。

---

## 2. 核心概念

### 2.1 评审数据结构

```text
Critique
  scores: Dict[str, float]  # clarity, novelty, evidence, methodology, related_work
  suggestions: List[Suggestion(dimension, target, edit)]
  round: int
  overall_reason: str
```

### 2.2 收敛规则（按顺序）

1. 如果五个维度 ≥ target_score（默认 8.0） → converged: target
2. 如果连续两轮提升 < plateau_epsilon（默认 0.1） → converged: plateau
3. 如果轮次 ≥ max_rounds（默认 5） → stopped: budget

### 2.3 确定性评审器

本课不使用模型。评审器基于三个信号评分：平均章节体长度（清晰度）、图表和引用数（证据）、论文元数据的原创性标签（新颖性）。

---

## 3. 从零实现

```python
"""评审循环——5 维评分+收敛检测+轮次迹线。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Suggestion:
    dimension: str; target_section: str; edit: str

@dataclass
class Critique:
    scores: Dict[str, float]; suggestions: List[Suggestion]; round: int; overall_reason: str

@dataclass
class LoopResult:
    final_draft: "MiniPaper"; convergence: str; rounds: int; trace: List[Dict]


class MiniPaper:
    def __init__(self, abstract="", sections=None, originality_tag="medium", figures=None):
        self.abstract = abstract
        self.sections = sections or {"intro": "", "method": "", "results": ""}
        self.originality_tag = originality_tag
        self.figures = figures or []

    def copy(self):
        return MiniPaper(self.abstract, dict(self.sections), self.originality_tag, list(self.figures))


DIMS = ["clarity", "novelty", "evidence", "methodology", "related_work"]

def deterministic_critic(draft: MiniPaper, round: int) -> Critique:
    scores = {}
    # clarity: avg section body length
    lengths = [len(v) for v in draft.sections.values()]
    avg_len = sum(lengths) / max(len(lengths), 1)
    scores["clarity"] = min(10, avg_len / 15)
    # novelty
    scores["novelty"] = {"high": 9.0, "medium": 6.0, "low": 3.0}.get(draft.originality_tag, 5.0)
    # evidence
    scores["evidence"] =  min(10, (len(draft.figures) * 2) + 2)
    # methodology
    scores["methodology"] = 7.0 if "method" in draft.sections and draft.sections["method"] else 2.0
    # related work
    scores["related_work"] = 6.0 if "related" in draft.sections or "related_work" in draft.sections else 2.0
    suggestions = []
    if scores["clarity"] < 7:
        suggestions.append(Suggestion("clarity", "all", "扩展章节内容"))
    if scores["novelty"] < 7:
        suggestions.append(Suggestion("novelty", "meta", "明确新颖性声明"))
    if scores["evidence"] < 7:
        suggestions.append(Suggestion("evidence", "results", "添加更多图表"))
    if scores["methodology"] < 7:
        suggestions.append(Suggestion("methodology", "method", "详细描述方法"))
    if scores["related_work"] < 7:
        suggestions.append(Suggestion("related_work", "intro", "补充相关工作"))
    return Critique(scores, suggestions, round, f"第{round}轮评审完成")


def deterministic_reviser(draft: MiniPaper, suggestions: List[Suggestion]) -> MiniPaper:
    revised = draft.copy()
    for s in suggestions:
        if s.dimension == "clarity" and s.target_section in revised.sections:
            revised.sections[s.target_section] += " 添加更多细节以提高清晰度。"
        elif s.dimension == "evidence":
            revised.figures.append(f"fig_{s.dimension}_{len(revised.figures)}")
        elif s.dimension == "novelty":
            revised.originality_tag = "high"
        elif s.dimension == "related_work":
            if "related" not in revised.sections:
                revised.sections["related"] = ""
            revised.sections["related"] += " 讨论了相关工作。"
    return revised


class CriticLoop:
    def __init__(self, critic: Callable, reviser: Callable, max_rounds=5, target_score=8.0, plateau_epsilon=0.1):
        self.critic = critic; self.reviser = reviser
        self.max_rounds = max_rounds; self.target_score = target_score; self.plateau_epsilon = plateau_epsilon

    def run(self, draft: MiniPaper) -> LoopResult:
        trace = []
        prev_mean = 0.0; plateau_count = 0
        for r in range(1, self.max_rounds + 1):
            critique = self.critic(draft, r)
            scores = critique.scores
            mean_score = sum(scores.values()) / len(scores)
            trace.append({"round": r, "scores": scores, "suggestion_count": len(critique.suggestions)})
            if all(s >= self.target_score for s in scores.values()):
                return LoopResult(draft, "target", r, trace)
            if r > 1 and (mean_score - prev_mean) < self.plateau_epsilon:
                plateau_count += 1
                if plateau_count >= 2:
                    return LoopResult(draft, "plateau", r, trace)
            else:
                plateau_count = 0
            prev_mean = mean_score
            if r < self.max_rounds:
                draft = self.reviser(draft, critique.suggestions)
        return LoopResult(draft, "budget", self.max_rounds, trace)


def main():
    draft = MiniPaper(
        abstract="研究注意力稀疏性。",
        sections={"intro": "引言内容。", "method": "", "results": ""},
        originality_tag="medium",
    )
    loop = CriticLoop(deterministic_critic, deterministic_reviser)
    result = loop.run(draft)
    print(f"收敛状态: {result.convergence}")
    print(f"轮次数: {result.rounds}")
    for t in result.trace:
        scores = ", ".join(f"{k}={v:.1f}" for k, v in t["scores"].items())
        print(f"  第{t['round']}轮: {scores} 建议数: {t['suggestion_count']}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 分数向量 | 五个维度的评分——结构化的评审合约 |
| 目标收敛 | 所有维度达到目标分数（默认 8.0） |
| 平台收敛 | 连续两轮改进幅度低于 epsilon |
| 预算收敛 | 达到最大轮次上限时停止 |
| 修订差异 | 基于建议的结构化追加而非全文重写 |

---

## 5. 工程最佳实践

- **迹线优先**：始终输出每轮的分数轨迹——这比仅输出最终草稿包含更多诊断信息。
- **中文场景建议**：评审器和修订器都使用中文，确保非英语使用者也能理解评审反馈。

---

## 6. 常见错误

- **单轮平台检测过于敏感**：一轮分数持平可能是噪声。连续两轮才是真正的平台。
- **维度权重未考虑**：某些任务中新颖性比清晰度更重要——收敛检查应支持加权平均。
- **评审器未传递轮次数**：评审器需要知道轮次数以调整苛刻程度（早期关键问题，后期精细调整）。

---

## 📖 参考资料

1. [GitHub] ChatGPT Critic — LLM 作为评审器的实现参考.
