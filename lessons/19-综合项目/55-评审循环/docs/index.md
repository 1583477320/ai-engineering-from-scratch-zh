# 综合项目55——评审循环（Critic Loop）

> 第一次就"看起来不错"的评审器是坏的。一直"需要改进"的评审器也是坏的。有趣的评审器是会收敛的那个。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 在五个固定维度评分草稿：清晰度、新颖性、证据、方法论、相关工作
- 应用结构化修订差异而非自由格式重写
- 检测收敛：目标达成、平台、预算耗尽

---

## 1. 问题

自由格式评审器返回一段建议文本，下一轮修订将文本视为环境上下文。重写是否解决批评无法验证。五个维度给了编排器合约，分数是向量，收敛检查观察每个维度。

---

## 2. 核心概念

### 2.1 评审数据结构

```text
Critique(scores: {dim: float}, suggestions: [Suggestion], round: int)
```

### 2.2 收敛规则

1. 五维 ≥ target_score（8.0）→ target
2. 连续两轮提升 < epsilon（0.1）→ plateau
3. 轮次 ≥ max（5）→ budget

---

## 3. 从零实现

```python
"""评审循环——5 维评分+收敛检测。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List

@dataclass
class Suggestion:
    dimension: str; target_section: str; edit: str

@dataclass
class Critique:
    scores: Dict[str, float]; suggestions: List[Suggestion]; round: int

@dataclass
class LoopResult:
    convergence: str; rounds: int; trace: List[Dict]

class MiniPaper:
    def __init__(self, abstract="", sections=None, originality="medium", figures=None):
        self.abstract = abstract
        self.sections = sections or {"intro":"","method":"","results":""}
        self.originality = originality; self.figures = figures or []
    def copy(self):
        return MiniPaper(self.abstract, dict(self.sections), self.originality, list(self.figures))

DIMS = ["clarity","novelty","evidence","methodology","related_work"]

def critic(draft: MiniPaper, rnd: int) -> Critique:
    avg = sum(len(v) for v in draft.sections.values())/max(len(draft.sections),1)
    s = {"clarity":min(10,avg/15), "novelty":{"high":9,"medium":6,"low":3}.get(draft.originality,5),
         "evidence":min(10,len(draft.figures)*2+2), "methodology":7 if draft.sections.get("method") else 2,
         "related_work":6 if any("related" in k for k in draft.sections) else 2}
    su = []
    for dim,thresh in [("clarity",7),("novelty",7),("evidence",7),("methodology",7),("related_work",7)]:
        if s[dim] < thresh: su.append(Suggestion(dim,"all",f"提高{dim}"))
    return Critique(s, su, rnd)

def reviser(draft: MiniPaper, su: List[Suggestion]) -> MiniPaper:
    d = draft.copy()
    for s in su:
        if s.dimension=="clarity" and s.target_section in d.sections:
            d.sections[s.target_section]+=" 更多细节。"
        elif s.dimension=="evidence": d.figures.append(f"fig_{len(d.figures)}")
        elif s.dimension=="novelty": d.originality="high"
    return d

class CriticLoop:
    def __init__(self, c=None, r=None, max_r=5, target=8.0, eps=0.1):
        self.c=c or critic; self.r=r or reviser; self.max_r=max_r; self.target=target; self.eps=eps
    def run(self, draft: MiniPaper) -> LoopResult:
        trace=[]; prev=0.0; pc=0
        for rnd in range(1,self.max_r+1):
            cr=self.c(draft,rnd); ms=sum(cr.scores.values())/len(cr.scores)
            trace.append({"round":rnd,"scores":cr.scores,"sugs":len(cr.suggestions)})
            if all(s>=self.target for s in cr.scores.values()): return LoopResult("target",rnd,trace)
            if rnd>1 and (ms-prev)<self.eps: pc+=1
            else: pc=0
            if pc>=2: return LoopResult("plateau",rnd,trace)
            prev=ms
            if rnd<self.max_r: draft=self.r(draft,cr.suggestions)
        return LoopResult("budget",self.max_r,trace)

def main():
    d=MiniPaper("稀疏性研究",{"intro":"引言","method":"","results":""},"medium")
    r=CriticLoop().run(d)
    print(f"收敛: {r.convergence} 轮次: {r.rounds}")
    for t in r.trace: print(f"  轮{t['round']}: {', '.join(f'{k}={v:.1f}' for k,v in t['scores'].items())}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
```

---

## 4. 工业工具

| 系统 | 维度 | 自动评分 | 修改建议 |
|:----|:----|:--------|:--------|
| OASYS | 多维度 | ✓ | ✓ |
| ReviewRevise | 5 维 | ✓ | ✓ |
| ChatGPT Critic | 自由格式 | ✗ | ✓ |

---

## 5. 工程最佳实践

- 迹线优先：始终输出每轮分数轨迹
- **中文场景建议**：评审器和修订器使用中文

---

## 6. 常见错误

- **单轮平台检测**：一轮持平可能是噪声，连续两轮才是真平台
- **未传递轮次数**：评审器需知道轮次数调整苛刻程度

---

## 7. 面试考点

**Q1：5 个固定维度比自由格式评审好在哪？**（难度：⭐⭐）

**参考答案：** 维度结构化使收敛可量化——编排器可以检测每个维度的改善，而非依赖不可验证的文本建议。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 分数向量 | 五维评分结构 |
| 目标收敛 | 所有维度达标 |
| 平台收敛 | 连续两轮提升不足 |
| 修订差异 | 结构化追加而非重写 |

---

## 📚 小结

评审循环自动改进草稿质量。下一节将调度器编排假设、实验和评审。

---

## ✏️ 练习

1. 【实现】添加 `dimension_weights` 支持加权收敛
2. 【实验】将 `target_score` 从 6.0 调到 9.0，观察收敛轮次变化

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 评审循环 | `code/main.py` |

---

## 📖 参考资料

1. [GitHub] ReviewRevise. https://github.com/review-revise
