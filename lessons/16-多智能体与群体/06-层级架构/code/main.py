"""层级多智能体与分解漂移演示。

三级层级：顶层管理者 → 子管理者 → 工作者。
演示快乐路径和扰动路径——顶层管理者错误标记一个分支。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeafOutput:
    worker: str
    question: str
    answer: str


@dataclass
class SubSummary:
    sub_manager: str
    leaves: list[LeafOutput]
    summary: str


@dataclass
class TopSynthesis:
    top_manager: str
    branches: list[SubSummary]
    synthesis: str


class Worker:
    def __init__(self, name: str, canned: dict[str, str]) -> None:
        self.name = name
        self.canned = canned

    def run(self, question: str) -> LeafOutput:
        key = next((k for k in self.canned if k in question.lower()), "default")
        ans = self.canned.get(key, f"[无预设答案: '{question}']")
        return LeafOutput(self.name, question, ans)


class SubManager:
    def __init__(self, name: str, workers: list[Worker], split: dict[str, str]) -> None:
        self.name = name
        self.workers = workers
        self.split = split

    def run(self, task: str) -> SubSummary:
        leaves = []
        for w in self.workers:
            sub_q = self.split.get(w.name, task)
            leaves.append(w.run(sub_q))
        summary = f"[{self.name}] 聚合: " + " | ".join(l.answer for l in leaves)
        return SubSummary(self.name, leaves, summary)


class TopManager:
    def __init__(self, name: str, subs: dict[str, SubManager]) -> None:
        self.name = name
        self.subs = subs

    def run(self, task: str, branch_labels: list[str]) -> TopSynthesis:
        summaries = []
        for label in branch_labels:
            if label not in self.subs:
                summaries.append(SubSummary(
                    f"MISSING[{label}]", [],
                    f"[top] 试图委派给 '{label}'——无此子管理者"))
                continue
            summaries.append(self.subs[label].run(f"{task} -- 分支: {label}"))
        synth = "top synthesis: " + " || ".join(s.summary for s in summaries)
        return TopSynthesis(self.name, summaries, synth)


def build_hierarchy() -> TopManager:
    fe = Worker("fe", {"frontend": "React 组件审查完成；2 个问题。"})
    be = Worker("be", {"backend": "API 端点审查完成；1 个问题。"})
    eng = SubManager("eng-manager", [fe, be],
                      {"fe": "功能的前端审查", "be": "功能的后端审查"})
    lw = Worker("lawyer", {"legal": "合同条款 A 和 B 不合规。"})
    legal = SubManager("legal-manager", [lw], {"lawyer": "功能的法律审查"})
    fw = Worker("finance", {"finance": "预计成本 $42k/月；超出预算 12%。"})
    finance = SubManager("finance-manager", [fw], {"finance": "功能的财务审查"})
    return TopManager("vp-eng", {"engineering": eng, "legal": legal, "finance": finance})


def render(label: str, synth: TopSynthesis) -> None:
    print(f"\n=== {label} ===")
    for branch in synth.branches:
        print(f"  [子管理者] {branch.sub_manager}")
        for leaf in branch.leaves:
            print(f"    [叶子] {leaf.worker:8s} 被问: {leaf.question}")
            print(f"           回答: {leaf.answer}")
        print(f"    [摘要] {branch.summary}")
    print(f"  [顶层] {synth.synthesis}")


def main() -> None:
    print("层级多智能体与分解漂移演示")
    print("-" * 60)

    top = build_hierarchy()
    task = "将高级功能发布到生产环境。"

    happy = top.run(task, branch_labels=["engineering", "legal"])
    render("快乐路径（正确分支）", happy)

    perturbed = top.run(task, branch_labels=["engineering", "finance"])
    render("扰动路径（顶层管理者误标 'legal' 为 'finance'）", perturbed)

    print("\n用户询问法律/工程审查。")
    print("快乐路径：法律和工程都如实回答。")
    print("扰动路径：财务忠实地回答，法律问题无人回答。")
    print("错误在顶层合成时出现——比人工本可以捕获的地方高一级。")


if __name__ == "__main__":
    main()
