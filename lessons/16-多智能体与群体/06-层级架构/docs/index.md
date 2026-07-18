# 层级架构及其失败模式

> 层级是嵌套的监督者。管理者智能体管理子管理者，子管理者管理工作组。CrewAI `Process.hierarchical` 是教科书版本：`manager_llm` 动态委派任务并验证输出。LangGraph 的等价物是 `create_supervisor(create_supervisor(...))`。当任务是真实组织结构图时，这是自然模式。它也是最可能崩溃为管理循环的模式——管理者智能体委派不当、误解子输出或无法达成共识。顺序流水线常常优于它。

**类型：** 概念课 + 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 16 · 05（监督者模式）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释层级架构的结构——管理者管理子管理者，子管理者管理工作组
- [ ] 识别三种 2026 年事后分析中反复出现的失败模式：任务分配错误、输出误解、共识循环
- [ ] 实现一个三级层级结构并演示分解漂移——管理者错误标记分支导致错误答案级联
- [ ] 决定何时使用层级（真实组织结构）vs 顺序（线性流水线）vs 平坦监督者

---

## 1. 问题

一旦监督者模式成立，自然的下一步是"如果工作者本身也是管理者会怎样？"团队有子团队；公司有部门中的部门。层级架构反映了这一点。

问题：LLM 管理者不同于人类管理者。人类管理者对下属知道什么有稳定的先验。LLM 管理者每轮从上下文中的内容重新推理组织。上下文中的微小漂移，整个树就错误分配工作。

---

## 2. 概念

### 2.1 形状

```
                 管理者
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       子管理者 A         子管理者 B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个内部节点计划、委派和合成。只有叶子做工作。

### 2.2 三种失败模式（2026 年事后分析反复发现）

| 失败 | 说明 | 为什么危险 |
|------|------|----------|
| **任务分配错误** | 管理者读目标，幻觉分解，委派给错误子管理者 | 错误在顶层合成时才暴露 |
| **输出误解** | 子管理者返回"无法验证 X"，顶层总结为"X 未确认" | 含义在每一级漂移 |
| **共识循环** | 两个子管理者不同意；顶层要求协调；他们重新委派；循环 | CrewAI 的步骤限制防止但本身成为超参数 |

### 2.3 决定性问题

顺序（线性流水线）vs 层级：**你的任务真的有独立子团队吗，还是一个线性流水假装是树？** 如果是后者，用顺序。如果是前者，用层级但预算明确的协调规则。

### 2.4 CrewAI 的实现

`Process.hierarchical` 将管理者 LLM 连接到专家团队。管理者：接收顶层任务 → 委派子任务给团队 → 评估团队输出 → 决定接受、重新委派或迭代。

### 2.5 LangGraph 的实现

LangGraph 使用嵌套的 `create_supervisor` 调用。内部监督者有自己的图；外部监督者将内部图视为不透明节点。调试更干净（可以单独步进每个图），但表达动态树重塑更难。

---

## 3. 从零实现

### 第 1 步：定义工作者和子管理者

```python
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

class Worker:
    def __init__(self, name, canned):
        self.name = name
        self.canned = canned

    def run(self, question):
        key = next((k for k in self.canned if k in question.lower()), "default")
        return LeafOutput(self.name, question, self.canned.get(key, "[无预设答案]"))

class SubManager:
    def __init__(self, name, workers, split):
        self.name = name
        self.workers = workers
        self.split = split

    def run(self, task):
        leaves = []
        for w in self.workers:
            sub_q = self.split.get(w.name, task)
            leaves.append(w.run(sub_q))
        summary = f"[{self.name}] 聚合: " + " | ".join(l.answer for l in leaves)
        return SubSummary(self.name, leaves, summary)
```

### 第 2 步：实现顶层管理者

```python
class TopManager:
    def __init__(self, name, subs):
        self.name = name
        self.subs = subs

    def run(self, task, branch_labels):
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
```

### 第 3 步：构建层级结构并演示

```python
def main():
    top = build_hierarchy()  # vp-eng → eng-manager + legal-manager

    # 快乐路径：正确分支
    happy = top.run("Ship premium feature.", branch_labels=["engineering", "legal"])
    render("快乐路径（正确分支）", happy)

    # 扰动路径：管理者误标 'legal' 为 'finance'
    perturbed = top.run("Ship premium feature.", branch_labels=["engineering", "finance"])
    render("扰动路径（顶层管理者误标）", perturbed)

    print("用户询问法律/工程审查。")
    print("快乐路径：法律和工程都如实回答。")
    print("扰动路径：财务忠实地回答，法律问题无人回答。")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 CrewAI vs LangGraph

| 特性 | CrewAI `Process.hierarchical` | LangGraph 嵌套 `create_supervisor` |
|------|---------------------------|-----------------------------------|
| 易用性 | 更高——声明式 | 更低——手动图构建 |
| 调试 | 困难——步骤限制是超参数 | 更干净——可以步进每个图 |
| 协调循环检测 | 步骤限制防护 | 无内置防护 |
| 动态树重塑 | 不支持 | 更难表达但更灵活 |

---

## 5. 工程最佳实践

### 5.1 层级架构部署原则

| 原则 | 说明 |
|------|------|
| 树深度限制为 2 | 3+ 层使可观测性崩溃 |
| 明确协调预算 | 设定顶层管理者必须提交前的最大轮数（通常 2） |
| 每个合成有溯源 | 每个节点的摘要必须引用产生它的叶子输出 |
| 分解漂移警报 | 记录管理者的每步分解；与用户查询对比 |

---

## 6. 常见错误

### 错误 1：任务分配错误

**现象：** 管理者委派给错误的子管理者。子管理者忠实地在错误目标上工作。

**原因：** 管理者的分解与用户查询不匹配，错误在合成时才暴露。

**修复：** 在每个子管理者处添加金丝雀工作者——总是问原始用户查询，检测分解漂移。

### 错误 2：输出误解级联

**现象：** 子管理者说"无法验证 X"，顶层总结为"X 未确认"。含义在每一级漂移。

**原因：** 摘要过程丢失细微差别。

**修复：** 每个合成必须引用具体叶子输出。使用溯源链。

### 错误 3：共识循环

**现象：** 两个子管理者不同意；顶层要求协调；他们重新委派；循环直到预算耗尽。

**原因：** 没有明确的协调规则。

**修复：** 设定协调预算（通常 2 轮）。超过后顶层必须提交。

---

## 7. 面试考点

### Q1：层级架构的三种失败模式是什么？（难度：⭐）

**参考答案：**
1. 任务分配错误——管理者委派给错误子管理者
2. 输出误解——含义在每一级漂移
3. 共识循环——子管理者不同意，顶层反复协调

### Q2：为什么层级架构不是总是正确的选择？（难度：⭐⭐）

**参考答案：**
决定性问题：任务真的有独立子团队吗？如果任务是一个线性流水（步骤 2 字面上需要步骤 1 的输出），用顺序流水线。如果任务是树状（法律审查+工程审查+财务审查），用层级但预算协调规则。

层级的三种失败在顺序流水线中不发生——顺序是确定性的，没有分解漂移、没有输出误解级联、没有共识循环。

### Q3：深度 2 天花板是什么？（难度：⭐⭐⭐）

**参考答案：**
经验观察：3+ 层级使可观测性崩溃。顶层管理者看不到叶子输出——它只看到子管理者的摘要。错误在叶子处产生，在子管理者处被误解，在顶层管理者处被进一步误解——三层后完全不可追溯。

修复：限制树深度为 2。如果任务需要更多层级，拆分为多个顺序流水线而非更深的树。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 层级 | "组织结构图模式" | 管理者管管理者；只有叶子做工作 |
| 管理者 LLM | "老板" | 在内部节点分解、委派、验证的 LLM |
| 分解漂移 | "老板迷失了" | 顶层管理者的拆分不再覆盖原始问题 |
| 协调循环 | "无尽会议" | 子管理者不同意；顶层重新委派；循环直到预算耗尽 |
| 深度 2 天花板 | "不要超过 2 层" | 经验护栏：3+ 层使可观测性崩溃 |

---

## 📚 小结

层级是嵌套的监督者——管理者管子管理者，子管理者管工作组。三种失败反复出现：任务分配错误、输出误解级联、共识循环。深度 2 天花板——3+ 层使错误不可追溯。决定性问题：任务真的有独立子团队吗？如果线性流水，用顺序。如果树状，用层级但预算协调规则。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。对比快乐路径和扰动路径。多少级管理者委派后顶层输出与用户问题完全偏离？

2. **【实现】** 添加第三级（顶→子→子子→工作者）。测量扰动路径随着深度增长自我修正 vs 完全偏离的频率。

3. **【实现】** 在每个子管理者处添加金丝雀工作者——总是被问原始用户查询。用金丝雀答案检测分解漂移。当金丝雀与合成答案不一致时管理者应如何反应？

4. **【阅读】** 阅读 CrewAI `Process.hierarchical` 文档。识别 CrewAI 应用的一个具体护栏并描述它针对的失败模式。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 层级演示 | `code/main.py` | 三级层级 + 分解漂移演示 |
| 技能提示词 | `outputs/skill-hierarchy-fitness.md` | 评估任务是否应使用层级、顺序或平坦监督者 |

---

## 📖 参考资料

1. [文档] CrewAI. "Process.hierarchical". https://docs.crewai.com/en/introduction
2. [文档] LangGraph Supervisor. https://reference.langchain.com/python/langgraph-supervisor
3. [博客] Anthropic. "Research System". https://www.anthropic.com/engineering/multi-agent-research-system — 为什么 Anthropic 选择平坦监督者而非层级
4. [论文] Cemri et al. "Why Do Multi-Agent LLM Systems Fail?". https://arxiv.org/abs/2503.13657 — MAST 分类法；分解漂移文档

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
