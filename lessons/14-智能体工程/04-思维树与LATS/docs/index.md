# 思维树与 LATS：深思搜索

> 单条思维链轨迹没有回溯空间。ToT（Yao 等人，2023）将推理变成一棵树，在每个节点上做自我评估。LATS（Zhou 等人，2024）将 ToT 与 ReAct 和 Reflexion 在蒙特卡洛树搜索下统一。Game of 24 从 CoT 的 4% 提升到 ToT 的 74%；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、03（Reflexion）| **时间：** ~75 分钟
**所着阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 将推理建模为搜索——节点是"思考"，边是"扩展"，价值是"有多有希望"
- [ ] 实现 ToT 的 BFS/DFS 搜索策略
- [ ] 理解 LATS 如何用 MCTS 统一 ToT、ReAct、Reflexion
- [ ] 对比 CoT、ToT、LATS 的性能差异

---

## 1. 问题

链式思维（CoT）推理是一条直线——每一步推理只能往前走，不能回溯。如果推理的某一步出了错，整个链条就错了。

**思维树（ToT）** 将推理变成一棵树——在每个节点探索多个可能的"思考"，评估每个思考的质量，保留最有前途的分支，剪掉不好的分支。这就像国际象棋 AI 的搜索——不只是"下一步"，而是"搜索 5 步后的最佳路径"。

---

## 2. 概念

### 2.1 ToT 核心思想

```
CoT:  思考1 → 思考2 → 思考3 → 答案
ToT:  思考1A → 思考2A → 思考3A
      思考1A → 思考2B → 思考3B
      思考1B → 思考2C → 思考3C
      评估每条路径，保留最好的
```

### 2.2 ToT 三要素

| 要素 | 说明 | 类比 |
|------|------|------|
| **扩展** | 从一个节点生成多个候选思考 | 搜索树的分支 |
| **评估** | LLM 评估每个候选的前景 | 节点价值估计 |
| **剪枝** | 只保留最好的 B 个分支 | 剪枝搜索 |

### 2.3 搜索策略

| 策略 | 方法 | 适用场景 |
|------|------|---------|
| **BFS** | 层级遍历 | 所有分支深度相同 |
| **DFS** | 深度优先 | 快速找到一条可行路径 |
| **MCTS** | 蒙特卡洛树搜索 | 复杂空间（LATS） |

### 2.4 LATS——统一框架

LATS（Language Agent Tree Search）将 ToT、ReAct、Reflexion 统一到 MCTS 框架下：

```
选择：MCTS 选择要扩展的节点
扩展：生成多个候选思考
评估：LLM 评估每个候选的前景
回溯：更新父节点的价值估计
行动：选择最佳动作执行
```

---

## 3. 从零实现

### Step 1：思维树节点

```python
class ThoughtNode:
    """思维树节点。"""
    def __init__(self, thought, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0

    def expand(self, child_thoughts):
        """扩展：生成多个候选子思考。"""
        for t in child_thoughts:
            self.children.append(ThoughtNode(t, parent=self))
        return self.children


class ThoughtTree:
    """思维树。"""
    def __init__(self):
        self.root = None

    def build(self, initial_thought):
        self.root = ThoughtNode(initial_thought)
        return self.root

    def select_best_path(self):
        """从根到叶的最佳路径。"""
        node = self.root
        path = [node.thought]
        while node.children:
            node = max(node.children, key=lambda n: n.value)
            path.append(node.thought)
        return path
```

### Step 2：ToT 搜索

```python
def tot_search(problem, llm_fn, evaluator_fn, n_branches=3, depth=3):
    """ToT 搜索——BFS 策略。"""
    tree = ThoughtTree()
    tree.build(f"问题: {problem}")

    current_leaves = [tree.root]

    for d in range(depth):
        all_children = []
        for leaf in current_leaves:
            # LLM 生成候选思考
            candidates = llm_fn(leaf.thought, n_candidates=n_branches)
            children = leaf.expand(candidates)

            # LLM 评估每个候选
            for child in children:
                child.value = evaluator_fn(child.thought)

            all_children.extend(children)

        # 保留最好的 B 个
        current_leaves = sorted(all_children, key=lambda n: -n.value)[:n_branches]

    return tree.select_best_path()
```

---

## 4. 工具

### 4.1 ToT 实现

```python
# 常见实现框架
frameworks = {
    "LangGraph": "支持自定义搜索策略的状态图",
    "LLaMA-Index": "支持 ToT 的推理框架",
    "手写": "完全控制搜索策略",
}
```

### 4.2 性能对比

| 任务 | CoT | ToT | LATS |
|------|-----|-----|------|
| Game of 24 | 4% | 74% | — |
| Creative Writing | 基准 | +15% | +20% |
| Code Generation | 基准 | +10% | +15% |

---

## 5. 工程最佳实践

### 5.1 搜索策略选择

| 任务类型 | 推荐策略 | 原因 |
|---------|---------|------|
| 数学/逻辑 | BFS | 需要探索多个解 |
| 创意写作 | DFS | 快速找到一条好路径 |
| 复杂推理 | MCTS (LATS) | 最优但成本最高 |

### 5.2 踩坑经验

- **分支数太多**：n_branches=5 太多 → 推理时间爆炸。n_branches=2-3 通常够用
- **深度太浅**：depth=2 时可能错过关键推理步骤。depth=3-5 是常用范围
- **评估不准确**：LLM 的自我评估可能有偏差——用多次评估或投票

---

## 6. 常见错误

### 错误 1：ToT 分支数过多导致推理超时

**现象：** 每层扩展 10 个分支 → 深度 3 时需要 1000 次 LLM 调用。

**修复：** n_branches=2-3，depth=3-4 是最佳平衡。

### 错误 2：评估函数不准确

**现象：** LLM 自我评估有偏差——选了错误的分支。

**修复：** 用多个评估提示词或投票——或用验证器（如代码测试）替代 LLM 评估。

---

## 7. 面试考点

### Q1：ToT 和 CoT 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
CoT 是单条链式推理——每步只能往前走，不能回溯。ToT 将推理变成一棵树——在每个节点探索多个候选思考，评估每个候选的前景，保留最好的分支。这就像国际象棋 AI 的搜索——不只是"下一步"，而是"搜索 5 步后的最佳路径"。ToT 在 Game of 24 上从 CoT 的 4% 提升到 74%。

### Q2：LATS 是如何统一 ToT、ReAct、Reflexion 的？（难度：⭐⭐⭐）

**参考答案：**
LATS 将三者纳入 MCTS 框架：(1) 从 ToT 借鉴了分支扩展——每个节点生成多个候选；(2) 从 ReAct 借鉴了行动执行——节点可以调用工具；(3) 从 Reflexion 借鉴了反思记忆——失败后存储经验供后续搜索使用。LATS 的核心：MCTS 的选择、扩展、评估、回溯——同时支持推理和行动。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| ToT | "思维树" | 将推理建模为搜索树——在每个节点探索多个候选思考 |
| LATS | "语言智能体树搜索" | 统一 ToT+ReAct+Reflexion 的 MCTS 框架 |
| BFS | "广度优先搜索" | 层级遍历所有分支——保持探索宽度 |
| DFS | "深度优先搜索" | 深度优先——快速找到一条可行路径 |
| 剪枝 | "砍掉不好的分支" | 只保留最好的 B 个候选——减少搜索空间 |

---

## 📚 小结

ToT 将推理从链式（CoT）变为树式——在每个节点探索、评估、剪枝。Game of 24 从 4% 提升到 74%。LATS 用 MCTS 统一 ToT+ReAct+Reflexion——在 HumanEval 上达到 92.7% pass@1。关键：搜索空间越大，ToT 越有优势。

---

## ✏️ 练习

1. **【实现】** 实现 ToT 的 BFS 搜索——在数学推理任务上测试
2. **【对比】** 对比 ToT 和 CoT 在 Game of 24 上的表现

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| ToT 实现 | `code/main.py` | 思维树 + BFS 搜索 + 评估 |

---

## 📖 参考资料

1. [论文] Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models". NeurIPS, 2023. https://arxiv.org/abs/2305.10601
2. [论文] Zhou et al. "LATS: Language Agent Tree Search Unifies Reasoning Acting and Planning". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
