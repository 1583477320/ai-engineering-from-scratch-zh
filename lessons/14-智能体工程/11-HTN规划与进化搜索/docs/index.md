# HTN 规划与进化搜索

> 符号规划处理计划可被证明正确的场景。进化代码搜索处理适应度函数可机器验证的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了当这些与 LLM 配对时各自能解锁什么。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 02（ReWOO）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释分层任务网络（HTN）：任务、方法、算子、前提、效果
- [ ] 实现 HTN 规划——将复杂任务分解为子任务
- [ ] 理解进化搜索如何结合 LLM 生成代码变异
- [ ] 对比符号规划和进化搜索的适用场景

---

## 1. 问题

ReWOO 让 LLM 生成计划——但计划的正确性没有保证。对于需要证明正确的任务（如逻辑推理），需要**符号规划**。对于需要搜索最优解的任务（如代码优化），需要**进化搜索**。

两种方法各有适用场景——LLM + 符号规划/进化搜索是 2026 年的前沿组合。

---

## 2. 概念

### 2.1 HTN（分层任务网络）

| 概念 | 说明 |
|------|------|
| **任务** | 需要完成的目标 |
| **方法** | 将任务分解为子任务的策略 |
| **算子** | 可执行的原子操作 |
| **前提** | 执行算子前必须满足的条件 |
| **效果** | 执行算子后状态的变化 |

### 2.2 HTN 规划过程

```
任务: "准备晚餐"
    ↓ HTN 分解
子任务: ["买菜", "洗菜", "切菜", "烹饪", "装盘"]
    ↓ 每个子任务进一步分解
算子: ["去超市买蔬菜", "用水清洗蔬菜", ...]
    ↓ 验证前提
确认每步前提满足 → 执行
```

### 2.3 进化代码搜索

```
初始种群: [代码变体1, 代码变体2, ..., 代码变体N]
    ↓ 适应度评估（运行测试）
选择: 保留适应度高的变体
    ↓ 交叉/变异
新种群: [更好的代码变体1, ..., 更好的代码变体N]
    ↓ 重复
直到找到最优代码
```

### 2.4 ChatHTN 和 AlphaEvolve

| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| **ChatHTN** | LLM 生成 HTN 分解 | 需要证明正确的推理 |
| **AlphaEvolve** | LLM + 进化搜索 | 可验证的代码/数学优化 |

---

## 3. 从零实现

### Step 1：HTN 规划器

```python
class HTNPlanner:
    """分层任务网络规划器。"""
    def __init__(self):
        self.methods = {}
        self.operators = {}

    def add_method(self, task, subtasks):
        self.methods[task] = subtasks

    def add_operator(self, name, preconditions, effects):
        self.operators[name] = {"preconditions": preconditions, "effects": effects}

    def plan(self, task):
        """递归分解任务。"""
        if task in self.operators:
            return [{"type": "operator", "name": task,
                     "preconditions": self.operators[task]["preconditions"],
                     "effects": self.operators[task]["effects"]}]

        if task in self.methods:
            plan = []
            for subtask in self.methods[task]:
                plan.extend(self.plan(subtask))
            return plan

        return [{"type": "task", "name": task, "status": "unresolved"}]

    def verify(self, plan):
        """验证计划的前提是否满足。"""
        state = set()
        for step in plan:
            if step["type"] == "operator":
                if not all(p in state for p in step["preconditions"]):
                    return False, f"前提未满足: {step['preconditions']}"
                state.update(step["effects"])
        return True, "计划验证通过"


if __name__ == "__main__":
    print("HTN 规划 + 进化搜索演示\n")

    # HTN
    print("HTN 规划:")
    planner = HTNPlanner()
    planner.add_method("准备晚餐", ["购买食材", "烹饪", "装盘"])
    planner.add_method("购买食材", ["列出清单", "去超市"])
    planner.add_operator("列出清单", [], {"清单已准备好"})
    planner.add_operator("去超市", ["清单已准备好"], {"食材已购买"})
    planner.add_operator("烹饪", ["食材已购买"], {"菜肴已做好"})
    planner.add_operator("装盘", ["菜肴已做好"], {"晚餐准备完成"})

    plan = planner.plan("准备晚餐")
    print(f"  计划步骤: {len(plan)}")
    for step in plan:
        print(f"    {step['type']}: {step['name']}")

    valid, msg = planner.verify(plan)
    print(f"  验证: {msg}")

    # 进化搜索
    print("\n进化搜索:")
    population = [f"def solve(): return {i}" for i in range(5)]
    for gen in range(3):
        scores = [len(code) for code in population]
        best_idx = scores.index(max(scores))
        new_pop = [population[best_idx] for _ in range(5)]
        population = new_pop
        print(f"  第{gen+1}代: 最佳长度={max(scores)}")

---

## 4. 工具

### 4.1 HTN 规划器

```python
# HTN 规划器：任务分解为子任务和算子
```

### 4.2 框架对比

| 框架 | 特点 |
|------|------|
| ChatHTN | LLM + HTN 分解 |
| AlphaEvolve | LLM + 进化搜索 |

---

## 5. 工程最佳实践

### 5.1 HTN 设计原则

- **任务分解**：将复杂任务递归分解为子任务
- **前提验证**：确保每个算子的前提条件满足
- **效果跟踪**：每个算子执行后更新状态

### 5.2 进化搜索设计

- **适应度函数**：可机器验证（如单元测试通过）
- **变异策略**：LLM 生成变异候选
- **种群多样性**：避免过早收敛

---

## 6. 常见错误

### 错误 1：HTN 前提不满足

**现象：** 计划执行失败。

**修复：** 在规划前验证所有前提条件。

### 错误 2：进化搜索过早收敛

**现象：** 所有种群相同。

**修复：** 增加变异率或引入新随机个体。

---

## 7. 面试考点

### Q1：HTN 规划和传统规划有什么区别？（难度：⭐⭐）

**参考答案：** HTN 将任务递归分解为子任务——比传统的全状态空间搜索更高效。传统规划需要枚举所有状态，HTN 只在任务分解的子空间搜索。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| HTN | "分层任务网络" | 将任务递归分解为子任务和算子 |
| 进化搜索 | "自然选择搜索" | 用变异+选择迭代优化——适合可验证的任务 |

---

## 📚 小结

HTN 通过任务分解处理可证明正确的规划。进化搜索通过变异+选择优化可验证的适应度函数。两者与 LLM 结合——ChatHTN 和 AlphaEvolve 是 2025-2026 年的前沿。

---

## ✏️ 练习

1. **【实现】** 用 HTN 分解"写一篇论文"为子任务
2. **【实验】** 用进化搜索优化 Python 函数的性能

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| HTN 规划器 | `code/main.py` | 分层任务分解+前提验证 |

---

## 📖 参考资料

1. [论文] ChatHTN: https://arxiv.org/abs/2502.xxxxx
2. [论文] AlphaEvolve: https://arxiv.org/abs/2502.xxxxx

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
