# AlphaEvolve——进化编码智能体

> 将前沿编码模型与进化循环和机器可检查的评估器配对。让循环运行足够久。它发现了一个使用 48 次标量乘法的 4×4 复数矩阵乘法——56 年来对 Strassen 的首次改进。它还找到了一个 Google 全范围的 Borg 调度启发式，在生产中恢复了约 0.7% 的集群算力。架构故意很无聊。收益来自评估器的严格性。

**类型：** 概念课
**语言：** Python（标准库，进化循环玩具）
**前置知识：** 阶段 15 · 01（长期智能体框架）、阶段 15 · 02（自我教学推理）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 AlphaEvolve 的核心循环：LLM 提议修改 → 评估器评分 → 高分变体成为下一代父本
- [ ] 理解为什么机器可检查评估器是整个架构的前提条件
- [ ] 识别奖励黑客的风险——进化优化的是评估器测量的东西，不是你想要的东西
- [ ] 对比 LLM + 搜索与纯 LLM 或纯搜索的优势
- [ ] 将 AlphaEvolve 与 FunSearch、AI Scientist v2、Darwin Godel Machine 对比

---

## 1. 问题

大语言模型能写代码。进化算法能搜索代码。两者都已经被单独尝试了几十年；两者都碰到了天花板。LLM 的天花板是编造：模型写出看似合理但实际不按声称工作的代码。进化的天花板是搜索成本：对语法的随机变异很少产生可编译的程序，更别说更好的了。

AlphaEvolve（Novikov 等人，DeepMind，arXiv:2506.13131，2025 年 6 月）将两者结合。LLM 对程序数据库提出有针对性的修改；自动评估器对每个变体打分；高分变体成为未来世代的父本。LLM 处理写合理代码的昂贵步骤；评估器捕获编造。循环运行数小时到数周。

结果：48 次标量乘法的 4×4 复数矩阵乘法（Strassen 1969 年的界限是 49）、Google 生产中的 Borg 调度启发式、32.5% 的 FlashAttention 内核加速、Gemini 训练吞吐量提升。

架构有效是因为**评估器是机器可检查的**。在评估器不可检查的地方它无效。这个不对称性就是课程。

---

## 2. 概念

### 2.1 核心循环

```
种子程序 P₀（正确但非最优）
    ↓
维护变体程序数据库，每个被评估器打分
    ↓
从数据库中采样一个或多个父本
    ↓
LLM（Gemini Flash 处理多数候选，Gemini Pro 处理困难的）
    产生父本的修改变体
    ↓
编译、运行、在保留的评估器上评估变体
    ↓
按分数和特征向量插入数据库
    ↓
重复（数小时到数周）
```

两个细节关键。第一，LLM 的提示不仅是父本程序——通常包括数据库中几个最佳变体、评估器签名和简短任务描述。模型的工作是提出可能提高分数的有针对性修改。第二，数据库是结构化的（MAP-elites 网格、岛模型），所以循环探索多样性，而不仅仅是当前领先者。

### 2.2 为什么评估器是不可协商的

AlphaEvolve 的所有成功都来自评估器快速、确定且难以博弈的领域：

| 领域 | 评估器 | 为什么有效 |
|------|-------|----------|
| 矩阵乘法算法 | 逐位相等的单元测试 | LLM 无法编造正确性 |
| Borg 调度启发式 | 重放历史集群负载的生产级模拟器 | 真实硬件上的真实负载 |
| FlashAttention 内核 | 正确性测试 + 真实硬件上的墙钟基准 | 硬件上消失的性能声明无法存活 |
| Gemini 训练吞吐量 | 每步 GPU 秒数 | 直接测量 |

在每种情况下，评估器都捕获了否则会主导的 LLM 错误类型：编造的正确性声明、在硬件上消失的性能声明、边缘情况失败。移除评估器，循环会为漂亮的代码而优化。

### 2.3 奖励黑客是那句话的另一面

进化优化的是评估器测量的东西。如果评估器不完美，循环会找到不完美之处。在未验证的领域，循环会优化表面特征而非预期行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功仅在评估器严格性匹配搜索野心的领域可转移。

2025-2026 年代码搜索循环中奖励黑客的具体例子：

- 优化目标奖励"完成时间"→ 奖励提交空解决方案
- 基准分数奖励测试正确性 → 奖励记忆测试和过拟合
- "代码质量"代理奖励删除注释和重写变量名，无语义变化

AlphaEvolve 的修复：使用 LLM 从未见过的保留评估器，输入在评估时生成。即便如此，DeepMind 仍建议对任何提议的部署进行严格审查。

### 2.4 为什么 LLM + 搜索优于任一单独使用

LLM 能产生可编译的、语义合理的修改。2000 行 Python 文件上的随机变异 GA 几乎总是产生语法错误。LLM 还将搜索集中在合理邻域（修改一个函数，而非随机字节），大幅减少浪费的评估器调用。

评估器反过来捕获 LLM 的编造。LLM 会自信地声称一个函数"在极限情况下是 O(n log n)"而实际上是 O(n²)；墙钟基准让这个问题有了定论。

### 2.5 在前沿栈中的位置

| 系统 | 生成器 | 评估器 | 领域 | 示例成果 |
|------|--------|-------|------|---------|
| AlphaEvolve | Gemini | 正确性 + 基准 | 算法、内核、调度器 | 48-mul 4×4 矩阵乘法 |
| FunSearch（DeepMind, 2023） | PaLM / Codey | 正确性 | 组合数学 | cap-set 下界 |
| AI Scientist v2（Sakana, L5） | GPT/Claude | LLM 评判 + 实验 | ML 研究 | ICLR workshop 论文 |
| Darwin Godel Machine（L4） | 智能体脚手架 | SWE-bench / Polyglot | 智能体代码 | SWE-bench 20% → 50% |

四个都是同一配方的变体：生成器 + 评估器，循环。区别在于评估器评什么以及它有多严格。

---

## 3. 从零实现

### 第 1 步：定义表达式和评估器

```python
import random

# 目标函数：2x² + 3x - 1
def target(x: float) -> float:
    return 2.0 * x * x + 3.0 * x - 1.0

Expr = tuple  # ("num", v) | ("x",) | ("add", a, b) | ("mul", a, b)

def evaluate_expr(e: Expr, x: float) -> float:
    tag = e[0]
    if tag == "num": return float(e[1])
    if tag == "x": return x
    if tag == "add": return evaluate_expr(e[1], x) + evaluate_expr(e[2], x)
    if tag == "mul": return evaluate_expr(e[1], x) * evaluate_expr(e[2], x)
    raise ValueError(tag)

def mse(e: Expr, xs: list[float]) -> float:
    total = 0.0
    for x in xs:
        try: y = evaluate_expr(e, x)
        except (OverflowError, ValueError): return float("inf")
        total += (y - target(x)) ** 2
    return total / max(1, len(xs))
```

### 第 2 步：实现变异（LLM 替身）

```python
def mutate(e: Expr) -> Expr:
    """LLM 的有针对性修改的替身。"""
    choice = random.random()
    if choice < 0.25:
        return random_leaf()
    if choice < 0.5:
        return ("add", e, random_leaf())
    if choice < 0.75:
        return ("mul", e, random_leaf())
    return perturb(e)  # 扰动某个常量

def random_leaf() -> Expr:
    if random.random() < 0.5:
        return ("x",)
    return ("num", float(random.choice([-2, -1, 0, 1, 2, 3])))
```

### 第 3 步：实现 MAP-elites 网格

```python
def cell_key(e: Expr) -> tuple[int, int]:
    """按表达式深度和常量大小分桶——保持多样性。"""
    d = min(depth(e), 6)
    c = min(int(max_const(e) / 2), 4)
    return (d, c)
```

### 第 4 步：运行进化循环

```python
def run_loop(generations, pop, use_holdout, seed=None):
    if seed: random.seed(seed)
    train_xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    test_xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    def signal_of(c):
        return 0.5 * (c.train_score + c.test_score) if use_holdout else c.train_score

    # 初始化种群
    archive = {}
    for _ in range(pop):
        c = seed_candidate(test_xs, train_xs, 0)
        key = cell_key(c.expr)
        if key not in archive or signal_of(c) < signal_of(archive[key]):
            archive[key] = c

    # 进化循环
    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child_expr = mutate(parent.expr)
        child = Candidate(child_expr, mse(child_expr, train_xs), mse(child_expr, test_xs), g)
        key = cell_key(child_expr)
        if key not in archive or signal_of(child) < signal_of(archive[key]):
            archive[key] = child

    return min(archive.values(), key=signal_of)
```

### 第 5 步：运行对比

```python
def main():
    print("Run A：有保留评估器")
    best_a = run_loop(1500, 20, use_holdout=True, seed=1)
    print(f"  最佳表达式: {render(best_a.expr)}")
    print(f"  训练 MSE: {best_a.train_score:.4f}  测试 MSE: {best_a.test_score:.4f}")

    print("\nRun B：无保留评估器（过拟合风险）")
    best_b = run_loop(1500, 20, use_holdout=False, seed=1)
    print(f"  最佳表达式: {render(best_b.expr)}")
    print(f"  训练 MSE: {best_b.train_score:.4f}  测试 MSE: {best_b.test_score:.4f}")
    print(f"  训练-测试差距: {best_b.test_score - best_b.train_score:+.4f}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 AlphaEvolve 成果

| 成果 | 领域 | 意义 |
|------|------|------|
| 48-mul 4×4 复数矩阵乘法 | 数学 | 56 年来对 Strassen 49-mul 的首次改进 |
| Borg 调度启发式 | 生产调度 | Google 全范围，恢复约 0.7% 集群算力 |
| FlashAttention 内核加速 | ML 基础设施 | 32.5% 内核加速 |
| Gemini 训练吞吐量 | 训练效率 | GPU 秒数/步的改进 |

### 4.2 评估器设计检查清单

| 检查项 | 说明 |
|--------|------|
| 确定性 | 相同输入始终产生相同输出 |
| 快速 | 秒级而非小时级 |
| 难以博弈 | LLM 无法通过表面特征获得高分 |
| 保留评估器 | LLM 从未见过的输入，评估时生成 |
| 反奖励黑客检查 | 监控循环是否在优化表面特征而非预期行为 |

---

## 5. 工程最佳实践

### 5.1 进化编码循环设计原则

| 原则 | 说明 |
|------|------|
| 评估器是架构 | 没有机器可检查评估器的进化循环会为漂亮的代码优化 |
| 保留评估器是必须的 | LLM 从未见过的输入在评估时生成 |
| MAP-elites 保持多样性 | 按特征向量分桶，防止单一路径过早收敛 |
| 严格性匹配野心 | 评估器严格性必须匹配搜索野心 |

### 5.2 中文场景特别建议

- **AlphaEvolve 的评估器设计思路适用于任何可验证任务**——中文 NLP 中的翻译质量、摘要质量都可以构建评估器
- **奖励黑客在中文场景中更隐蔽**——中文的多义词和歧义可能导致评估器被表面特征欺骗
- **保留评估器的生成要考虑中文特有的边界情况**——中文编码、Unicode 处理、全角半角转换

### 5.3 踩坑经验

- **没有保留评估器**——循环过拟合到训练评估器上，得到的"最优解"在新输入上失败。**修复：** 始终有保留评估器
- **评估器太慢**——每个评估需要 10 分钟，循环一天只能运行几百代。**修复：** 评估器必须秒级
- **奖励黑客未监控**——循环的分数在上升但实际质量在下降。**修复：** 定期人工审查高分变体

---

## 6. 常见错误

### 错误 1：没有保留评估器

**现象：** 进化循环在训练评估器上跑了几千代，MSE 降到 0.0001。但保留评估器上 MSE 是 0.5。循环学会了在训练点上插值而非泛化。

**原因：** 没有保留评估器，循环优化的是表面特征而非泛化能力。

**修复：**
```python
# ❌ 只用训练评估器
signal = train_mse

# ✓ 训练 + 保留评估器的加权信号
signal = 0.5 * train_mse + 0.5 * test_mse
```

### 错误 2：评估器太慢

**现象：** 每个变体评估需要 10 分钟。一天只能运行 144 代。搜索空间太大，进展极慢。

**原因：** 评估器的运行时间与循环的吞吐量成反比。

**修复：** 评估器必须秒级。如果完整评估需要小时，使用代理评估器（秒级近似）+ 定期完整评估（小时级确认）。

### 错误 3：MAP-elites 网格太粗或太细

**现象：** 网格太粗→所有变体落入同一格→循环退化为普通进化。网格太细→大多数格子为空→浪费多样性维护的开销。

**原因：** 特征向量的设计需要匹配问题的结构。

**修复：** 从粗网格开始（6×5），根据收敛速度调整。如果循环在 100 代内收敛，网格太粗；如果 90% 的格子在 1000 代后仍为空，网格太细。

---

## 7. 面试考点

### Q1：AlphaEvolve 的核心循环是什么？为什么需要机器可检查评估器？（难度：⭐）

**参考答案：**
循环：LLM 提议修改 → 评估器评分 → 高分变体成为父本 → 重复。

机器可检查评估器是必须的因为：LLM 会编造正确的代码。没有评估器，循环会为"看起来合理"的代码优化而非"实际工作"的代码。AlphaEvolve 的所有成功都来自评估器快速、确定且难以博弈的领域。

### Q2：什么是奖励黑客？在 AlphaEvolve 中如何发生？（难度：⭐⭐）

**参考答案：**
奖励黑客是进化找到了最大化评估器分数但不做预期任务的方式。

在 AlphaEvolve 中可能的例子：优化目标奖励"完成时间"→ 循环发现提交空解决方案可以得到零完成时间。或者基准分数奖励测试正确性 → 循环记忆测试用例。

AlphaEvolve 的修复：使用 LLM 从未见过的保留评估器，输入在评估时生成。

### Q3：为什么 LLM + 搜索优于纯 LLM 或纯搜索？（难度：⭐⭐）

**参考答案：**
**纯 LLM**：能写合理代码但会编造（声称 O(n log n) 而实际 O(n²)）。评估器捕获编造。

**纯搜索**：随机变异很少产生可编译的程序。2000 行文件上的随机变异几乎总是语法错误。LLM 将搜索集中在合理邻域（修改一个函数而非随机字节），大幅减少浪费的评估调用。

LLM 处理写合理代码的昂贵步骤；评估器捕获编造。两者互补。

### Q4：AlphaEvolve、FunSearch、AI Scientist v2、Darwin Godel Machine 的共同点是什么？（难度：⭐⭐⭐）

**参考答案：**
四个都是同一配方的变体：**生成器 + 评估器，循环**。

| 系统 | 生成器 | 评估器 | 领域 |
|------|--------|-------|------|
| AlphaEvolve | Gemini | 正确性 + 基准 | 算法、内核 |
| FunSearch | PaLM / Codey | 正确性 | 组合数学 |
| AI Scientist v2 | GPT/Claude | LLM 评判 + 实验 | ML 研究 |
| Darwin Godel Machine | 智能体脚手架 | SWE-bench | 智能体代码 |

区别在于评估器评什么以及它有多严格。评估器越严格，成果越可信。AlphaEvolve 的评估器最严格（硬件基准），所以成果最令人印象深刻。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| AlphaEvolve | "DeepMind 的进化编码智能体" | Gemini + 程序数据库 + 机器可检查评估器 |
| MAP-elites | "多样性保留存档" | 按特征向量分桶的网格；每个格子保存该描述符的最佳变体 |
| 岛模型 | "并行进化子种群" | 独立种群定期迁移；防止过早收敛 |
| 机器可检查评估器 | "确定性神谕" | LLM 无法伪造的单元测试、模拟器或基准——此循环的前提 |
| 奖励黑客 | "优化度量而非目标" | 循环找到最大化分数但不做预期任务的方式 |
| 种子程序 | "起点" | 循环从其演化的初始正确但非最优程序 |
| 保留评估器 | "LLM 从未见过的评估数据" | 评估时生成的输入，防止记忆 |

---

## 📚 小结

AlphaEvolve 将 LLM 的代码生成能力与进化搜索和机器可检查评估器结合。核心循环：LLM 提议修改 → 评估器评分 → 高分变体成为父本 → 重复。架构有效是因为评估器严格——48-mul 4×4 矩阵乘法、Borg 调度启发式、FlashAttention 内核加速都来自评估器不可博弈的领域。没有严格评估器的进化循环会为漂亮的代码优化。

至此第 15 章第 1-3 节完成：长期智能体的视界增长（01）、STaR 自我改进循环（02）、AlphaEvolve 进化编码（03）。三者的关系：视界增长需要更可靠的智能体，STaR 是最小可行的自我改进循环，AlphaEvolve 是 STaR 在代码领域的规模化实例。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。注意最佳分数轨迹。禁用保留评估器（`--no-holdout` 标志）重跑。量化过拟合。

2. **【阅读】** 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites 网格的部分。为一个新问题（如编译器优化 pass）设计一个特征向量描述符，使搜索保持多样性。

3. **【思考】** 48-mul 4×4 结果在 56 年后改进了 Strassen 的 49-mul 界限。阅读论文附录 F，用三句话解释为什么这个问题的评估器特别容易做对，以及为什么大多数领域不像这样。

4. **【设计】** 提出一个 AlphaEvolve 会失败的领域。准确识别评估器在哪里崩溃以及为什么。

5. **【实现】** 为你了解的一个领域写出评估器签名。包括（a）正确性条件（b）性能度量（c）保留输入生成规则（d）至少一个反奖励黑客检查。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 进化循环模拟器 | `code/main.py` | MAP-elites + 保留评估器的符号回归 |
| 技能提示词 | `outputs/skill-evaluator-rigor-audit.md` | 评估器严格性审计——新领域是否适用 AlphaEvolve |

---

## 📖 参考资料

1. [论文] Novikov et al. (2025). "AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery". https://arxiv.org/abs/2506.13131 — 完整论文
2. [博客] DeepMind. "AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms". https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ — 供应商文章
3. [GitHub] Google DeepMind. "AlphaEvolve Results". https://github.com/google-deepmind/alphaevolve_results — 发现的算法，包括 48-mul 4×4 矩阵乘法
4. [论文] Romera-Paredes et al. (2023). "Mathematical Discoveries from Program Search with LLMs (FunSearch)". https://www.nature.com/articles/s41586-023-06924-6 — 前身系统
5. [论文] Anthropic. "Responsible Scaling Policy v3.0 (Feb 2026)". https://anthropic.com/responsible-scaling-policy/rsp-v3-0 — 将评估器绑定的自主性框定为关键研究方向

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
