# 群体优化用于 LLM（PSO、ACO）

> 生物启发优化正在 LLM 中复兴。LMPSO（arXiv:2504.09247）使用 PSO，每个粒子的速度是一个提示词，LLM 生成下一个候选。AMRO-S（arXiv:2603.12933）是 ACO 启发的多智能体路由——4.7x 加速，可解释的路由证据，质量门控异步更新。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 09（并行群体网络）、阶段 16 · 14（共识与 BFT）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 学习目标

完成本课后，你能够：
- 实现 PSO 在提示参数空间上的优化
- 实现 ACO 风格的智能体路由——信息素记录哪些智能体在哪类任务上表现最好
- 理解 PSO 适合连续搜索空间、ACO 适合路由选择
- 识别 LLM 时代的实践限制：评估成本、灾难性漂移、探索-利用权衡

---

## 1. 问题

你有一个提示词在评估上得分 62%。想改进它。梯度无关手动微调扩展性差。强化学习需要大量轮次。通过提示词的反向传播不可行。经典生物启发优化（PSO、ACO）正是为此设计：梯度无关、基于种群、每次评估便宜。将它们与 LLM 配对用于搜索步骤，就得到了实用的优化器。

---

## 2. 概念

### 2.1 PSO 复习（Kennedy & Eberhart 1995）

粒子群优化：连续搜索空间中的粒子种群。每个粒子有位置和速度。每次迭代更新速度（朝向个人最佳和全局最佳）和位置，评估适应度。

``` 
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
```

### 2.2 ACO 复习（Dorigo 1992）

蚂蚁遍历图；每条路径有信息素。蚂蚁移动概率按信息素强度加权。完成任务的蚂蚁按解决方案质量沉积信息素。信息素随时间衰减。

### 2.3 LMPSO

PSO 用于 LLM 生成的结构化输出。每个粒子是候选输出。速度是一个描述如何向最佳方向修改的输出提示词。适用于：输出结构化、适应度自动、种群小（10-30）。

### 2.4 AMRO-S——ACO 用于智能体路由

信息素表记录（任务类型 x 智能体）的强度。每次路由后按质量沉积。质量门控更新：只有通过质量检查的运行才沉积信息素，防止快速但错误的智能体锁定路由。

---

## 3. 从零实现

### 第 1 步：PSO 粒子

```python
@dataclass
class Particle:
    x: list[float]       # 位置
    v: list[float]       # 速度
    p_best: list[float]  # 个人最佳
    p_best_fit: float    # 个人最佳适应度
```

### 第 2 步：LMPSO 循环

```python
def run_lmpso(n_particles=20, iterations=30, seed=0):
    rng = random.Random(seed)
    w, c1, c2 = 0.6, 1.2, 1.2
    bounds = ((0.0, 1.0), (0.0, 1.0))
    particles = []
    for _ in range(n_particles):
        x = [rng.uniform(*bounds[0]), rng.uniform(*bounds[1])]
        v = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)]
        f = fitness(x)
        particles.append(Particle(x=list(x), v=v, p_best=list(x), p_best_fit=f))
    g_best = max(particles, key=lambda p: p.p_best_fit)
    history = [g_best.p_best_fit]
    for _ in range(iterations):
        for p in particles:
            r1, r2 = rng.random(), rng.random()
            for d in range(2):
                p.v[d] = w * p.v[d] + c1 * r1 * (p.p_best[d] - p.x[d]) + c2 * r2 * (g_best.p_best[d] - p.x[d])
                p.x[d] += p.v[d]
                p.x[d] = max(bounds[d][0], min(bounds[d][1], p.x[d]))
            f = fitness(p.x)
            if f > p.p_best_fit:
                p.p_best = list(p.x); p.p_best_fit = f
                if f > g_best.p_best_fit: g_best = p
        history.append(g_best.p_best_fit)
    return history
```

### 第 3 步：ACO 信息素路由器

```python
class PheromoneRouter:
    def __init__(self, task_types, agents, decay=0.05, reinforce=0.2, quality_threshold=0.6):
        self.pheromones = {t: {a: 1.0 for a in agents} for t in task_types}
        self.decay = decay; self.reinforce = reinforce
        self.quality_threshold = quality_threshold

    def choose(self, task_type, rng):
        table = self.pheromones[task_type]
        total = sum(table.values())
        r = rng.random() * total
        upto = 0.0
        for a, p in table.items():
            upto += p
            if r <= upto: return a
        return list(table.keys())[-1]

    def deposit(self, task_type, agent, quality):
        for a in self.pheromones[task_type]:
            self.pheromones[task_type][a] *= (1.0 - self.decay)
        if quality >= self.quality_threshold:
            self.pheromones[task_type][agent] += self.reinforce * quality
```

### 第 4 步：运行演示

```python
def main():
    print("LMPSO — 20 粒子, 30 迭代")
    history = run_lmpso()
    print(f"  final g_best={history[-1]:.4f}")

    print("
AMRO-S — 200 任务, 3 智能体 x 4 任务类型")
    rand_q, aco_q, router = run_amro_s()
    print(f"  random avg: {rand_q:.3f}")
    print(f"  ACO avg:    {aco_q:.3f}")
    print(f"  improvement: {(aco_q-rand_q)/rand_q*100:+.1f}%")
```

完整代码见 code/main.py。

---

## 4. 工业工具

| 方法 | 适用 | 限制 |
|------|------|------|
| PSO | 连续参数搜索、提示优化 | 需要廉价适应度评估 |
| ACO | 路由选择、任务分配 | 路径选择问题 |
| Model Swarms | 专家模型权重优化 | 需要模型参数访问 |

---

## 5. 常见错误

### 错误 1：种群太大

PSO 20 粒子 50 迭代约 $20。评估成本随种群增长线性增加。

### 错误 2：无质量门控的 ACO

快速但错误的智能体积累信息素，系统锁定在错误路由上。质量门控更新：只有通过质量检查的运行才沉积信息素。

### 错误 3：灾难性漂移

适应度函数分布变化时，旧的最佳解和信息素过时。监控最佳适应度稳定性；分布变化时重置或加倍衰减率。

---

## 6. 面试考点

### Q1：PSO 和 ACO 分别适合什么？（难度：⭐）

PSO：连续搜索空间，如提示词参数、LoRA 权重。ACO：路径/路由选择，如任务分配、智能体选择。

### Q2：PSO 为什么适合 LLM？（难度：⭐⭐）

梯度无关、基于种群、每次评估便宜。LLM 输出不是可微参数，无法反向传播。PSO 只需要一个评估函数——如果你能对候选输出打分，就可以优化空间。

---

## 关键术语

| 术语 | 含义 |
|------|------|
| PSO | 粒子群优化，1995，梯度无关 |
| ACO | 蚁群优化，1992，路径选择 |
| LMPSO | PSO 用于 LLM 结构化输出 |
| AMRO-S | ACO 用于多智能体路由 |
| 信息素 | 路由记忆；随时间衰减；按质量沉积 |
| 质量门控 | 只有好运行才更新信息素 |

---

## 小结

PSO 和 ACO 是经典的梯度无关优化算法，适合 LLM 的结构化输出搜索和智能体路由。PSO 需要廉价适应度评估，ACO 需要可重复的路由选择。质量门控防止快速但错误的智能体锁定路由。

---

## 练习

1. 运行 code/main.py。观察 LMPSO 收敛。改变种群大小 5、10、20、50。何时收敛时间饱和？
2. 实现灾难性漂移实验：第 30 迭代后改变适应度函数。PSO 多快适应？
3. 给 AMRO-S 添加质量门控：只有评估分数 > 0.7 时才沉积信息素。这与无门控版本相比收敛如何？

---

## 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| PSO+ACO 演示 | code/main.py | LMPSO 优化 + AMRO-S 路由 |

---

## 参考资料

1. Kennedy & Eberhart. Particle Swarm Optimization. 1995.
2. Dorigo. Ant Colony Optimization. 1992.
3. LMPSO. arXiv:2504.09247
4. AMRO-S. arXiv:2603.12933
