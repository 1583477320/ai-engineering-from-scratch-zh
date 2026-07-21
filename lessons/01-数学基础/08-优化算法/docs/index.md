# 优化算法——训练神经网络就是找山谷底部

> 训练神经网络不过是找到山谷的底部。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 04-05
**预计时间：** 75 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 08（优化器详解）— 本节的优化器是该阶段的核心工具

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 vanilla 梯度下降、带动量的 SGD 和 Adam
- [ ] 在 Rosenbrock 函数上对比优化器收敛性，解释 Adam 为什么自适应每权重学习率
- [ ] 区分凸与非凸损失曲面，解释鞍点在高维中的作用
- [ ] 配置学习率调度（阶梯衰减、余弦退火、预热）确保训练稳定

---

## 1. 问题

你有损失函数。它告诉你模型有多错。你有梯度。它们告诉哪个方向会让损失变大。现在需要一个策略走下坡。

朴素方法：沿梯度反方向走，用学习率缩放步长，重复。这就是梯度下降。但"能用"有前提：学习率太大就跳过山谷；太小则爬几千步才到；碰到鞍点就停了。

每个深度学习优化器都在回答同一个问题：如何更快更可靠地到达谷底。

---

## 2. 核心概念

### 2.1 三种变体

| 变体 | 批次大小 | 梯度质量 | 速度 | 噪声 |
|:-----|:---------|:---------|:-----|:-----|
| 批量 GD | 全部数据 | 精确 | 慢 | 无 |
| SGD | 1 个样本 | 噪声大 | 快 | 高 |
| Mini-batch | 32-256 | 良好估计 | 平衡 | 适中 |

### 2.2 动量——球滚下坡

```
v = β × v + 梯度
w = w - lr × v
```

β=0.9（典型值）。动量加速一致方向、抑制振荡。

### 2.3 Adam——自适应学习率

每权重跟踪两个量：

```
m = β₁ × m + (1-β₁) × g          # 梯度一阶矩（动量）
v = β₂ × v + (1-β₂) × g²         # 梯度二阶矩（幅度）
m̂ = m / (1 - β₁ᵗ)                # 偏差修正
v̂ = v / (1 - β₂ᵗ)                # 偏差修正
w = w - lr × m̂ / (√v̂ + ε)       # 更新
```

默认参数：lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8。

### 2.4 学习率调度

| 调度策略 | 适用场景 |
|:---------|:---------|
| 阶梯衰减 | 简单，手动控制 |
| 余弦退火 | Transformer，现代训练 |
| 预热+衰减 | 大模型，防止早期不稳定 |

### 2.5 凸 vs 非凸

凸函数只有一个最小值。神经网络损失是非凸的——有局部最小值、鞍点。实践中，高维网络的局部最小值损失接近全局最小值。鞍点才是真正的障碍。

---

## 3. 从零实现

```python
"""从零实现优化器——GD、SGD+Momentum、Adam。"""

def rosenbrock(params):
    x, y = params
    return (1-x)**2 + 100*(y-x**2)**2

def rosenbrock_grad(params):
    x, y = params
    return [-2*(1-x)+200*(y-x**2)*(-2*x), 200*(y-x**2)]


class GradientDescent:
    def __init__(self, lr=0.001): self.lr = lr
    def step(self, params, grads):
        return [p - self.lr*g for p, g in zip(params, grads)]


class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr; self.beta = momentum; self.v = None
    def step(self, params, grads):
        if self.v is None: self.v = [0.0]*len(params)
        self.v = [self.beta*v + g for v, g in zip(self.v, grads)]
        return [p - self.lr*v for p, v in zip(params, self.v)]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m=None; self.v=None; self.t=0
    def step(self, params, grads):
        if self.m is None:
            self.m=[0.0]*len(params); self.v=[0.0]*len(params)
        self.t += 1
        self.m=[self.b1*m+(1-self.b1)*g for m,g in zip(self.m,grads)]
        self.v=[self.b2*v+(1-self.b2)*g**2 for v,g in zip(self.v,grads)]
        mh=[m/(1-self.b1**self.t) for m in self.m]
        vh=[v/(1-self.b2**self.t) for v in self.v]
        return [p-self.lr*mh/(vh**0.5+self.eps) for p,mh,vh in zip(params,mh,vh)]


def optimize(opt, func, grad, start, steps=5000):
    p = list(start); history = [p[:]]
    for _ in range(steps):
        p = opt.step(p, grad(p)); history.append(p[:])
    return history


def main():
    start = [-1.0, 1.0]
    for name, opt in [("GD", GradientDescent(0.0005)),
                      ("SGD+M", SGDMomentum(0.0001, 0.9)),
                      ("Adam", Adam(0.01))]:
        h = optimize(opt, rosenbrock, rosenbrock_grad, start)
        loss = rosenbrock(h[-1])
        print(f"{name:6s} → x={h[-1][0]:.6f}, y={h[-1][1]:.6f}, loss={loss:.8f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| PyTorch `optim.Adam` | 生产默认优化器 |
| PyTorch `optim.SGD` | 需要最佳最终精度时 |
| PyTorch `optim.AdamW` | Transformer 训练 |
| PyTorch `lr_scheduler.CosineAnnealingLR` | 学习率调度 |

经验法则：Adam(lr=0.001) → SGD(lr=0.01, mom=0.9) → AdamW(lr=0.001, wd=0.01)

---

## 5. 知识连线

- **第 03 阶段 · 08（优化器详解）**：本节的优化器在 PyTorch 中的完整实现
- **第 10 阶段（LLM 从零）**：AdamW + 余弦调度是 GPT 训练的标准配置

---

## 6. 工程最佳实践

- **从 Adam 开始**：对大多数问题无需调参即可工作
- **长期训练用学习率调度**：余弦退火+预热是现代标准
- **中文场景建议**：大规模分布式训练时，学习率需要按设备数线性缩放

---

## 7. 常见错误

- **学习率太大**：损失震荡或发散。从 0.001 开始扫描。
- **学习率太小**：收敛缓慢。观察损失曲线判断。

---

## 8. 面试考点

### Q1：为什么 Adam 适合大多数任务？（难度：⭐⭐）

**参考答案：** Adam 为每个权重维护自适应学习率——频繁接收大梯度的权重步长自动减小，很少变化的权重步长自动增大。默认超参在大多数问题上工作良好，无需手动调参。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 梯度下降 | 减去学习率×梯度来最小化损失 |
| 动量 | 累积历史梯度的向量，加速收敛、抑制振荡 |
| Adam | 自适应矩估计——为每权重维护一阶矩和二阶矩 |
| 学习率调度 | 训练过程中动态调整学习率的策略 |
| 鞍点 | 梯度为零但非最小值的点 |

---

## 📚 小结

优化器决定了训练的效率和稳定性。你实现了 GD、SGD+Momentum 和 Adam，对比了它们在 Rosenbrock 函数上的表现，理解了凸与非凸损失曲面。下一课学习信息论。

---

## ✏️ 练习

1. 【实验】用不同学习率运行 GD，找到收敛的最大学习率
2. 【实现】添加动量到 GD，对比有无动量的收敛速度
3. 【实现】在鞍点函数上测试三种优化器，哪个最先逃离

---

## 📖 参考资料

1. [博客] Sebastian Ruder 梯度下降综述. https://ruder.io/optimizing-gradient-descent/
2. [论文] Kingma & Ba. "Adam: A Method for Stochastic Optimization". https://arxiv.org/abs/1412.6980
