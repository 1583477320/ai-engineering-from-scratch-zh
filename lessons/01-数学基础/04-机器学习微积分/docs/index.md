# 机器学习微积分——导数告诉你下坡方向

> 导数告诉你哪个方向是下坡的。这正是神经网络学习所需要的全部。

**类型：** 概念课
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-03
**预计时间：** 60 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 05（反向传播）— 本节的链式法则是反向传播的数学基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 计算常见 ML 函数（x²、sigmoid、交叉熵）的数值和解析导数
- [ ] 从零实现梯度下降，在 1D 和 2D 上最小化损失函数
- [ ] 推导线性回归模型的梯度并通过手动权重更新训练
- [ ] 解释 Hessian 矩阵、泰勒级数近似及其与优化方法的联系

---

## 1. 问题

你有一个包含百万参数的神经网络。每个参数是一个旋钮。你需要知道每个旋钮应该向哪个方向转动才能让模型稍微不那么错误。微积分给出这个方向。

没有微积分，训练神经网络意味着随机改变然后祈祷。有了导数，你精确知道每个权重如何影响误差。你每次都正确地转动每个旋钮。

---

## 2. 核心概念

### 2.1 什么是导数

导数衡量变化率。对于函数 y = f(x)，f'(x) 告诉你：如果 x 微小变化，y 变化多少？

### 2.2 梯度：所有偏导数的向量

```
grad f = [df/dx, df/dy, df/dz]
```

梯度指向最陡上升方向。要最小化函数，走反方向——这就是梯度下降。

### 2.3 梯度下降更新规则

```
w_new = w_old - 学习率 × dL/dw
```

对每个权重：计算偏导数 → 乘以小学习率 → 从权重中减去。重复。

### 2.4 链式法则

```
y = f(g(x))  →  dy/dx = f'(g(x)) × g'(x)
```

神经网络就是函数的链：输入→线性→激活→线性→损失。反向传播就是链式法则的重复应用。

### 2.5 Hessian 矩阵与优化

| 方法 | 使用 | 计算成本 | 收敛速度 |
|:-----|:-----|:---------|:---------|
| 梯度下降 | 一阶导数 | O(N)/步 | 慢（线性） |
| 牛顿法 | 完整 Hessian | O(N³)/步 | 快（二次） |
| Adam | 每参数自适应率 | O(N)/步 | 中（超线性） |

---

## 3. 从零实现

```python
"""机器学习微积分——数值导数、梯度下降、Hessian。"""
import math

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, point, h=1e-7):
    return [(f([point[j] + (h if j == i else 0) for j in range(len(point))])
             - f([point[j] - (h if j == i else 0) for j in range(len(point))]) / (2 * h))
            for i in range(len(point))]

def gradient_descent_2d(f, start, lr=0.1, steps=30):
    point = list(start)
    for step in range(steps):
        grads = [(f([p + (1e-7 if j == i else 0) for j in range(len(point))])
                  - f([p - (1e-7 if j == i else 0) for j in range(len(point))])) / 2e-7
                 for i, p in enumerate(point)]
        point = [p - lr * g for p, g in zip(point, grads)]
    return point

def main():
    print("=== 数值导数 vs 解析导数 ===")
    f = lambda x: x**2
    for x in [-2, -1, 0, 1, 2]:
        print(f"  x={x:2d}  数值={numerical_derivative(f,x):.6f}  解析={2*x}")

    print("\n=== 2D 梯度下降: x² + y² ===")
    pt = [4.0, 3.0]
    for step in range(0, 31, 5):
        grads = [(f([pt[j]+(1e-7 if j==i else 0) for j in range(2)])
                  -f([pt[j]-(1e-7 if j==i else 0) for j in range(2)]))/2e-7
                 for i in range(2)]
        pt = [p - 0.1*g for p, g in zip(pt, grads)]
        print(f"  步{step:2d}: ({pt[0]:.4f}, {pt[1]:.4f})  f={pt[0]**2+pt[1]**2:.6f}")

    print("\n=== 线性回归: y = 2x + 1 ===")
    w, b, lr = 0.0, 0.0, 0.01
    xs, ys = [1,2,3,4,5.0], [3,5,7,9,11.0]
    for epoch in range(200):
        dw = sum(2*(w*x+b-y)*x for x,y in zip(xs,ys))/len(xs)
        db = sum(2*(w*x+b-y) for x,y in zip(xs,ys))/len(xs)
        w -= lr*dw; b -= lr*db
    print(f"  学到: y = {w:.2f}x + {b:.2f} (真实: y = 2x + 1)")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| PyTorch `tensor.backward()` | 反向传播自动计算梯度 |
| NumPy | 数值导数验证 |
| SciPy `optimize` | 高级优化器（L-BFGS 等） |

---

## 5. 知识连线

- **第 03 阶段 · 05（反向传播）**：链式法则是反向传播的核心
- **第 03 阶段 · 08（优化）**：Adam 是一阶导数的自适应应用
- **第 19 阶段 · 44（余弦学习率）**：学习率控制梯度下降的步长

---

## 6. 工程最佳实践

- **学习率是最重要的超参数**：太大→发散，太小→收敛慢。0.001 是 Adam 的良好起点
- **数值导数用于验证**：实现新操作时用数值导数检查解析导数正确性
- **中文场景建议**：反向传播的梯度流在大语言模型中是所有训练的基础

---

## 7. 常见错误

### 错误 1：学习率太大

**现象：** 损失震荡或发散。

**原因：** 步长超过损失曲面的曲率，跨过最小值。

**修复：** 降低学习率，从 0.001 开始扫描。

### 错误 2：未实现二阶导数

**现象：** 无法使用牛顿法优化。

**原因：** 只计算了一阶导数。

**修复：** 用数值二阶导数或 Hessian 矩阵。

---

## 8. 面试考点

### Q1：为什么深度学习用梯度下降而不是牛顿法？（难度：⭐⭐）

**参考答案：** 牛顿法需要 N×N 的 Hessian 矩阵（N 是参数数量）。百万参数模型需要万亿参数的矩阵——内存和计算都不可行。梯度下降只用一阶导数，O(N) 计算成本。Adam 近似二阶信息而不需要 Hessian。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 导数 | 变化率——函数在某点的斜率 |
| 梯度 | 所有偏导数的向量——指向最陡上升方向 |
| 梯度下降 | 减去学习率×梯度来最小化损失 |
| 链式法则 | 复合函数的导数=各层局部导数之积 |
| Hessian | 二阶偏导数矩阵——描述曲率 |

---

## 📚 小结

微积分是理解神经网络如何学习的语言。你从零实现了数值导数、梯度下降和 Hessian 计算。下一课深入链式法则和自动微分。

---

## ✏️ 练习

1. 【实现】实现数值二阶导数 `numerical_second_derivative`
2. 【实验】用梯度下降最小化 f(x,y) = (x-3)² + (y+1)²，从 (0,0) 开始
3. 【理解】用动量加速梯度下降：维护速度向量累积历史梯度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 微积分工具 | `code/calculus_ml.py` | 数值导数、梯度、梯度下降、Hessian |

---

## 📖 参考资料

1. [视频] 3Blue1Brown 微积分的本质. https://www.3blue1brown.com/topics/calculus
2. [论文] Stanford CS231n 反向传播. https://cs231n.github.io/optimization-2/
