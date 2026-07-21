# 向量矩阵运算——神经网络就是矩阵乘法加额外步骤

> 每个神经网络就是矩阵乘法加额外步骤。

**类型：** 构建
**编程语言：** Python、Julia
**前置知识：** 第 01 阶段 · 01（线性代数直觉）
**预计时间：** 60 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 10（神经网络框架）— 本节的 Matrix 类是该框架的基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建 Matrix 类，实现逐元素运算、矩阵乘法、转置、行列式和逆
- [ ] 区分逐元素乘法和矩阵乘法，说明各自适用场景
- [ ] 仅用从零 Matrix 类实现一个密集神经网络层 `relu(W @ x + b)`
- [ ] 解释广播规则和偏差加法在神经网络框架中的工作方式

---

## 1. 问题

你想构建神经网络。你读代码看到这行：

```
output = activation(weights @ input + bias)
```

`@` 是矩阵乘法。`weights` 是矩阵。`input` 是向量。如果你不知道这些操作做什么，这一行就是魔法。如果你知道，这就是一个层的全部前向传播——三个操作。

模型处理的每张图像是像素值矩阵。每个词嵌入是向量。神经网络的每一层是矩阵变换。不能流畅掌握矩阵操作就不能构建 AI 系统——就像不理解变量就不能写代码一样。

---

## 2. 核心概念

### 2.1 矩阵乘法规则

`(m × n) @ (n × p) = (m × p)` — 内维必须匹配。

### 2.2 逐元素 vs 矩阵乘法

```
逐元素 (逐位置相乘)：    矩阵乘法（行×列求和）：
| 1  2 |   | 5  6 |   | 5  12 |    | 1  2 |   | 5  6 |   | 19  22 |
| 3  4 | × | 7  8 | = | 21 32 |    | 3  4 | @ | 7  8 | = | 43  50 |
```

### 2.3 广播（Broadcasting）

当形状不匹配时，NumPy 自动沿缺失维度扩展较小的数组。这就是偏差加法在每个神经网络框架中的工作方式。

---

## 3. 从零实现

```python
"""从零实现向量和矩阵运算。"""
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.shape = (len(self.data), len(self.data[0]))

    def __add__(self, other):
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])]
                       for i in range(self.shape[0])])

    def scalar_mul(self, s):
        return Matrix([[x * s for x in row] for row in self.data])

    def matmul(self, other):
        return Matrix([[sum(self.data[i][k] * other.data[k][j]
                         for k in range(self.shape[1])) for j in range(other.shape[1])]
                       for i in range(self.shape[0])])

    def transpose(self):
        return Matrix([[self.data[j][i] for j in range(self.shape[0])]
                       for i in range(self.shape[1])])

    def determinant(self):
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.shape[1]):
            minor = Matrix([[self.data[i][k] for k in range(self.shape[1]) if k != j]
                            for i in range(1, self.shape[0])])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        return Matrix([[self.data[1][1] / det, -self.data[0][1] / det],
                       [-self.data[1][0] / det, self.data[0][0] / det]])

    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    def __repr__(self):
        return f"Matrix({self.data})"

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])


# 演示：矩阵运算
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
print(f"A @ B = {A.matmul(B).data}")
print(f"det(A) = {A.determinant()}")
print(f"A × A⁻¹ = {A.matmul(A.inverse_2x2()).data}")

# 演示：神经网络层
import random; random.seed(42)
inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([[random.uniform(-1,1) for _ in range(3)] for _ in range(2)])
bias = Matrix([[0.1], [0.1]])
output = relu_matrix(weights.matmul(inputs) + bias)
print(f"\n神经网络层: {inputs.shape} → {output.shape}")
print(f"输出: {output.data}")

# NumPy 等价
A_np = np.array([[1,2],[3,4]]); B_np = np.array([[5,6],[7,8]])
print(f"\nNumPy A @ B:\n{A_np @ B_np}")
```

---

## 4. 工业工具

| 工具 | 加速倍数 | 适用 |
|:-----|:---------|:-----|
| 本课手写 | 1x | 教学理解 |
| NumPy (BLAS) | 100x | CPU 生产 |
| PyTorch | GPU 加速 | 大规模训练 |

---

## 5. 知识连线

- **第 03 阶段 · 10（神经网络框架）**：本节的 Matrix 类是该框架的基础
- **第 07 阶段（Transformer 深入）**：矩阵乘法是注意力机制的核心
- **第 19 阶段 · 01（GPT 模型组装）**：矩阵操作贯穿 GPT 的每一层

---

## 6. 工程最佳实践

- **逐元素乘法 vs 矩阵乘法**：逐元素是位置对应相乘（需要相同形状），矩阵乘法是行点积（内维匹配）
- **广播是自动的但要理解**：理解广播规则防止混淆形状不匹配但代码能运行的情况
- **中文场景建议**：矩阵操作的维度错误是最常见的调试问题——始终打印 `tensor.shape`

---

## 7. 常见错误

### 错误 1：维度不匹配

**现象：** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**原因：** `(m × n) @ (n × p)` 要求内维 `n` 相等。

**修复：** 打印矩阵 `.shape` 检查维度。

### 错误 2：混淆 `@` 和 `*`

**现象：** `A * B` 返回逐元素结果而非矩阵乘法。

**原因：** Python 中 `*` 是逐元素乘法，`@` 是矩阵乘法。

**修复：** 矩阵乘法用 `A @ B` 或 `np.dot(A, B)`。

---

## 8. 面试考点

### Q1：为什么矩阵乘法不可交换？（难度：⭐⭐）

**参考答案：** `(m×n) @ (n×p) = (m×p)` 但 `(n×p) @ (m×n)` 可能维度不匹配。即使两者都有效（如两个方阵），`A @ B` 和 `B @ A` 通常不相等——旋转后缩放 vs 缩放后旋转产生不同结果。这在神经网络中意味着层的顺序至关重要。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 矩阵乘法 | 行与列的点积——内维必须匹配 |
| 广播 | 较小数组沿缺失维度自动扩展 |
| 逐元素乘法 | 位置对应相乘——要求相同形状 |
| 恒等矩阵 | 乘以它不改变任何东西——残差连接的基础 |

---

## 📚 小结

矩阵运算是神经网络的底层语言。你实现了完整的 Matrix 类并构建了第一个神经网络层。下一课学习矩阵变换和特征值。

---

## ✏️ 练习

1. 【实现】用 Matrix 类构建两层神经网络：输入(3)→隐藏(4)→输出(2)
2. 【实验】验证 `A @ A⁻¹` 等于恒等矩阵
3. 【理解】NumPy 中 `A * B` 和 `A @ B` 的区别

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| Matrix 类 | `code/matrix_operations.py` | 完整矩阵运算+神经网络层 |

---

## 📖 参考资料

1. [视频] 3Blue1Brown 线性代数的本质. https://www.3blue1brown.com/topics/linear-algebra
2. [官方文档] PyTorch `torch.matmul`. https://pytorch.org/docs/stable/generated/torch.matmul.html
