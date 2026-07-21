# 奇异值分解——线性代数的瑞士军刀

> SVD 是线性代数的瑞士军刀。每个矩阵都有。每个数据科学家都需要。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-03
**预计时间：** 120 分钟
**所处阶段：** Tier 1
**关联课程：** 第 02 阶段 · 04（特征选择）— PCA 的底层就是 SVD

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 通过幂迭代实现 SVD，解释 U、Sigma、V^T 的几何含义
- [ ] 应用截断 SVD 进行图像压缩并测量压缩比与重建误差
- [ ] 通过 SVD 计算 Moore-Penrose 伪逆求解超定最小二乘系统
- [ ] 将 SVD 与 PCA、推荐系统（潜在因子）和 NLP 潜在语义分析联系起来

---

## 1. 问题

你有一个 1000×2000 矩阵。可能是用户-电影评分、文档-词频表或图像像素值。你需要压缩它、去噪、找隐藏结构或求解超定系统。特征分解只能用于方阵。SVD 对任何矩阵都适用——任何形状，任何秩，无条件。

---

## 2. 核心概念

### 2.1 SVD 的几何含义

任何矩阵执行三步操作：旋转→缩放→旋转。

```
A = U × Sigma × V^T

V^T 旋转输入空间
Sigma 沿轴缩放
U 旋转到输出空间
```

### 2.2 外积形式

```
A = Σ σᵢ × uᵢ × vᵢ^T
```

截断此和得到最优低秩近似（Eckart-Young 定理）。

### 2.3 图像压缩

```
原始: 800×600 = 480,000 值
k=10:  14,010 值 (2.9%)
k=50:  70,050 值 (14.6%)
k=100: 140,100 值 (29.2%)
```

自然图像的奇异值快速衰减——前几个捕获主体结构。

---

## 3. 从零实现

```python
"""SVD 从零实现——幂迭代+截断+伪逆。"""
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]; v = np.random.randn(n); v = v / np.linalg.norm(v)
    for _ in range(num_iters):
        Mv = M @ v; v = Mv / np.linalg.norm(Mv)
    return v @ M @ v, v

def svd_from_scratch(A, k=None):
    m, n = A.shape; k = k or min(m, n)
    sigmas, us, vs, A_res = [], [], [], A.copy().astype(float)
    for _ in range(k):
        AtA = A_res.T @ A_res
        ev, v = power_iteration(AtA, 200)
        if ev < 1e-10: break
        sigma = np.sqrt(ev); u = A_res @ v / sigma
        sigmas.append(sigma); us.append(u); vs.append(v)
        A_res -= sigma * np.outer(u, v)
    return np.column_stack(us) if us else np.empty((m,0)), np.array(sigmas), np.column_stack(vs) if vs else np.empty((n,0))

def main():
    np.random.seed(42)
    A = np.random.randn(5, 4)
    U, S, V = svd_from_scratch(A)
    _, S_np, _ = np.linalg.svd(A, full_matrices=False)
    print(f"奇异值 (我们): {np.round(S, 4)}")
    print(f"奇异值 (NumPy): {np.round(S_np, 4)}")
    err = np.linalg.norm(A - U @ np.diag(S) @ V.T)
    print(f"重建误差: {err:.8f}")
    return 0
if __name__ == "__main__": import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| `np.linalg.svd()` | 生产级 SVD |
| `scipy.sparse.linalg.svds()` | 大型稀疏矩阵截断 SVD |
| `sklearn.TruncatedSVD` | 降维预处理 |

---

## 5. 知识连线

- **第 02 阶段 · 04（PCA）**：PCA 就是中心化数据的 SVD
- **第 10 阶段（LLM 从零）**：LoRA 使用低秩分解优化权重
- **第 19 阶段 · 78（ZeRO）**：参数分片利用低秩结构

---

## 6. 工程最佳实践

- **用 `np.linalg.svd(A)` 而非 `np.linalg.eig(A.T @ A)`**：后者条件数平方，损失数值精度
- **中文场景建议**：LSA 中文文档聚类时，先用 jieba 分词再构建词-文档矩阵

---

## 7. 常见错误

- **条件数过大**：SVD 是数值稳定的，但 A^TA 的特征分解不是。始终用 SVD 而非 A^TA 的特征分解。
- **截断秩过小**：信号混入噪声。用奇异值差距判断截断点。

---

## 8. 面试考点

### Q1：SVD 与特征分解的核心区别？（难度：⭐⭐）

**参考答案：** 特征分解要求方阵且有完整特征向量基。SVD 对任何形状、任何秩的矩阵都适用——它将旋转分解为两部分（输入空间的 V^T 和输出空间的 U），使得任何矩阵都能分解为旋转-缩放-旋转。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| SVD | A = U × Sigma × V^T，任意矩阵的分解 |
| 奇异值 | Sigma 对角元素——衡量每主方向的重要性 |
| 截断 SVD | 保留前 k 个奇异值的最优低秩近似 |
| 伪逆 | V × Sigma+ × U^T，超定系统的最小二乘解 |

---

## 📚 小结

SVD 是最通用的矩阵分解。你实现了幂迭代 SVD、截断压缩和伪逆，并理解了 SVD 与 PCA 的等价关系。

---

## ✏️ 练习

1. 【实现】不使用幂迭代，通过 A^TA 的特征分解计算 SVD
2. 【实验】对灰度图像进行不同秩的 SVD 压缩
3. 【实现】构建小型推荐系统——SVD 分解用户-电影矩阵

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| SVD 实现 | `code/svd.py` |

---

## 📖 参考资料

1. [视频] 3Blue1Brown SVD 直觉. https://www.youtube.com/watch?v=vSczTbgc8Rc
2. [书籍] Trefethen & Bau. Numerical Linear Algebra. https://people.maths.ox.ac.uk/trefethen/text.html
