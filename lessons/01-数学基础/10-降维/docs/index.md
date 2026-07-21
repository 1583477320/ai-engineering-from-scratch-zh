# 降维——高维数据的结构

> 高维数据有结构。从正确的角度看就能发现它。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-03, 06
**预计时间：** 90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 02 阶段 · 04（特征选择）— 降维是特征工程的核心工具

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 PCA：中心化数据→计算协方差矩阵→特征分解→投影
- [ ] 用方差解释比和肘部法选择主成分数量
- [ ] 对比 PCA、t-SNE 和 UMAP 在 MNIST 2D 可视化上的权衡
- [ ] 用 RBF 核核化 PCA 分离非线性数据

---

## 1. 问题

数据集每个样本 784 个特征。可能是手写数字像素、基因表达、用户行为。你无法可视化 784 维。但大多数特征是冗余的——真正的信息在一个小得多的表面上。降维找到那个小表面。

---

## 2. 核心概念

### 2.1 PCA 五步

1. 中心化数据
2. 计算协方差矩阵
3. 特征分解
4. 按特征值排序
5. 投影到前 k 个特征向量

### 2.2 方差解释比

```
explained_ratio_k = eigenvalue_k / Σ 所有特征值
```

累积到 0.95 时，后续成分大多是噪声。

### 2.3 t-SNE vs UMAP

| 方法 | 适用 | 保留 | 速度 |
|:-----|:-----|:-----|:-----|
| PCA | 预处理 | 全局方差 | 快 |
| t-SNE | 发表级可视化 | 局部邻域 | 慢 |
| UMAP | 大规模可视化 | 局部+部分全局 | 中 |

### 2.4 核 PCA

标准 PCA 找线性子空间。核 PCA 在高维特征空间中执行 PCA——不显式计算高维坐标。

---

## 3. 从零实现

```python
"""从零实现 PCA。"""
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n = n_components; self.components = None; self.mean = None
        self.eigenvalues = None; self.explained_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = np.cov(Xc, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        self.components = vecs[:, :self.n].T
        self.eigenvalues = vals[:self.n]
        total = np.sum(vals)
        self.explained_ratio = self.eigenvalues / total
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X):
        self.fit(X); return self.transform(X)


def main():
    np.random.seed(42)
    t = np.random.uniform(0, 2*np.pi, 500)
    X = np.column_stack([3*np.cos(t)+np.random.normal(0,0.2,500),
                         3*np.sin(t)+np.random.normal(0,0.2,500),
                         0.5*np.cos(t)+0.3*np.sin(t)+np.random.normal(0,0.1,500)])
    pca = PCA(2)
    Xr = pca.fit_transform(X)
    print(f"原始: {X.shape} → 降维: {Xr.shape}")
    print(f"方差解释: {pca.explained_ratio}")
    print(f"总方差保留: {sum(pca.explained_ratio):.4f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| sklearn `PCA` | 生产级 PCA |
| sklearn `TSNE` | 2D 可视化 |
| umap-learn `UMAP` | 大规模可视化 |
| PyTorch `PCA` | GPU 加速降维 |

---

## 5. 知识连线

- **第 02 阶段 · 04（特征选择）**：PCA 降维是特征工程的核心
- **第 19 阶段 · 10（数据管理）**：降维后的数据更适合存储和传输

---

## 6. 工程最佳实践

- **PCA 做预处理，t-SNE/UMAP 做可视化**——两者不混用
- **保留 95% 方差**作为降维维度选择的基准
- **中文场景建议**：中文 NLP 中降维常用于词嵌入分析——将 768 维嵌入降至 2-3 维观察聚类

---

## 7. 常见错误

- **在 t-SNE 输出上比较簇间距离**：t-SNE 只保留局部邻域，簇间距离无意义。
- **PCA 降维后直接聚类未检查方差保留**：用肘部法确认保留了足够方差。

---

## 8. 面试考点

### Q1：PCA 和 t-SNE 该用哪个？（难度：⭐⭐）

**参考答案：** PCA 是线性方法，快且全局——适合预处理和数据压缩。t-SNE 是非线性方法，擅长发现复杂簇结构——适合 2D 可视化。生产预处理用 PCA，论文可视化用 t-SNE/UMAP。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| PCA | 找最大方差方向，投影降维 |
| 主成分 | 协方差矩阵的特征向量 |
| 方差解释比 | 每个主成分捕获的方差比例 |
| 核 PCA | 在高维特征空间中执行 PCA |

---

## 📚 小结

降维将高维数据压缩到可理解的维度。你实现了从零 PCA，理解了方差解释比和选择主成分数量的方法。这些工具将在特征工程中反复使用。

---

## ✏️ 练习

1. 【实现】为 PCA 添加 `inverse_transform`，重建降维数据
2. 【实验】在 MNIST 上对比 10/50/200 维 PCA 的重建误差
3. 【实验】对同心圆数据运行核 PCA，验证线性 PCA 无法分离

---

## 📖 参考资料

1. [论文] Shlens. "A Tutorial on PCA". https://arxiv.org/abs/1404.1100
2. [论文] Wattenberg et al. "How to Use t-SNE Effectively". https://distill.pub/2016/misread-tsne/
