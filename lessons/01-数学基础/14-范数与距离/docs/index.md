# 范数与距离——定义"相似"的度量

> 你的距离函数定义了什么是"相似"。选错了，下游一切都会崩。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-02
**预计时间：** 90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 19 阶段 · 65（混合检索）— BM25 与余弦相似度的融合

---

## 🎯 学习目标

- [ ] 从零实现 L1、L2、余弦、马氏、Jaccard 和编辑距离
- [ ] 为给定 ML 任务选择合适距离度量
- [ ] 将 L1/L2 范数与 LASSO/Ridge 正则化联系起来
- [ ] 演示同一数据集在不同度量下产生不同最近邻

---

## 1. 问题

两个向量有多近？答案完全取决于距离函数。KNN、推荐系统、向量数据库、聚类、损失函数全依赖此选择。选错就优化错误目标。

---

## 2. 核心概念

| 距离 | 公式 | 适用 |
|:-----|:-----|:-----|
| L1 | Σ\|xᵢ\| | 稀疏高维、异常值鲁棒 |
| L2 | √Σxᵢ² | 空间数据、图像 |
| 余弦 | 1-cos(θ) | NLP、嵌入 |
| 马氏 | √((x-y)^T S⁻¹ (x-y)) | 离群检测 |
| Jaccard | 1-\|A∩B\|/\|A∪B\| | 集合重叠 |
| 编辑 | 最少插入/删除/替换 | 字符串匹配 |

正则化联系：L1(Lasso) → 稀疏权重；L2(Ridge) → 小权重。

---

## 3. 从零实现

```python
"""距离度量——L1+L2+余弦+Jaccard+编辑距离。"""
import math

def l1_dist(a,b): return sum(abs(x-y) for x,y in zip(a,b))
def l2_dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def cosine_sim(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x**2 for x in a)); nb=math.sqrt(sum(x**2 for x in b))
    return dot/(na*nb) if na*nb else 0
def jaccard_sim(a,b): sa,sb=set(a),set(b); return len(sa&sb)/max(len(sa|sb),1)
def edit_distance(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[n][m]

def main():
    a,b=(1,2,3),(4,0,6)
    print(f"L1={l1_dist(a,b)} L2={l2_dist(a,b):.4f}")
    print(f"余弦相似度={cosine_sim(a,b):.4f}")
    print(f"Jaccard({{'cat','dog'}},{{'cat','bird','fish'}})={jaccard_sim(['cat','dog'],['cat','bird','fish']):.3f}")
    print(f"编辑距离('kitten','sitting')={edit_distance('kitten','sitting')}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| FAISS | 大规模 ANN 搜索 |
| scikit-learn | 距离函数和近邻算法 |
| vector DB (Qdrant等) | 向量搜索服务 |

---

## 5. 知识连线

- **第 19 阶段 · 65（混合检索）**：BM25 + 余弦的 RRF 融合
- **第 10 阶段（LLM 从零）**：嵌入空间中的距离是语义搜索的基础

---

## 6. 工程最佳实践

- **文本相似性用余弦**：长度是噪音，方向是含义
- **嵌入已 L2 归一化时**：点积和余弦等价
- **中文场景建议**：中文向量数据库（如 Milvus）支持余弦和 L2

---

## 7. 常见错误

- **在 t-SNE 输出上比较距离**：t-SNE 只保留局部邻域，全局距离无意义
- **余弦 vs 点积混淆**：未归一化嵌入时两者不同

---

## 8. 面试考点

### Q1：L1 正则化为什么产生稀疏？（难度：⭐⭐）

**参考答案：** L1 约束区域是菱形（2D 中），损失函数等高线（椭圆）最易与菱形顶点相切——某些权重恰好为零。L2 约束是圆，无此特性，权重变小但不为零。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 余弦相似度 | 点积归一化——向量方向度量 |
| 编辑距离 | 最少单字符操作数——字符串相似度 |
| Mahalanobis | 考虑协方差的距离 |
| Jaccard | 集合重叠度 |
| L1/L2 正则化 | 稀疏 vs 平滑权重约束 |

---

## 📚 小结

距离函数定义了"相似"的含义。你实现了六种距离，理解了 L1/L2 与正则化的联系，以及不同度量产生不同最近邻。

---

## ✏️ 练习

1. 【实现】在同一数据集上对比 L1、L2、余弦的最近邻
2. 【实验】构建余弦高但 L2 大的向量对，解释原因

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 距离度量 | `code/main.py` |

---

## 📖 参考资料

1. [GitHub] FAISS. https://github.com/facebookresearch/faiss
2. [论文] Arjovsky et al. "Wasserstein GAN". https://arxiv.org/abs/1701.07875
