# 综合项目63——多模态评估（Multimodal Evaluation）

> 训练是循环的一半。另一半是测量。本节从基元构建三个评估面：图像-标题检索（R@K）、视觉问答（精确匹配）、图像标题生成（BLEU-4）。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第58-62节
**预计时间：** 90分钟

---

## 学习目标

- 从相似度矩阵计算 Recall@K
- 计算 VQA 精确匹配准确率
- 无外部库从零实现 BLEU-4
- 三个指标联合评估合成评测套件

---

## 1. 问题

训练损失停滞不等于模型完成。训练损失衡量训练分布的拟合度，不衡量模型能否排序、回答问题或写出可接受的标题。三个评估面覆盖三种核心能力。

---

## 2. 核心概念

### 2.1 Recall@K

构建 `(N, N)` 余弦相似度矩阵。对每行按降序排序。R@K = 对角线索引在前 K 位中的比例。K=1 表示第一个结果就是正确匹配。

### 2.2 VQA 精确匹配

对每 (图像, 问题, 答案) 三元组，模型预测一个答案词元。正确当且仅当预测 = 参考。平均。

### 2.3 BLEU-4

```
BLEU-4 = BP × exp(mean(log p1, log p2, log p3, log p4))
```

BP 是简短惩罚。pn 是修正 n-gram 精度。使用 Chen-Cherry 平滑。

---

## 3. 从零实现

```python
"""多模态评估——R@K + VQA + BLEU-4。"""
import math
from collections import Counter


def recall_at_k(sim_matrix, k):
    """sim_matrix: (N, N) 余弦相似度。返回 (r@k, r@k_i2t)。"""
    N = sim_matrix.shape[0]
    i2r = sum(1 for i in range(N) if sim_matrix[i].argsort(descending=True)[:k].tolist().count(i) > 0) / N
    t2r = sum(1 for i in range(N) if sim_matrix[:, i].argsort(descending=True)[:k].tolist().count(i) > 0) / N
    return i2r, t2r


def vqa_exact_match(predictions, references):
    return sum(p == r for p, r in zip(predictions, references)) / max(len(predictions), 1)


def n_grams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu4(generated, references, smoothing=True):
    weights = [0.25] * 4
    precisions = []
    for n in range(1, 5):
        gen_ng = n_grams(generated, n)
        if not gen_ng:
            precisions.append(1e-10 if smoothing else 0)
            continue
        ref_counts = Counter()
        for r in references:
            for ng in n_grams(r, n):
                ref_counts[ng] = max(ref_counts[ng], r.count(ng) if False else sum(1 for x in n_grams(r, n) if x == ng))
        clipped = sum(min(gen_ng.count(ng), ref_counts.get(ng, 0)) for ng in set(gen_ng))
        p = clipped / len(gen_ng) if gen_ng else 0
        if p == 0 and smoothing: p = 1e-10
        precisions.append(p)
    bp = min(1.0, math.exp(1 - max(1, len(references[0])) / max(1, len(generated))) if references else 1.0)
    score = bp * math.exp(sum(w * math.log(max(p, 1e-10)) for w, p in zip(weights, precisions)))
    return score


def main():
    import torch
    # 模拟评估
    N = 20
    sim = torch.randn(N, N)  # 模拟相似度矩阵
    # 对角线加偏置使其更匹配
    for i in range(N): sim[i, i] += 0.5

    r1, _ = recall_at_k(sim, 1)
    r5, _ = recall_at_k(sim, 5)
    print(f"R@1={r1:.3f} R@5={r5:.3f}")

    preds = ["猫", "狗", "鸟", "鱼", "猫"]
    refs = ["猫", "狗", "鸟", "蛇", "猫"]
    em = vqa_exact_match(preds, refs)
    print(f"VQA精确匹配: {em:.3f}")

    cap = ["一只", "黑色", "的", "猫"]
    refs_cap = [["一只", "黑色", "猫"], ["黑色", "猫咪"]]
    b4 = bleu4(cap, refs_cap)
    print(f"BLEU-4: {b4:.4f}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 指标 | 评测集 | 工具 |
|:----|:------|:-----|
| R@K | MS-COCO, Flickr30K | `trec_eval` |
| VQA | VQA v2, GQA | `vqa.evaluation` |
| BLEU-4 | MS-COCO captioning | `sacrebleu`, `nltk` |

---

## 5. 工程最佳实践

- 评估数据必须与训练数据严格不重叠
- BLEU-4 使用平滑处理零概率 n-gram
- **中文场景建议**：BLEU 对中文需要先分词——中文字符级 BLEU 会高估质量

---

## 6. 常见错误

- **R@K 计算错方向**：图像到文本和文本到图像需要分别报告
- **BLEU 未使用平滑**：小样本零概率 n-gram 导致 log(0)
- **VQA 未归一化答案**：不同格式的相同答案应归一化

---

## 7. 面试考点

**Q1：为什么 R@1 比 BLEU 更直观？**（难度：⭐⭐）

**参考答案：** R@1 是二元判断（是否命中），人类可直接解释。BLEU 是 n-gram 几何平均，不直觉，但能衡量文本生成的细微质量差异。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| R@K | 正确匹配在前 K 个结果中的比例 |
| BLEU-4 | 1-4-gram 精度的几何平均 + 简短惩罚 |
| 精确匹配 | 预测与参考完全一致 |

---

## 📚 小结

多模态评估覆盖检索、问答和生成三种能力。你实现了三个指标的从零计算。下一节转向 RAG 中的分块策略。

---

## ✏️ 练习

1. 【实现】添加 CIDEr 指标（TF-IDF 加权 n-gram）
2. 【实验】对比训练前后的 R@K，验证模型确实学到了检索能力

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 多模态评估 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Papineni et al. "BLEU: a Method for Automatic Evaluation". ACL 2002.
2. [论文] Antol et al. "VQA: Visual Question Answering". ICCV 2015.
