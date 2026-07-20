# 综合项目04——多模态文档问答（视觉优先PDF、表格、图表）

> 2026年的文档问答前沿已从"先OCR再文本"转向"视觉优先的延迟交互"。ColPali、ColQwen2.5和ColQwen3-omni将每页PDF作为图像处理，使用多向量延迟交互嵌入，让查询直接关注图像块。在金融10-K报表、科学论文和手写笔记上，这种模式大幅超越先OCR再文本。本综合项目要求你构建端到端的管道，处理1万页文档，并与先OCR再文本基线做对比评测。

**类型：** 综合项目
**编程语言：** Python（管道），TypeScript（查看器UI）
**前置知识：** 第4章（计算机视觉）、第5章（NLP）、第7章（Transformer）、第11章（LLM工程）、第12章（多模态）、第17章（基础设施）
**涉及章节：** P4 · P5 · P7 · P11 · P12 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建视觉优先的文档问答管道：页渲染 → 多向量嵌入 → MaxSim检索 → VLM合成
- 实现ColPali/ColQwen风格的延迟交互检索
- 实现DocPruner风格的补丁剪枝（50%压缩，接近零精度损失）
- 在ViDoRe v3基准上对比视觉优先 vs 先OCR再文本

---

## 1. 问题

企业拥有大量OCR管道难以处理的PDF：带有旋转表格的扫描版10-K报表、充满方程式的科学论文、只能作为图像理解的图表、手写注释。将它们作为文本优先处理意味着丢失一半信息。

2026年的答案是原始页面图像上的延迟交互多向量检索。ColPali（Illuin Tech）首次提出；ColQwen2.5-v0.2和ColQwen3-omni提升了准确率。在ViDoRe v3上，视觉优先检索的分数远超先OCR再文本——差异在图表、表格和手写内容上更大。

权衡是存储和延迟。ColQwen嵌入每页约2048个图像块向量，而非单个1024维向量。DocPruner（2026）实现了50%剪枝而无显著精度损失。

---

## 2. 核心概念

### 2.1 延迟交互（Late Interaction）

每个查询token与每个图像块token分别评分，每个查询token的最大评分累加。你获得细粒度的匹配，无需单个池化向量。

多向量索引（Vespa、Qdrant multi-vector或AstraDB）存储每个图像块的嵌入，在检索时运行MaxSim。

### 2.2 视觉语言模型合成器

答案生成器是一个视觉语言模型，它接收查询加上top-k检索到的页面图像作为输入，输出带证据区域（边界框或页面引用）的答案。

2026年前沿选择：Qwen3-VL-30B、Gemini 2.5 Pro、InternVL3。对于方程和科学符号，可选OCR回退（Nougat、dots.ocr）作为额外文本通道。

### 2.3 评测矩阵

两个维度的矩阵。一个轴：内容类型（纯文本段落、密集表格、柱状/线图、手写笔记、方程式）。另一个轴：检索方法（视觉优先延迟交互 vs 先OCR再文本 vs 混合）。每个单元格获得nDCG@5和答案准确率。

---

## 3. 从零实现

`code/main.py`实现MaxSim延迟交互检索的核心算法——在合成图像块嵌入上端到端运行，不依赖实际ColQwen模型。

```python
"""多模态文档QA——ColPali风格延迟交互脚手架。

核心架构原语是延迟交互检索：每个查询token与每个文档图像块token评分，
每个查询token的MaxSim累加，返回top-k页面。
本脚手架在合成图像块嵌入上端到端实现MaxSim，
包含DocPruner风格的补丁剪枝。

运行：python3 code/main.py
"""

import math
import random
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 图像块嵌入——每页模拟16维向量
# ---------------------------------------------------------------------------

EMB_DIM = 16


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def hash_embed(tok: str) -> list[float]:
    """基于哈希的确定性嵌入生成"""
    rnd = random.Random(hash(tok) & 0xFFFFFFFF)
    v = [rnd.gauss(0, 1) for _ in range(EMB_DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@dataclass
class Page:
    doc_id: str
    page_num: int
    content_tokens: list[str]          # 代替页面内容
    patches: list[list[float]] = field(default_factory=list)

    def embed_patches(self) -> None:
        """多向量：每个内容词元成为一个图像块向量"""
        self.patches = [hash_embed(t) for t in self.content_tokens]


# ---------------------------------------------------------------------------
# DocPruner——按规范保持top分数的图像块
# ---------------------------------------------------------------------------

def doc_prune(patches: list[list[float]], keep_fraction: float = 0.5) -> list[list[float]]:
    """保留每个图像块范数最高的部分"""
    scored = [(sum(abs(x) for x in p), p) for p in patches]
    scored.sort(key=lambda x: -x[0])
    keep_n = max(1, int(len(scored) * keep_fraction))
    return [p for _, p in scored[:keep_n]]


# ---------------------------------------------------------------------------
# MaxSim延迟交互——ColPali/ColQwen的核心算法
# ---------------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def max_sim_score(query_tokens: list[list[float]],
                  doc_patches: list[list[float]]) -> float:
    """每个查询token嵌入取与任何文档图像块的最大点积，
    跨查询token累加。这就是MaxSim/延迟交互。"""
    total = 0.0
    for q in query_tokens:
        best = -1e9
        for p in doc_patches:
            s = dot(q, p)
            if s > best:
                best = s
        total += best
    return total


# ---------------------------------------------------------------------------
# 索引 + 检索——按MaxSim排名的top-k
# ---------------------------------------------------------------------------

@dataclass
class Index:
    pages: list[Page] = field(default_factory=list)

    def add(self, p: Page) -> None:
        self.pages.append(p)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Page, float]]:
        q_tokens = [hash_embed(t) for t in tokenize(query)]
        scored = [(pg, max_sim_score(q_tokens, pg.patches)) for pg in self.pages]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# 合成语料库——10页，覆盖表格、图表、手写、文本
# ---------------------------------------------------------------------------

CORPUS = [
    ("10k-2024", 88, "segment EMEA operating margin 18.2 to 16.8 decline 140bp table four"),
    ("10k-2024", 92, "MDA operating performance EMEA macro headwinds FX impact narrative"),
    ("10k-2024", 14, "executive summary revenue growth 7 percent consolidated totals"),
    ("paper-vidore-v3", 3, "late interaction multi vector retrieval ColPali ColQwen benchmark"),
    ("paper-vidore-v3", 7, "nDCG results table vision first vs OCR then text columns"),
    ("paper-m3docrag", 2, "M3DocVQA multi page reasoning evaluation protocol"),
    ("handwritten-lab", 5, "experiment notes circuit board pH readings handwritten"),
    ("handwritten-lab", 6, "graph with annotated error bars figure 3 caption"),
    ("chart-report", 11, "line chart revenue by segment EMEA americas APAC Q1 Q4"),
    ("chart-report", 12, "bar chart operating margin by segment with 2023 2024 comparison"),
]


def build_index(prune: bool = True) -> Index:
    idx = Index()
    for doc, page, text in CORPUS:
        p = Page(doc_id=doc, page_num=page, content_tokens=tokenize(text))
        p.embed_patches()
        if prune:
            p.patches = doc_prune(p.patches, keep_fraction=0.5)
        idx.add(p)
    return idx


def main() -> None:
    print("=== 构建索引（DocPruner 50%图像块）===")
    idx = build_index(prune=True)
    print(f"索引页面数: {len(idx.pages)}")

    queries = [
        "what was the 2024 operating margin change for EMEA",
        "late interaction retrieval vs OCR",
        "handwritten experimental figures with error bars",
        "bar chart comparing segment margins",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        hits = idx.retrieve(q, k=3)
        for pg, score in hits:
            print(f"  score={score:+.3f}  {pg.doc_id} p.{pg.page_num}")

    # 剪枝消融实验
    print("\n=== 消融实验：剪枝关闭 vs 开启 ===")
    full = build_index(prune=False)
    pruned = build_index(prune=True)
    q = "chart comparing segment margins"
    full_top = [(p.doc_id, p.page_num) for p, _ in full.retrieve(q, 3)]
    prn_top = [(p.doc_id, p.page_num) for p, _ in pruned.retrieve(q, 3)]
    print(f"  完整    top-3 : {full_top}")
    print(f"  剪枝    top-3 : {prn_top}")
    print(f"  重叠         : {len(set(full_top) & set(prn_top))}/3")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 构建索引（DocPruner 50%图像块）===
索引页面数: 10

Q: what was the 2024 operating margin change for EMEA
  score=+2.021  10k-2024 p.88
  score=+1.320  10k-2024 p.92
  score=+0.972  10k-2024 p.14

Q: late interaction retrieval vs OCR
  score=+1.991  paper-vidore-v3 p.3
  score=+1.177  paper-vidore-v3 p.7
  score=+1.039  paper-m3docrag p.2

Q: handwritten experimental figures with error bars
  score=+1.790  handwritten-lab p.6
  score=+1.288  handwritten-lab p.5
  score=+0.531  chart-report p.11

Q: bar chart comparing segment margins
  score=+1.477  chart-report p.12
  score=+1.298  chart-report p.11
  score=+1.260  10k-2024 p.88

=== 消融实验：剪枝关闭 vs 开启 ===
  完整    top-3 : [('chart-report', 12), ('chart-report', 11), ('10k-2024', 88)]
  剪枝    top-3 : [('chart-report', 12), ('chart-report', 11), ('10k-2024', 88)]
  重叠         : 3/3
```

---

## 4. 工具实践

**技术栈：**
- 页面渲染：PyMuPDF（fitz），180 DPI
- 延迟交互模型：ColQwen2.5-v0.2或ColQwen3-omni（vidore团队，Hugging Face）
- 索引：Vespa多向量字段或Qdrant multi-vector或AstraDB
- 剪枝：DocPruner 2026策略
- VLM答案合成：Qwen3-VL-30B自托管或Gemini 2.5 Pro托管
- 评测：ViDoRe v3基准，M3DocVQA多页推理

---

## 5. LLM视角

**视觉优先视角**：延迟交互让视觉信息在检索阶段就参与进来，而非在OCR后丢失图表和表格中的信息。在财务报告和科学论文上效果尤为显著。

**剪枝视角**：DocPruner将每页2048个图像块剪枝到1024个，精度损失不到0.5%。这是实用性的关键——否则存储成本过高。

**多页推理视角**：VLM可以同时查看多页图像进行推理，而OCR管道很难跨页关联上下文。

---

## 6. 工程最佳实践

**管道设计**：
- 每页渲染为1536x2048 PNG
- ColQwen嵌入，每页约2048个128维图像块向量
- DocPruner剪枝保留高信号一半
- MaxSim检索top-k页面

**答案合成**：
- VLM接收查询+top-5页面图像
- 要求引用（doc_id, page）和证据区域
- 可选OCR回退通道

**评测**：
- ViDoRe v3（检索nDCG@5）
- M3DocVQA（多页QA准确率）
- 按内容类型×方法构建矩阵

---

## 7. 常见错误

**错误1：先OCR再文本处理文档**
症状：表格和图表信息丢失，检索质量差
修复：使用视觉优先的延迟交互

**错误2：不剪枝图像块**
症状：存储和延迟成本过高
修复：应用DocPruner 50%剪枝

**错误3：忽略多页推理**
症状：答案仅基于单页信息
修复：VLM同时查看多页图像

---

## 8. 面试考点

**Q1：延迟交互（Late Interaction）与双编码器有什么不同？**
考察：对检索方法的理解

**Q2：为什么视觉优先在文档问答中优于先OCR再文本？**
考察：对多模态检索的理解

**Q3：DocPruner如何在不损失精度的情况下减少存储？**
考察：对剪枝方法的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 延迟交互 | "ColPali风格检索" | 查询token与页面图像块独立评分；MaxSim聚合 |
| 多向量 | "每图像块嵌入" | 每个文档有许多向量，而非单个池化向量 |
| MaxSim | "延迟交互评分" | 每个查询token取文档向量上的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026年剪枝，保持50%图像块，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026年视觉文档检索的衡量标准 |
| 证据区域 | "引用边界框" | 源页面上的边界框，定位答案范围 |

---

## 参考文献

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali)
- [ColPali论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449)
- [ColQwen家族（Hugging Face）](https://huggingface.co/vidore)
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952)
- [Vespa多向量教程](https://docs.vespa.ai/en/colpali.html)
- [Qdrant多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors)
