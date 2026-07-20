# 综合项目54——论文写作器（Paper Writer）

> LaTeX 骨架是研究者与排版器之间的合约。如果合约被破坏，文档无法编译，且失败是响亮的。先构建骨架，再填充内容。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 将研究论文视为具有已知章节图的结构化产物，而非自由格式文档
- 生成声明了摘要、章节、图表插槽和参考文献键的 LaTeX 骨架
- 通过确定性插槽机制将实验输出的图表注入骨架
- 编写从结构化大纲填充每个章节的模拟散文生成器
- 输出 `paper.tex` + `references.bib` + 包含所有引用图表信息的清单

---

## 1. 问题

从散文开始的草稿会积累结构性债务。引言长出三段落应该在相关工作里。图表在定义之前被引用。参考文献以三个键指向同一篇论文结束。等作者注意到时，重写成本已经高于写作成本。

骨架倒置了这一切。结构作为数据预先声明。章节是具有名称和顺序的插槽。图表是具有 ID 和标题的插槽。参考文献键在顶部声明。散文逐个插槽生成到这些位置。在写任何散文之前，编排器可以验证每个图表都有插槽、每个引用都有条目、每个章节都出现在目录中。

---

## 2. 核心概念

### 2.1 论文数据结构

```text
Paper
  metadata: title, authors, abstract
  sections: list[Section(id, title, body, cites)]
  figures:  list[Figure(id, path, caption, label)]
  bib:      list[BibEntry(key, fields)]
```

### 2.2 渲染合约

渲染器保证三个属性：每个图表插槽发出 `\begin{figure}` 块（标签 `fig:<id>`），每个章节发出 `\section{}`（标签 `sec:<id>`），参考文献发出 `\bibliography` 块，其 `references.bib` 包含声明在论文上的条目——不多不少。

### 2.3 图表注入

```mermaid
flowchart LR
    Exp[实验 JSON] --> Reader[读取实验清单]
    Reader --> Figs[Figure 列表]
    Figs --> Paper[Paper.figures]
    Paper --> Render[render_latex]
    Render --> Out[paper.tex]
```

### 2.4 验证门

写入任何文件前运行四个门：每个图表 ID 唯一、每个章节引用的是已声明的参考文献、摘要非空、标题非空。

---

## 3. 从零实现

```python
"""论文写作器——结构化骨架+图表注入+LaTeX 渲染。"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BibEntry:
    key: str; fields: Dict[str, str] = field(default_factory=dict)

    def to_bibtex(self) -> str:
        items = "".join(f"  {k} = {{{v}}},\n" for k, v in self.fields.items())
        return f"@{self.fields.get('type', 'article')}{{{self.key},\n{items}}}"


@dataclass
class Figure:
    id: str; path: str; caption: str; label: str = ""

    def __post_init__(self):
        if not self.label: self.label = f"fig:{self.id}"


@dataclass
class Section:
    id: str; title: str; body: str = ""; cites: List[str] = field(default_factory=list)


@dataclass
class Paper:
    title: str; authors: List[str]; abstract: str
    sections: List[Section] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    bib: Dict[str, BibEntry] = field(default_factory=dict)

    def validate(self):
        if not self.title: raise ValueError("标题为空")
        if not self.abstract: raise ValueError("摘要为空")
        fig_ids = [f.id for f in self.figures]
        if len(fig_ids) != len(set(fig_ids)):
            raise ValueError("图表 ID 重复")
        for sec in self.sections:
            for c in sec.cites:
                if c not in self.bib:
                    raise ValueError(f"引用 {c} 未在参考文献中定义")


class MockProseGenerator:
    def generate(self, section_title: str, outline: str) -> str:
        return f"{outline} 本节讨论 {section_title}。实验结果表明了有意义的结果。"


def render_latex(paper: Paper) -> str:
    lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        rf"\title{{{paper.title}}}",
        rf"\author{{{', '.join(paper.authors)}}}",
        r"\maketitle",
        rf"\begin{{abstract}}{paper.abstract}\end{{abstract}}",
        r"\tableofcontents",
    ]
    for sec in paper.sections:
        lines.append(rf"\section{{{sec.title}}}\label{{sec:{sec.id}}}")
        cites = ",".join(sec.cites) if sec.cites else ""
        cite_text = rf"\cite{{{cites}}}" if cites else ""
        lines.append(f"{sec.body} {cite_text}")
    for fig in paper.figures:
        lines.extend([
            r"\begin{figure}",
            rf"\centering\includegraphics[width=0.5\textwidth]{{{fig.path}}}",
            rf"\caption{{{fig.caption}}}\label{{{fig.label}}}",
            r"\end{figure}",
        ])
    if paper.bib:
        lines.extend([r"\bibliographystyle{plain}", r"\bibliography{references}"])
    lines.append(r"\end{document}")
    return "\n".join(lines)


def read_experiment_manifest(exp_paths: List[str]) -> List[Figure]:
    figs = []
    for path in exp_paths:
        data = json.load(open(path))
        artifacts = data.get("artifacts", [])
        for i, art in enumerate(artifacts):
            figs.append(Figure(id=f"{os.path.basename(path)}_{i}", path=art["path"],
                               caption=art.get("caption", "实验结果")))
    return figs


class PaperWriter:
    def __init__(self, prose_gen: MockProseGenerator = None):
        self.gen = prose_gen or MockProseGenerator()

    def write(self, paper: Paper, out_dir: str) -> Dict:
        paper.validate()
        os.makedirs(out_dir, exist_ok=True)
        tex_path = os.path.join(out_dir, "paper.tex")
        bib_path = os.path.join(out_dir, "references.bib")
        manifest = {"figures": [], "citations": [], "sections": []}

        for sec in paper.sections:
            if not sec.body:
                sec.body = self.gen.generate(sec.title, f"关于 {sec.title}")
            manifest["sections"].append(sec.id)

        with open(tex_path, "w") as f:
            f.write(render_latex(paper))
        with open(bib_path, "w") as f:
            for entry in paper.bib.values():
                f.write(entry.to_bibtex() + "\n")

        for fig in paper.figures:
            manifest["figures"].append({"id": fig.id, "path": fig.path, "caption": fig.caption})
        for sec in paper.sections:
            manifest["citations"].extend(sec.cites)
        manifest["citations"] = list(set(manifest["citations"]))

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def main():
    paper = Paper(title="注意力稀疏性的实证研究", authors=["研究者A", "研究者B"],
                  abstract="本文研究了小型 Transformer 中的注意力稀疏性模式。")
    paper.sections = [
        Section(id="intro", title="引言", body="注意力稀疏性是提高效率的关键方向。"),
        Section(id="method", title="方法", body="我们使用 top-k 路由实现稀疏注意力。", cites=["ref01"]),
        Section(id="results", title="结果", body="实验表明稀疏注意力匹配密集注意力。", cites=["ref01", "ref02"]),
    ]
    paper.figures = [
        Figure(id="loss_curve", path="figs/loss.png", caption="训练损失曲线"),
    ]
    paper.bib["ref01"] = BibEntry("ref01", {"type": "article", "title": "Efficient Attention", "author": "Someone", "year": "2023"})
    paper.bib["ref02"] = BibEntry("ref02", {"type": "inproceedings", "title": "Sparse is Enough", "author": "Nobody", "year": "2024"})

    writer = PaperWriter()
    manifest = writer.write(paper, "/tmp/paper_output")
    print(f"写入 {len(manifest['sections'])} 个章节")
    print(f"写入 {len(manifest['figures'])} 个图表")
    print(f"写入 {len(manifest['citations'])} 个引用")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 论文骨架 | 章节、图表、引用预先声明为数据结构的论文框架 |
| 图表注入 | 从实验清单自动将图表插入骨架 |
| 渲染合约 | LaTeX 输出的确定性保证（标签、引用完整性） |
| 验证门 | 写入前检查的四个条件，防止生成损坏的 LaTeX |
| 模拟散文生成器 | 从大纲确定性生成散文的 mock |

---

## 5. 工程最佳实践

- 骨架优先的工作流适用于任何文档类型（论文、博客、报告）
- **中文场景特别建议**：LaTeX 中直接使用中文需要 `ctex` 包或 XeLaTeX 编译，在渲染合约中应包含此项。

---

## 6. 常见错误

- **图表路径使用绝对路径**：LaTeX 编译时路径映射可能不同。应使用相对路径或复制图表到输出目录。
- **参考文献键未声明**：`\cite{key}` 必须有对应的 BibEntry，否则 LaTeX 编译报错。
- **验证门顺序错误**：验证应先于文件写入——部分写入后验证失败会留下不一致的产物。

---

## 📖 参考资料

1. [官方文档] LaTeX 文档类. https://www.latex-project.org/
2. [GitHub] `arxiv-latex-cleaner` — LaTeX 文件清理工具.
