# 综合项目54——论文写作器（Paper Writer）

> LaTeX 骨架是研究者与排版器之间的合约。先构建骨架，再填充内容。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第50-53节
**预计时间：** 90分钟

---

## 学习目标

- 将论文视为结构化产物——已知章节图而非自由格式文档
- 生成 LaTeX 骨架，声明摘要、章节、图表插槽和参考文献键
- 通过确定性插槽从实验输出注入图表
- 输出 `paper.tex` + `references.bib` + 清单

---

## 1. 问题

从散文开始的草稿积累结构性债务——引言长出应在相关工作的段落，图表在定义前被引用，参考文献三键指同一篇论文。等作者发现时，重写成本已高于写作成本。骨架颠倒这一切——结构作为数据预先声明。

---

## 2. 核心概念

### 2.1 论文数据结构

```text
Paper(title, authors, abstract)
  sections: [Section(id, title, body, cites)]
  figures:  [Figure(id, path, caption)]
  bib:      {key: BibEntry(key, fields)}
```

### 2.2 渲染合约

每个图表插槽发出 `\begin{figure}`（标签 `fig:<id>`），每个章节发出 `\section{}`（标签 `sec:<id>`），参考文献内容精确匹配声明。

### 2.3 验证门

写入前：每个图表 ID 唯一、每个引用有 BibEntry、摘要和标题非空。

---

## 3. 从零实现

```python
"""论文写作器——骨架+图表注入+LaTeX 渲染。"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BibEntry:
    key: str; fields: Dict[str, str] = field(default_factory=dict)
    def to_bibtex(self) -> str:
        return f"@{self.fields.get('type','article')}{{{self.key},\n" + "".join(f"  {k}={{{v}}},\n" for k,v in self.fields.items()) + "}"

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
        if not self.title: raise ValueError("标题空")
        if not self.abstract: raise ValueError("摘要空")
        fids = [f.id for f in self.figures]
        if len(fids) != len(set(fids)): raise ValueError("图 ID 重复")
        for s in self.sections:
            for c in s.cites:
                if c not in self.bib: raise ValueError(f"引用 {c} 未声明")


def render_latex(paper: Paper) -> str:
    lines = [r"\documentclass{article}", r"\usepackage{graphicx}", r"\begin{document}",
             rf"\title{{{paper.title}}}", rf"\author{{{', '.join(paper.authors)}}}",
             r"\maketitle", rf"\begin{{abstract}}{paper.abstract}\end{{abstract}}"]
    for sec in paper.sections:
        lines.append(rf"\section{{{sec.title}}}\label{{sec:{sec.id}}}")
        lines.append(f"{sec.body} " + (rf"\cite{{{','.join(sec.cites)}}}" if sec.cites else ""))
    for fig in paper.figures:
        lines.extend([r"\begin{figure}", rf"\includegraphics[width=0.5\textwidth]{{{fig.path}}}",
                      rf"\caption{{{fig.caption}}}\label{{{fig.label}}}", r"\end{figure}"])
    if paper.bib: lines.extend([r"\bibliographystyle{plain}", r"\bibliography{references}"])
    lines.append(r"\end{document}")
    return "\n".join(lines)


class PaperWriter:
    def write(self, paper: Paper, out_dir: str) -> Dict:
        paper.validate(); os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "paper.tex"), "w") as f: f.write(render_latex(paper))
        with open(os.path.join(out_dir, "references.bib"), "w") as f:
            for e in paper.bib.values(): f.write(e.to_bibtex()+"\n")
        manifest = {"sections": [s.id for s in paper.sections],
                    "figures": [{"id": f.id, "path": f.path} for f in paper.figures],
                    "citations": list(set(c for s in paper.sections for c in s.cites))}
        with open(os.path.join(out_dir, "manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
        return manifest


def main():
    p = Paper("注意力稀疏性研究", ["A"], "本文研究注意力稀疏性。")
    p.sections = [Section("intro","引言","稀疏性可提效。"), Section("method","方法","使用 top-k。",["r1"])]
    p.figures = [Figure("loss","figs/loss.png","损失曲线")]
    p.bib["r1"] = BibEntry("r1", {"type":"article","title":"Efficient Attn","author":"X","year":"2023"})
    m = PaperWriter().write(p, "/tmp/paper_out")
    print(f"{len(m['sections'])} 节, {len(m['figures'])} 图, {len(m['citations'])} 引用")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 格式 | 特点 |
|:----|:-----|:-----|
| LaTeX 原生 | .tex | 学术标准 |
| Pandoc | 多格式 | 从 Markdown 生成 LaTeX |
| Typst | .typ | 现代替代，编译速度 > LaTeX |

---

## 5. 工程最佳实践

- 骨架优先工作流适用于任何文档类型
- **中文场景建议**：LaTeX 中使用中文需要 `ctex` 包或 XeLaTeX

---

## 6. 常见错误

- **图路径用绝对路径**：LaTeX 编译可能失败；使用相对路径
- **引用键未声明**：`\cite{key}` 无对应 BibEntry 则编译报错
- **验证门失效**：写入前必须验证——部分写入后失败留下不一致产物

---

## 7. 面试考点

**Q1：为什么骨架优先比直接写散文好？**（难度：⭐⭐）

**参考答案：** 骨架将结构从内容中分离，可在填充内容前验证完整性。编辑结构（重新排列章节）比编辑散文便宜。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 论文骨架 | 章节/图表/引用预声明为数据 |
| 图表注入 | 从实验清单自动插入 |
| 验证门 | 写入前检查合约完整性 |

---

## 📚 小结

论文写作器将结构化数据渲染为可编译的 LaTeX。下一节构建评审循环自动改进草稿。

---

## ✏️ 练习

1. 【实现】添加 `render_markdown` 函数输出 Markdown 而非 LaTeX
2. 【实验】验证 `paper.validate()` 在缺失引用键时抛错

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 论文写作器 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] LaTeX. https://www.latex-project.org/
