"""论文写作器——结构化骨架+图表注入+LaTeX 渲染。"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class BibEntry:
    key: str; fields: Dict[str,str] = field(default_factory=dict)
    def to_bibtex(self):
        return f"@{self.fields.get('type','article')}{{{self.key},\n" + "".join(f"  {k}={{{v}}},\n" for k,v in self.fields.items()) + "}"

@dataclass
class Figure:
    id: str; path: str; caption: str; label: str = ""
    def __post_init__(self):
        if not self.label: self.label=f"fig:{self.id}"

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
        fids=[f.id for f in self.figures]
        if len(fids)!=len(set(fids)): raise ValueError("图ID重复")
        for s in self.sections:
            for c in s.cites:
                if c not in self.bib: raise ValueError(f"引用{c}未声明")

def render_latex(paper):
    L=[r"\documentclass{article}",r"\begin{document}",rf"\title{{{paper.title}}}",rf"\author{{{', '.join(paper.authors)}}}",r"\maketitle"]
    for s in paper.sections: L.append(rf"\section{{{s.title}}}\label{{sec:{s.id}}}")
    for f in paper.figures: L.extend([r"\begin{figure}",rf"\includegraphics{{{f.path}}}",rf"\caption{{{f.caption}}}",r"\end{figure}"])
    L.append(r"\end{document}"); return "\n".join(L)

class PaperWriter:
    def write(self, paper, out_dir):
        paper.validate(); os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,"paper.tex"),"w") as f: f.write(render_latex(paper))
        with open(os.path.join(out_dir,"references.bib"),"w") as f:
            for e in paper.bib.values(): f.write(e.to_bibtex()+"\n")
        manifest={"sections":[s.id for s in paper.sections],
                  "figures":[{"id":f.id,"path":f.path} for f in paper.figures],
                  "citations":list(set(c for s in paper.sections for c in s.cites))}
        with open(os.path.join(out_dir,"manifest.json"),"w") as f: json.dump(manifest,f,indent=2)
        return manifest

def main():
    p=Paper("注意力稀疏性研究",["A"],"本文研究注意力稀疏性。")
    p.sections=[Section("intro","引言","稀疏性可提效。"),Section("method","方法","top-k路由。",["r1"])]
    p.figures=[Figure("loss","figs/loss.png","损失曲线")]
    p.bib["r1"]=BibEntry("r1",{"type":"article","title":"Efficient Attn","author":"X","year":"2023"})
    m=PaperWriter().write(p,"/tmp/paper_out")
    print(f"{len(m['sections'])}节 {len(m['figures'])}图 {len(m['citations'])}引用")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
