"""文献检索——BM25 + 引用图遍历 + 去重排序。"""
from __future__ import annotations
import math, re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Paper:
    id: str; title: str; abstract: str; year: int
    references: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1=k1; self.b=b; self.doc_freq=Counter(); self.term_docs=defaultdict(list)
        self.doc_lens={}; self.avgdl=0.0; self.N=0

    def build(self, papers):
        for p in papers:
            toks=tokenize(p.abstract); self.doc_lens[p.id]=len(toks)
            for t,c in Counter(toks).items(): self.doc_freq[t]+=1; self.term_docs[t].append((p.id,c))
        self.N=len(papers); self.avgdl=sum(self.doc_lens.values())/max(self.N,1)

    def score(self, query, doc_id):
        q=set(tokenize(query)); dl=self.doc_lens.get(doc_id,0)
        if not q or dl==0: return 0.0
        total=0.0
        for t in q:
            f=sum(c for did,c in self.term_docs.get(t,[]) if did==doc_id)
            df=self.doc_freq.get(t,0)
            idf=math.log((self.N-df+0.5)/(df+0.5)+1.0)
            total+=idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/self.avgdl))
        return total


class CitationGraph:
    def __init__(self, papers):
        self.fwd={p.id:list(p.references) for p in papers}
        self.bwd={p.id:list(p.citations) for p in papers}

    def bfs(self, seeds, hops=2):
        dist={s:0 for s in seeds}; queue=list(seeds)
        for node in queue:
            if dist[node]>=hops: continue
            for nb in self.fwd.get(node,[])+self.bwd.get(node,[]):
                if nb not in dist: dist[nb]=dist[node]+1; queue.append(nb)
        return dist


def build_corpus():
    return [
        Paper("p001","Attention Sparsity","We analyze attention sparsity in transformers.",2020,["p002"],[]),
        Paper("p002","Block Routing","Block selection for efficient routing.",2021,["p003"],["p001"]),
        Paper("p003","Efficient Attention Survey","Survey of efficient attention mechanisms.",2022,[],["p001","p002"]),
    ]


def main():
    papers=build_corpus()
    bm25=BM25Index(); bm25.build(papers)
    graph=CitationGraph(papers)
    query="sparse attention"
    scores=sorted([(p.id,bm25.score(query,p.id)) for p in papers],key=lambda x:-x[1])
    print(f"查询: '{query}'")
    for pid,s in scores:
        if s>0: print(f"  {pid}: {s:.4f}")
    hits=graph.bfs([pid for pid,s in scores if s>0],2)
    print(f"图遍历命中: {list(hits.keys())}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
