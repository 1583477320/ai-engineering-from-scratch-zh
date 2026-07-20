"""评审循环——5维评分+收敛检测。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Critique:
    scores: Dict[str,float]; round: int

@dataclass
class LoopResult:
    convergence: str; rounds: int; trace: List[Dict]

class MiniPaper:
    def __init__(self,abstract="",sections=None,originality="medium",figures=None):
        self.abstract=abstract; self.sections=sections or {"intro":"","method":"","results":""}
        self.originality=originality; self.figures=figures or []
    def copy(self): return MiniPaper(self.abstract,dict(self.sections),self.originality,list(self.figures))

def critic(draft,rnd):
    avg=sum(len(v) for v in draft.sections.values())/max(len(draft.sections),1)
    s={"clarity":min(10,avg/15),"novelty":{"high":9,"medium":6,"low":3}.get(draft.originality,5),
       "evidence":min(10,len(draft.figures)*2+2),"methodology":7 if draft.sections.get("method") else 2,
       "related_work":6 if any("related" in k for k in draft.sections) else 2}
    return Critique(s,rnd)

class CriticLoop:
    def __init__(self,ct=None,rw=None,max_r=5,target=8.0,eps=0.1):
        self.ct=ct or critic; self.rw=rw; self.max_r=max_r; self.target=target; self.eps=eps
    def run(self,draft):
        trace=[]; prev=0.0; pc=0
        for rnd in range(1,self.max_r+1):
            cr=self.ct(draft,rnd); ms=sum(cr.scores.values())/len(cr.scores)
            trace.append({"round":rnd,"scores":cr.scores})
            if all(s>=self.target for s in cr.scores.values()): return LoopResult("target",rnd,trace)
            if rnd>1 and (ms-prev)<self.eps: pc+=1
            else: pc=0
            if pc>=2: return LoopResult("plateau",rnd,trace)
            prev=ms
        return LoopResult("budget",self.max_r,trace)

def main():
    d=MiniPaper("稀疏性",{"intro":"引言","method":"","results":""},"medium")
    r=CriticLoop().run(d)
    print(f"收敛: {r.convergence} 轮: {r.rounds}")
    for t in r.trace: print(f"  轮{t['round']}: scores={t['scores']}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
