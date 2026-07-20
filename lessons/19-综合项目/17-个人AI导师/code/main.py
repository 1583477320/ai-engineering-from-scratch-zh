"""个人AI导师——贝叶斯知识追踪+苏格拉底策略脚手架。"""
from __future__ import annotations
import math, random
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class BKTParams:
    p_init:float=0.2; p_learn:float=0.12; p_slip:float=0.10; p_guess:float=0.15

def bkt_update(mastery, correct, p):
    if correct:
        num=mastery*(1-p.p_slip); denom=num+(1-mastery)*p.p_guess
    else:
        num=mastery*p.p_slip; denom=num+(1-mastery)*(1-p.p_guess)
    post=num/max(denom,1e-6)
    return post+(1-post)*p.p_learn

@dataclass
class Concept:
    name:str; prereqs:list[str]=field(default_factory=list); difficulty:float=0.3

ALGEBRA=[
    Concept("数轴",[],0.1),Concept("加法与减法",["数轴"],0.2),
    Concept("乘法与除法",["加法与减法"],0.35),Concept("负数",["加法与减法"],0.4),
    Concept("等式",["加法与减法"],0.3),Concept("一步变量隔离",["等式","加法与减法"],0.45),
    Concept("两步变量隔离",["一步变量隔离","乘法与除法"],0.6),
    Concept("分配律",["乘法与除法"],0.4),Concept("合并同类项",["加法与减法","分配律"],0.5),
    Concept("线性方程",["两步变量隔离","合并同类项"],0.65),
    Concept("二次方程基础",["线性方程","乘法与除法"],0.75),
]

@dataclass
class LearnerState:
    learner_id:str; mastery:dict[str,float]=field(default_factory=lambda:defaultdict(lambda:0.2))
    history:list[tuple[str,bool]]=field(default_factory=list)

def curriculum_map(c): return {x.name:x for x in c}

def next_concept(state,cmap,thresh=0.85):
    for c in cmap.values():
        if state.mastery[c.name]>=thresh: continue
        if all(state.mastery[p]>=thresh for p in c.prereqs): return c.name
    return None

def socratic_policy(state,concept,correct):
    m=state.mastery[concept]
    if correct and m>0.8: return "celebrate_and_advance"
    if correct: return "reinforce"
    if m>0.5: return "hint"
    return "scaffold"

def simulate_answer(ek,diff,rng):
    return rng.random()<1/(1+math.exp(-(ek-diff)))

def run_adaptive(lid,ability,cmap,n,rng):
    s=LearnerState(learner_id=lid); p=BKTParams(); last=None
    for _ in range(n):
        c=next_concept(s,cmap)
        if c is None: break
        d=cmap[c].difficulty
        if last=="scaffold": d-=0.15
        elif last=="hint": d-=0.08
        elif last=="celebrate_and_advance": s.mastery[c]=min(1,s.mastery[c]+0.02)
        ek=ability+s.mastery[c]*1.5; ok=simulate_answer(ek,d,rng)
        last=socratic_policy(s,c,ok); s.history.append((c,ok))
        s.mastery[c]=bkt_update(s.mastery[c],ok,p)
    return s

def run_baseline(lid,ability,cmap,n,rng):
    s=LearnerState(learner_id=lid); p=BKTParams(); order=list(cmap.keys())
    for i in range(n):
        c=order[i%len(order)]; d=cmap[c].difficulty
        ek=ability+s.mastery[c]*1.5; ok=simulate_answer(ek,d,rng)
        s.history.append((c,ok)); s.mastery[c]=bkt_update(s.mastery[c],ok,p)
    return s

def mastery_sum(s,cmap): return sum(s.mastery[c] for c in cmap)

def main():
    cmap=curriculum_map(ALGEBRA); rng=random.Random(29)
    print(f"=== 两周效果研究 === 课程: {len(cmap)}概念")
    ag,bg=[],[]
    for i in range(10):
        a=rng.gauss(0.3,0.4); seed=100+i
        ra=random.Random(seed); rb=random.Random(); rb.setstate(ra.getstate())
        ag.append(mastery_sum(run_adaptive(f"a{i}",a,cmap,60,ra),cmap))
        bg.append(mastery_sum(run_baseline(f"b{i}",a,cmap,60,rb),cmap))
    m=lambda x:sum(x)/len(x)
    print(f"自适应: {m(ag):.2f}  基线: {m(bg):.2f}  差值: {m(ag)-m(bg):+.2f}")

if __name__=="__main__": main()
