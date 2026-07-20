"""排行榜聚合——均值+胜率+bootstrap CI。"""
import random
from collections import defaultdict
from dataclasses import dataclass,field
from typing import List,Dict

@dataclass
class EvalRun:
    model_id:str; task_id:str; metric_name:str; score:float; category:str=""

def bootstrap_mean_ci(scores,b=500,alpha=0.05):
    if not scores: return 0,0,0
    means=sorted([sum(random.choices(scores,k=len(scores)))/len(scores) for _ in range(b)])
    return sum(scores)/len(scores),means[int(b*alpha/2)],means[int(b*(1-alpha/2))]

def win_rate(mid,by_task,all_m):
    w,t=0,0
    for tid,runs in by_task.items():
        sc={r.model_id:r.score for r in runs}
        if mid not in sc: continue
        t+=1
        best=max(sc[m] for m in all_m if m in sc)
        if sc[mid]>=best: w+=1
    return w/t if t else 0

def aggregate(runs,b=200):
    by_model=defaultdict(list); by_task=defaultdict(list)
    for r in runs: by_model[r.model_id].append(r); by_task[r.task_id].append(r)
    all_m=list(by_model.keys())
    rows=[]
    for mid,mruns in by_model.items():
        scores=[r.score for r in mruns]; mean,lo,hi=bootstrap_mean_ci(scores,b)
        wr=win_rate(mid,by_task,all_m)
        cats=defaultdict(list)
        for r in mruns: cats[r.category].append(r.score) if r.category else None
        cm={c:sum(s)/len(s) for c,s in cats.items()}
        rows.append({"model_id":mid,"mean":round(mean,3),"ci":f"{lo:.3f}-{hi:.3f}","win_rate":round(wr,2),"n":len(mruns),"cats":cm})
    rows.sort(key=lambda r:-r["mean"]); return rows

def render_md(rows):
    lines=["| Rank | Model | Mean | 95% CI | Win Rate | Tasks |","|------|-------|------|--------|----------|-------|"]
    for i,r in enumerate(rows,1):
        lines.append(f"| {i} | {r['model_id'][:20]} | {r['mean']} | {r['ci']} | {r['win_rate']} | {r['n']} |")
    return "\n".join(lines)

def main():
    random.seed(42); runs=[]
    base={"gpt":0.78,"claude":0.75,"random":0.10}
    for mid in base:
        for i in range(20):
            runs.append(EvalRun(mid,f"t{i}","exact_match",min(1,max(0,base[mid]+random.gauss(0,.1))),random.choice(["math","code","reasoning"])))
    rows=aggregate(runs,200)
    print(render_md(rows))
    return 0
if __name__=="__main__": import sys; sys.exit(main())
