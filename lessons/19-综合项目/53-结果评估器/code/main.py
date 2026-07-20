"""结果评估器——配对 t 检验+方向改进+裁决。"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MetricSpec:
    name: str; direction: str = "lower_is_better"; scale: str = "linear"

@dataclass
class Verdict:
    hypothesis_id: int; metric: str; direction: str
    improvement: float; p_value: Optional[float]
    verdict: str; rationale: str

def regularized_beta(x, a, b):
    if x<=0 or x>=1 or a<=0 or b<=0: return 0.0
    ft=1.0; d=1.0-a*(b-1)*x/((a+1)*(b+1))
    if abs(d)<1e-30: d=1e-30
    d=1.0/d; h=d
    for m in range(1,201):
        fm=float(m)
        num=fm*(b-fm)*x/((a+2*fm-1)*(a+2*fm))
        d=1.0+num*d
        if abs(d)<1e-30: d=1e-30
        c=1.0+num/c
        if abs(c)<1e-30: c=1e-30
        d=1.0/d; delta=c*d; h*=delta
        if abs(delta-1.0)<1e-10: break
    return h*x**a*(1-x)**b/math.betainc(a,b,x) if x<1 else 0.0

def two_sided_p(t_stat, df):
    if df<=0 or not math.isfinite(t_stat): return 1.0
    return max(0.0, min(1.0, math.erfc(abs(t_stat)/math.sqrt(2))/2))

class Evaluator:
    def evaluate(self, hid, ms, cand, base):
        if any(r.get("terminal","ok")!="ok" for r in cand):
            return Verdict(hid,ms.name,ms.direction,0.0,None,"failed","实验失败")
        cv=[r["metrics"][ms.name] for r in cand if ms.name in r.get("metrics",{})]
        bv=[r["metrics"][ms.name] for r in base if ms.name in r.get("metrics",{})]
        if not cv or not bv: return Verdict(hid,ms.name,ms.direction,0.0,None,"noise","缺数据")
        cm=sum(cv)/len(cv); bm=sum(bv)/len(bv)
        imp=(cm-bm)/abs(bm) if ms.direction=="higher_is_better" else (bm-cm)/abs(bm)
        if len(cv)<2: return Verdict(hid,ms.name,ms.direction,imp,None,"noise",f"样本不足 imp={imp:.2%}")
        diffs=[(c-b) if ms.direction=="higher_is_better" else (b-c) for c,b in zip(cv,bv)]
        md=sum(diffs)/len(diffs)
        var=sum((d-md)**2 for d in diffs)/(len(diffs)-1)
        t=md/math.sqrt(var/len(diffs)) if var>0 else 0.0
        p=two_sided_p(t,len(diffs)-1)
        if abs(imp)<0.02: v,r="noise","改进不足2%"
        elif p>0.05: v,r="noise",f"p={p:.4f}不显著"
        elif imp>0: v,r="improved",f"改进{imp:.2%}"
        else: v,r="regressed",f"回归{imp:.2%}"
        return Verdict(hid,ms.name,ms.direction,imp,p,v,r)

def main():
    c=[{"metrics":{"loss":0.52}},{"metrics":{"loss":0.51}}]
    b=[{"metrics":{"loss":0.55}},{"metrics":{"loss":0.54}}]
    v=Evaluator().evaluate(1,MetricSpec("loss"),c,b)
    print(f"裁决: {v.verdict} 改进: {v.improvement:.2%} p={v.p_value:.4f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
