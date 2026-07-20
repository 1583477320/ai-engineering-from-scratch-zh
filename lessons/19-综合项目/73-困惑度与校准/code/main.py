"""困惑度与校准——ECE + Brier + 可靠性图。"""
import math

def perplexity(nlls,counts):
    t=sum(counts); return math.exp(sum(nlls)/t) if t>0 else float("nan")

def ece(conf,correct,bins=10):
    if not conf: return 0.0,0
    edges=[i/bins for i in range(bins+1)]; e,pop=0.0,0
    for i in range(bins):
        ib=[(p,y) for p,y in zip(conf,correct) if edges[i]<=p<edges[i+1] or (i==bins-1 and p==edges[i+1])]
        if not ib: continue
        ac=sum(p for p,_ in ib)/len(ib); aa=sum(y for _,y in ib)/len(ib)
        e+=len(ib)/len(conf)*abs(ac-aa); pop+=1
    return e,pop

def brier(conf,correct):
    return sum((p-y)**2 for p,y in zip(conf,correct))/max(len(conf),1)

def rel_diagram(conf,correct,bins=10):
    edges=[i/bins for i in range(bins+1)]; mc,ma,co=[],[],[]
    for i in range(bins):
        ib=[(p,y) for p,y in zip(conf,correct) if edges[i]<=p<edges[i+1] or (i==bins-1 and p==edges[i+1])]
        mc.append(sum(p for p,_ in ib)/len(ib) if ib else 0)
        ma.append(sum(y for _,y in ib)/len(ib) if ib else 0)
        co.append(len(ib))
    return mc,ma,co

def main():
    print(f"困惑度: {perplexity([0.1,0.5,0.3],[5,5,5]):.3f}")
    conf=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.9]
    corr=[1,1,1,0,0,0,0,0,0,1]
    e,p=ece(conf,corr,5); print(f"ECE: {e:.3f} (填充桶:{p})")
    print(f"Brier: {brier(conf,corr):.3f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
