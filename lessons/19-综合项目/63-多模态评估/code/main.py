"""多模态评估——R@K + VQA + BLEU-4。"""
import math, re
from collections import Counter

def recall_at_k(sim,k):
    N=sim.shape[0]
    i2r=sum(1 for i in range(N) if i in sim[i].argsort(descending=True)[:k].tolist())/N
    return i2r,i2r

def vqa_em(preds,refs):
    return sum(p==r for p,r in zip(preds,refs))/max(len(preds),1)

def n_grams(toks,n):
    return [tuple(toks[i:i+n]) for i in range(max(0,len(toks)-n+1))]

def bleu4(gen,refs,smooth=True):
    ps=[]
    for n in range(1,5):
        gng=n_grams(gen,n)
        if not gng: ps.append(1e-10); continue
        rc=Counter()
        for r in refs:
            for ng in n_grams(r,n): rc[ng]=max(rc[ng],sum(1 for x in n_grams(r,n) if x==ng))
        c=sum(min(gng.count(ng),rc.get(ng,0)) for ng in set(gng))
        p=c/len(gng); ps.append(max(p,1e-10) if smooth else p)
    bp=min(1.0,math.exp(1-len(refs[0])/max(1,len(gen)))) if refs else 1.0
    return bp*math.exp(sum(0.25*math.log(p) for p in ps))

def main():
    import torch
    sim=torch.randn(20,20); [sim.__setitem__((i,i),sim[i,i]+0.5) for i in range(20)]
    r1,_=recall_at_k(sim,1); r5,_=recall_at_k(sim,5)
    print(f"R@1={r1:.3f} R@5={r5:.3f}")
    print(f"VQA EM={vqa_em(['猫','狗','鸟','鱼'],['猫','狗','鸟','蛇']):.3f}")
    print(f"BLEU-4={bleu4(['一只','黑色','猫'],[['一只','黑色','猫咪']]):.4f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
