"""经典指标——BLEU-4 + ROUGE-L + F1 + 精确匹配。"""
import re, math
from collections import Counter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
def tokenize(text): return TOKEN_RE.findall(text.lower())
def exact_match(pred, targets): return float(any(pred.strip()==t.strip() for t in targets))
def f1_score(pred, target):
    p,t=tokenize(pred),tokenize(target)
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    common=Counter(p)&Counter(t); inter=sum(common.values())
    prec=inter/len(p); rec=inter/len(t)
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0
def n_grams(tokens,n): return [tuple(tokens[i:i+n]) for i in range(max(0,len(tokens)-n+1))]
def bleu4(pred,ref,smooth=True):
    ps=[]
    for n in range(1,5):
        png=n_grams(tokenize(pred),n); tng=n_grams(tokenize(ref),n)
        if not png: ps.append(1e-10); continue
        rc=Counter(tng); c=sum(min(png.count(ng),rc.get(ng,0)) for ng in set(png))
        p=c/len(png); ps.append(max(p,1e-10) if smooth else p)
    pl,tl=len(tokenize(pred)),len(tokenize(ref))
    bp=min(1.0,math.exp(1-tl/max(pl,1))) if pl<tl else 1.0
    return bp*math.exp(sum(0.25*math.log(s) for s in ps))
def lcs_length(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1]=dp[i][j]+1 if a[i]==b[j] else max(dp[i+1][j],dp[i][j+1])
    return dp[n][m]
def rouge_l(pred,target):
    p,t=tokenize(pred),tokenize(target)
    if not p or not t: return 0.0
    lcs=lcs_length(p,t); prec=lcs/len(p); rec=lcs/len(t)
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0
def score(m,p,ts):
    if m=="exact_match": return exact_match(p,ts)
    if m=="f1": return max(f1_score(p,t) for t in ts)
    if m=="bleu_4": return max(bleu4(p,t) for t in ts)
    if m=="rouge_l": return max(rouge_l(p,t) for t in ts)
    if m=="accuracy": return float(p.strip().lower()==ts[0].strip().lower()) if ts else 0
    raise ValueError(f"unknown metric: {m}")

def main():
    for m,p,ts in [("exact_match","42",["42"]),("f1","cat on mat",["the cat sat"]),
                   ("bleu_4","the cat sat",["a cat is on the mat"]),
                   ("rouge_l","the cat sat",["the cat is on the mat"])]:
        print(f"  {m}: {score(m,p,ts):.4f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
