"""RAG 评估——P@K + R@K + MRR + nDCG@K。"""
import math
def precision_at_k(ret,gold,k): return sum(1 for d in ret[:k] if d in gold)/max(len(ret[:k]),1)
def recall_at_k(ret,gold,k): return sum(1 for d in ret[:k] if d in gold)/max(len(gold),1)
def mrr(results,golds):
    s=0
    for ret,gold in zip(results,golds):
        for rank,d in enumerate(ret,1):
            if d in gold: s+=1/rank; break
    return s/max(len(results),1)
def ndcg_at_k(ret,graded,k):
    dcg=sum(graded.get(d,0)/math.log2(i+2) for i,d in enumerate(ret[:k]))
    idcg=sum(g/math.log2(i+2) for i,g in enumerate(sorted(graded.values(),reverse=True)[:k]))
    return dcg/idcg if idcg>0 else 0

def main():
    qrels=[{"gold":["d1","d3"],"graded":{"d1":3,"d3":2}},
           {"gold":["d3","d4"],"graded":{"d3":3,"d4":1}},
           {"gold":["d2"],"graded":{"d2":3}}]
    results=[["d1","d3","d2"],["d3","d4","d1"],["d2","d1","d3"]]
    p5=[precision_at_k(r,q["gold"],5) for r,q in zip(results,qrels)]
    r5=[recall_at_k(r,q["gold"],5) for r,q in zip(results,qrels)]
    m=mrr(results,[q["gold"] for q in qrels])
    n=[ndcg_at_k(r,q["graded"],5) for r,q in zip(results,qrels)]
    print(f"P@5={sum(p5)/len(p5):.3f} R@5={sum(r5)/len(r5):.3f} MRR={m:.3f} nDCG@5={sum(n)/len(n):.3f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
