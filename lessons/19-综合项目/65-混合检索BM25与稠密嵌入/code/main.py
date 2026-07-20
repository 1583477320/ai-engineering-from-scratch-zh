"""混合检索——BM25 + 稠密 + RRF。"""
import math, re, hashlib
from collections import Counter, defaultdict

def tokenize(text): return re.findall(r"[a-z0-9_]+",text.lower())

class BM25Index:
    def __init__(self,k1=1.5,b=0.75):
        self.k1=k1; self.b=b; self.df=Counter(); self.td=defaultdict(list); self.dl={}; self.N=0
    def add(self,did,text):
        tokens=tokenize(text); self.dl[did]=len(tokens); self.N+=1
        for t,c in Counter(tokens).items(): self.df[t]+=1; self.td[t].append((did,c))
    def search(self,query,top_k=5):
        avgdl=sum(self.dl.values())/max(self.N,1); qt=tokenize(query); s={}
        for did in set(d for t in qt for d,_ in self.td.get(t,[])):
            sc=0; dl=self.dl.get(did,0)
            for t in set(qt):
                df=self.df.get(t,0); idf=math.log((self.df.get(t,0)+0.5)/(max(self.N,1)-self.df.get(t,0)+0.5)+1)
                f=sum(c for d,c in self.td.get(t,[]) if d==did)
                sc+=idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/max(avgdl,1)))
            if sc>0: s[did]=sc
        return sorted(s.items(),key=lambda x:-x[1])[:top_k]

def mock_embed(text,dim=64):
    h=hashlib.sha256(text.encode()).digest(); v=[0.0]*dim
    for i in range(dim): v[i]=(h[i%len(h)]/127.5-1.0)
    n=sum(x**2 for x in v)**.5 or 1; return [x/n for x in v]

def rrf(ranked,k=60):
    sc=defaultdict(float)
    for mod,ranked_list in ranked.items():
        for rank,(did,_) in enumerate(ranked_list,1): sc[did]+=1/(k+rank)
    return sorted(sc.items(),key=lambda x:-x[1])

def main():
    bm25=BM25Index()
    corpus=[("d1","AbortMultipartOnFail handles upload cancellation"),("d2","Large file chunking for reliability"),("d3","Retry policy for failed requests")]
    for did,text in corpus: bm25.add(did,text)
    for q in ["AbortMultipartOnFail","handle cancelled uploads","retry logic"]:
        br=bm25.search(q,3); dr=[(did,0) for did,_ in br]  # simplified
        print(f"查询: '{q}' BM25: {br} RRF: {rrf({'bm25':[(d,i+1) for i,(d,_) in enumerate(br)],'dense':dr})[:3]}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
