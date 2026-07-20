"""查询重写——HyDE + 多查询 + 分解。"""
import hashlib
from collections import defaultdict

class MockLLM:
    def __init__(self):
        self.hyp={"upload fail":"AbortMultipartOnFail handles upload cancellation and decrements retry budget.","network retry":"Failed network requests are retried with exponential backoff."}
        self.para={"upload fail":["upload failure handling","budget exhausted upload"],"network retry":["request retry policy","network failure recovery"]}
        self.decomp={"upload fail":["what happens when upload fails","what happens when budget is gone"],"network retry":["retry policy for failed requests"]}
    def get_hyp(self,q): return self.hyp.get(q,f"Documentation about {q}")
    def get_paras(self,q,n=2): return self.para.get(q,[f"{q} v{i}" for i in range(n)])
    def get_decomp(self,q): return self.decomp.get(q,[q])

def mock_embed(text,dim=16):
    h=hashlib.sha256(text.encode()).digest(); v=[(h[i%len(h)]/127.5-1.0) for i in range(dim)]
    n=sum(x**2 for x in v)**.5 or 1; return [x/n for x in v]
def cosine(a,b): return sum(x*y for x,y in zip(a,b))

def retrieve(q,corpus,ce,top_k=3):
    qe=mock_embed(q); s=[(i,cosine(qe,e)) for i,e in enumerate(ce)]
    return [(corpus[i],sc) for i,sc in sorted(s,key=lambda x:-x[1])[:top_k]]

def rrf(lists,k=60):
    sc=defaultdict(float)
    for lst in lists:
        for rank,(d,_) in enumerate(lst,1): sc[d]+=1/(k+rank)
    return sorted(sc.items(),key=lambda x:-x[1])

def hyde_retrieve(llm,q,corpus,ce,top_k=3): return retrieve(llm.get_hyp(q),corpus,ce,top_k)
def multi_query_retrieve(llm,q,corpus,ce,top_k=3,n=2):
    return rrf([retrieve(p,corpus,ce,top_k) for p in llm.get_paras(q,n)])[:top_k]
def decompose_retrieve(llm,q,corpus,ce,top_k=3):
    return rrf([retrieve(s,corpus,ce,top_k) for s in llm.get_decomp(q)])[:top_k]

def main():
    llm=MockLLM()
    corpus=["AbortMultipartOnFail aborts multipart upload and decrements budget.","Retry policy retries failed network requests."]
    ce=[mock_embed(d) for d in corpus]
    for q in ["upload fail","network retry"]:
        print(f"查询: '{q}'")
        print(f"  HyDE: {[d[:40] for d,_ in hyde_retrieve(llm,q,corpus,ce,2)]}")
        print(f"  多查询: {[d[:40] for d,_ in multi_query_retrieve(llm,q,corpus,ce,2)]}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
