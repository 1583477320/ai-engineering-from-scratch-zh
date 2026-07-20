"""端到端 RAG 系统。"""
import hashlib, time, re
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text:str; doc_id:str; chunk_idx:int; source:str=""

def mock_embed(text,dim=32):
    h=hashlib.sha256(text.encode()).digest(); v=[(h[i%len(h)]/127.5-1.0) for i in range(dim)]
    n=sum(x**2 for x in v)**.5 or 1; return [x/n for x in v]
def cosine(a,b): return sum(x*y for x,y in zip(a,b))

class HybridRetriever:
    def __init__(self,chunks): self.chunks=chunks; self.embeds=[mock_embed(c.text) for c in chunks]
    def retrieve(self,q,top_k=5):
        qe=mock_embed(q); s=sorted(range(len(self.chunks)),key=lambda i:-cosine(qe,self.embeds[i]))
        return [self.chunks[i] for i in s[:top_k]]

class MockReranker:
    def rerank(self,q,chunks,top_k=3):
        return sorted(chunks,key=lambda c:-len(set(q.lower().split())&set(c.text.lower().split())))[:top_k]

class MockGenerator:
    def generate(self,q,chunks):
        if not chunks: return "我不知道。",[]
        sents=re.split(r'(?<=[.!?])\s*',chunks[0].text)
        ans=(sents[0] if sents else chunks[0].text)+f" [{chunks[0].doc_id}:{chunks[0].chunk_idx}]"
        return ans,[f"{chunks[0].doc_id}:{chunks[0].chunk_idx}"]

@dataclass
class Result:
    answer:str; citations:List[str]; top_k:List[Chunk]; latency_ms:float

class Pipeline:
    def __init__(self,chunks): self.retriever=HybridRetriever(chunks); self.reranker=MockReranker(); self.gen=MockGenerator()
    def query(self,q):
        t=time.perf_counter()
        cands=self.retriever.retrieve(q,5); ranked=self.reranker.rerank(q,cands,3)
        ans,cites=self.gen.generate(q,ranked)
        return Result(ans,cites,ranked,(time.perf_counter()-t)*1000)

def main():
    corpus=[Chunk("AbortMultipartOnFail aborts multipart upload and decrements budget.","d1",0),
            Chunk("Large file chunking splits uploads for reliability.","d2",0),
            Chunk("Retry policy retries failed network requests.","d3",0)]
    pipe=Pipeline(corpus)
    for q in ["abort threshold","upload failure"]:
        r=pipe.query(q); print(f"查询:'{q}' 答案:{r.answer[:60]} 延迟:{r.latency_ms:.2f}ms")
    print("✓ 端到端完成")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
