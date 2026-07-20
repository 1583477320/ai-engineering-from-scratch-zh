"""交叉编码重排序器。"""
import torch, torch.nn as nn, time, re

class CrossEncoder(nn.Module):
    def __init__(self, vocab=256, dim=64, heads=4):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim); self.type_emb=nn.Embedding(2,dim)
        self.block=nn.TransformerEncoderLayer(dim,heads,dim*4,batch_first=True,norm_first=True)
        self.head=nn.Linear(dim,1)
    def forward(self,ids,tids):
        x=self.emb(ids)+self.type_emb(tids); return self.head(self.block(x)[:,0,:]).squeeze(-1)

def tok_pair(q,d,max_len=32):
    toks=re.findall(r"[a-z0-9]+",(q+" [SEP] "+d).lower())
    ids=[hash(t)%256 for t in toks[:max_len]]
    sep=ids.index(0) if 0 in ids else len(ids)
    return ids,[0]*sep+[1]*(len(ids)-sep)

def pad_batch(pairs):
    ml=max(len(i) for i,_ in pairs)
    return torch.tensor([p[0]+[0]*(ml-len(p[0])) for p in pairs]),torch.tensor([p[1]+[1]*(ml-len(p[1])) for p in pairs])

def rerank(model,q,cands,top_k=5):
    pairs=[tok_pair(q,c) for c in cands]; ids,tids=pad_batch(pairs)
    with torch.no_grad(): s=model(ids,tids)
    ranked=sorted(range(len(cands)),key=lambda i:-s[i].item())
    return [(cands[i],s[i].item()) for i in ranked[:top_k]]

def mock_retrieve(q,corpus,top_n=3):
    return [d for d in corpus if any(w in d.lower() for w in re.findall(r"[a-z]+",q.lower()))][:top_n] or corpus[:top_n]

def main():
    torch.manual_seed(42); model=CrossEncoder()
    corpus=["AbortMultipartOnFail handles upload cancellation.","Large file chunking for reliability.","Retry policy for failed requests."]
    q="what happens when upload fails"
    t0=time.perf_counter(); top_n=mock_retrieve(q,corpus); t1=(time.perf_counter()-t0)*1000
    t0=time.perf_counter(); top_k=rerank(model,q,top_n,2); t2=(time.perf_counter()-t0)*1000
    print(f"查询: '{q}'"); print(f"检索({t1:.1f}ms): {top_n}")
    for d,s in top_k: print(f"  重排序 {s:.4f}: {d[:50]}")
    print(f"重排序({t2:.1f}ms)")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
