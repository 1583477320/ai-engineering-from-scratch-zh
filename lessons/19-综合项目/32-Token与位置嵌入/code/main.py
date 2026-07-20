"""Token与位置嵌入——三个PyTorch模块+合成器。"""
import torch; import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self,V,D): super().__init__(); self.emb=nn.Embedding(V,D)
    def forward(self,x): return self.emb(x)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self,L,D): super().__init__(); self.L=L; self.emb=nn.Embedding(L,D)
    def forward(self,T):
        if T>self.L: raise ValueError(f"{T}>{self.L}")
        return self.emb(torch.arange(T))

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,L,D,base=10000.0):
        super().__init__(); self.L=L
        p=torch.arange(L, dtype=torch.float32).unsqueeze(1)
        i=torch.arange(D//2, dtype=torch.float32)
        a=p/(base**(2*i/D))
        pe=torch.zeros(L,D); pe[:,0::2]=torch.sin(a); pe[:,1::2]=torch.cos(a)
        self.register_buffer("pe",pe)
    def forward(self,T):
        if T>self.L: raise ValueError(f"{T}>{self.L}")
        return self.pe[:T]

class EmbeddingComposer(nn.Module):
    def __init__(self,tok,pos): super().__init__(); self.tok=tok; self.pos=pos
    def forward(self,ids):
        T=ids.shape[1]; return self.tok(ids)+self.pos(T).unsqueeze(0)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def neighbor_cosine(tbl,k=6):
    t=tbl.detach().float(); n=t/t.norm(dim=1,keepdim=True).clamp(min=1e-8)
    return [float((n[:-i]*n[i:]).sum(dim=1).mean()) for i in range(1,k+1)]

def main():
    V,D,L=320,64,128
    tok=TokenEmbedding(V,D); lp=LearnedPositionalEmbedding(L,D); sp=SinusoidalPositionalEmbedding(L,D)
    cl=EmbeddingComposer(tok,lp); cs=EmbeddingComposer(tok,sp)
    ids=torch.randint(0,V,(4,32),dtype=torch.long)
    print(f"形状: {tuple(cl(ids).shape)} 参数: token={count_params(tok)} lp={count_params(lp)} sp={count_params(sp)}")
    lc=neighbor_cosine(lp.emb.weight); sc=neighbor_cosine(sp.pe)
    print(f"邻位余弦(3): 学习={lc[2]:.3f} 正弦={sc[2]:.3f}")
    return 0

if __name__=="__main__": raise SystemExit(main())
