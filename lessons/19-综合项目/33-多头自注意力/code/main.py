"""多头自注意力——融合QKV投影+因果掩码+权重检查。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,max_ctx,attn_drop=0.0,out_drop=0.0):
        super().__init__()
        assert d_model%n_heads==0
        self.d_model=d_model; self.n_heads=n_heads; self.d_head=d_model//n_heads; self.max_ctx=max_ctx
        self.qkv_proj=nn.Linear(d_model,3*d_model); self.out_proj=nn.Linear(d_model,d_model)
        self.attn_drop=nn.Dropout(attn_drop); self.out_drop=nn.Dropout(out_drop)
        self.register_buffer("causal_mask",torch.tril(torch.ones(max_ctx,max_ctx)),persistent=False)
    def _split_heads(self,x):
        b,t,_=x.shape; return x.view(b,t,self.n_heads,self.d_head).transpose(1,2)
    def _merge_heads(self,x):
        b,h,t,dh=x.shape; return x.transpose(1,2).contiguous().view(b,t,h*dh)
    def forward(self,x,return_weights=False):
        b,t,d=x.shape
        q,k,v=self.qkv_proj(x).chunk(3,dim=-1)
        q,k,v=self._split_heads(q),self._split_heads(k),self._split_heads(v)
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_head)
        scores=scores.masked_fill(self.causal_mask[:t,:t]==0,float("-inf"))
        weights=self.attn_drop(F.softmax(scores,dim=-1))
        context=torch.matmul(weights,v)
        out=self.out_drop(self.out_proj(self._merge_heads(context)))
        return (out,weights) if return_weights else out

class TokenEmbedding(nn.Module):
    def __init__(self,V,D):
        super().__init__(); self.emb=nn.Embedding(V,D)
        with torch.no_grad(): self.emb.weight.normal_(0,0.02)
    def forward(self,ids): return self.emb(ids)

class SinPosEmbed(nn.Module):
    def __init__(self,L,D):
        super().__init__(); pe=torch.zeros(L,D)
        pos=torch.arange(L,dtype=torch.float32).unsqueeze(1); i=torch.arange(D//2,dtype=torch.float32)
        a=pos/(10000.0**(2*i/D)); pe[:,0::2]=torch.sin(a); pe[:,1::2]=torch.cos(a)
        self.register_buffer("pe",pe)
    def forward(self,T): return self.pe[:T]

class TinyAttentionLM(nn.Module):
    def __init__(self,V,D,H,L):
        super().__init__()
        self.tok_emb=TokenEmbedding(V,D); self.pos_emb=SinPosEmbed(L,D)
        self.attn=MultiHeadSelfAttention(D,H,L); self.lm_head=nn.Linear(D,V)
    def forward(self,ids,return_weights=False):
        tok=self.tok_emb(ids); pos=self.pos_emb(ids.shape[1]); x=tok+pos.unsqueeze(0)
        if return_weights:
            attn_out,weights=self.attn(x,True); return self.lm_head(attn_out),weights
        return self.lm_head(self.attn(x))

def main():
    V,D,H,L,B=64,32,4,12,16; attn=MultiHeadSelfAttention(D,H,L)
    x=torch.randn(B,L,D); out=attn(x)
    print(f"形状: input={tuple(x.shape)} output={tuple(out.shape)}")
    _,weights=attn(x,True); print(f"权重形状: {tuple(weights.shape)}")
    upper=torch.triu(torch.ones(L,L),diagonal=1).bool()
    print(f"未来位置权重和: {weights[0,0][upper].abs().sum().item():.6f}")
    model=TinyAttentionLM(V,D,H,L)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-3)
    for epoch in range(3):
        base=torch.randint(0,V,(B,1)); ids=base.expand(B,L+1).contiguous()
        loss=F.cross_entropy(model(ids[:,:-1]).reshape(-1,V),ids[:,1:].reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")
    print("Demo OK.")
    return 0

if __name__=="__main__": raise SystemExit(main())
