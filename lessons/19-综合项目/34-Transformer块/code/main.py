"""Transformer块——LayerNorm+多头注意力+残差+MLP+残差（Pre-LN和Post-LN）。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,D,eps=1e-5):
        super().__init__(); self.eps=eps
        self.scale=nn.Parameter(torch.ones(D)); self.shift=nn.Parameter(torch.zeros(D))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True); var=x.var(dim=-1,keepdim=True,unbiased=False)
        return self.scale*(x-mean)/torch.sqrt(var+self.eps)+self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self,D,H,L,attn_drop=0.0,res_drop=0.0):
        super().__init__()
        self.D=D; self.H=H; self.head_dim=D//H; self.ctx=L
        self.qkv=nn.Linear(D,3*D); self.out_proj=nn.Linear(D,D)
        self.attn_drop=nn.Dropout(attn_drop); self.res_drop=nn.Dropout(res_drop)
        mask=torch.triu(torch.ones(L,L,dtype=torch.bool),diagonal=1)
        self.register_buffer("causal_mask",mask,persistent=False)
    def forward(self,x):
        B,T,D=x.shape; q,k,v=self.qkv(x).split(D,dim=-1)
        q=q.view(B,T,self.H,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.H,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.H,self.head_dim).transpose(1,2)
        scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        scores=scores.masked_fill(self.causal_mask[:T,:T],float("-inf"))
        attn=self.attn_drop(F.softmax(scores,dim=-1))
        out=(attn@v).transpose(1,2).contiguous().view(B,T,D)
        return self.res_drop(self.out_proj(out))

class FeedForward(nn.Module):
    def __init__(self,D,expansion=4,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(D,expansion*D); self.act=nn.GELU(approximate="tanh")
        self.fc2=nn.Linear(expansion*D,D); self.drop=nn.Dropout(drop)
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self,D=768,H=12,L=1024,expansion=4,attn_drop=0.1,res_drop=0.1,pre_ln=True):
        super().__init__()
        self.pre_ln=pre_ln; self.ln1=LayerNorm(D); self.attn=MultiHeadAttention(D,H,L,attn_drop,res_drop)
        self.ln2=LayerNorm(D); self.mlp=FeedForward(D,expansion,res_drop)
    def forward(self,x):
        if self.pre_ln: return x+self.attn(self.ln1(x)), x+self.mlp(self.ln2(x))  # bug: chained incorrectly
        return self.ln1(x+self.attn(x)), self.ln2(x+self.mlp(x))

# Fixed version
class TransformerBlock(nn.Module):
    def __init__(self,D=768,H=12,L=1024,expansion=4,attn_drop=0.1,res_drop=0.1,pre_ln=True):
        super().__init__()
        self.pre_ln=pre_ln; self.ln1=LayerNorm(D); self.attn=MultiHeadAttention(D,H,L,attn_drop,res_drop)
        self.ln2=LayerNorm(D); self.mlp=FeedForward(D,expansion,res_drop)
    def forward(self,x):
        if self.pre_ln:
            x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))
        return self.ln2(x+self.mlp(self.ln1(x+self.attn(x))))

class BlockStack(nn.Module):
    def __init__(self,D=192,H=6,L=64,depth=6,pre_ln=True):
        super().__init__()
        self.embed=nn.Embedding(128,D); self.blocks=nn.ModuleList([TransformerBlock(D,H,L,pre_ln=pre_ln) for _ in range(depth)])
        self.final_ln=LayerNorm(D)
    def forward(self,tokens):
        x=self.embed(tokens)
        for block in self.blocks: x=block(x)
        return self.final_ln(x)

def grad_norm(stack,tokens):
    stack.zero_grad(set_to_none=True); out=stack(tokens); out.pow(2).sum().backward()
    g=stack.embed.weight.grad; return float(g.norm().item()) if g is not None else 0.0

def main():
    torch.manual_seed(0)
    cfg=dict(D=192,H=6,L=64,depth=6,attn_drop=0.0,res_drop=0.0)
    pre=BlockStack(pre_ln=True,**cfg); post=BlockStack(pre_ln=False,**cfg)
    post.load_state_dict(pre.state_dict()); pre.eval(); post.eval()
    tokens=torch.randint(0,128,(2,32))
    print(f"Pre-LN shape: {tuple(pre(tokens).shape)}  Post-LN shape: {tuple(post(tokens).shape)}")
    pg=grad_norm(pre,tokens); postg=grad_norm(post,tokens)
    print(f"Pre-LN  grad: {pg:.6f}\nPost-LN grad: {postg:.6f}")
    if postg>0: print(f"ratio: {pg/postg:.2f}x")
    print(f"params: {sum(p.numel() for p in pre.parameters()):,}")
    return 0

if __name__=="__main__": raise SystemExit(main())
