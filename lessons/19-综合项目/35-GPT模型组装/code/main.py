"""GPT模型组装——12块+token嵌入+位置嵌入+最终LN+权重绑定LM头。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,D,eps=1e-5):
        super().__init__(); self.eps=eps; self.scale=nn.Parameter(torch.ones(D)); self.shift=nn.Parameter(torch.zeros(D))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True); var=x.var(dim=-1,keepdim=True,unbiased=False)
        return self.scale*(x-mean)/torch.sqrt(var+self.eps)+self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        D=cfg.d_model; self.H=cfg.num_heads; self.head_dim=D//self.H; self.ctx=cfg.context_length
        self.qkv=nn.Linear(D,3*D,bias=cfg.use_bias); self.out_proj=nn.Linear(D,D,bias=cfg.use_bias)
        self.attn_drop=nn.Dropout(cfg.dropout); self.res_drop=nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask",torch.triu(torch.ones(self.ctx,self.ctx,dtype=torch.bool),diagonal=1),persistent=False)
    def forward(self,x):
        B,T,D=x.shape; q,k,v=self.qkv(x).split(D,dim=-1)
        q=q.view(B,T,self.H,self.head_dim).transpose(1,2); k=k.view(B,T,self.H,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.H,self.head_dim).transpose(1,2)
        scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        scores=scores.masked_fill(self.causal_mask[:T,:T],float("-inf"))
        attn=self.attn_drop(F.softmax(scores,dim=-1)); out=(attn@v).transpose(1,2).contiguous().view(B,T,D)
        return self.res_drop(self.out_proj(out))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__(); D=cfg.d_model; H=cfg.mlp_expansion*D
        self.fc1=nn.Linear(D,H,bias=cfg.use_bias); self.act=nn.GELU(approximate="tanh")
        self.fc2=nn.Linear(H,D,bias=cfg.use_bias); self.drop=nn.Dropout(cfg.dropout)
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.ln1=LayerNorm(cfg.d_model); self.attn=MultiHeadAttention(cfg)
        self.ln2=LayerNorm(cfg.d_model); self.mlp=FeedForward(cfg)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))

@dataclass
class GPTConfig:
    vocab_size:int=50257; context_length:int=1024; d_model:int=768
    num_heads:int=12; num_layers:int=12; mlp_expansion:int=4
    dropout:float=0.1; use_bias:bool=True; weight_tying:bool=True

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok_embed=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.pos_embed=nn.Embedding(cfg.context_length,cfg.d_model)
        self.embed_drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln=LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        if cfg.weight_tying: self.lm_head.weight=self.tok_embed.weight
        self.register_buffer("pos_ids",torch.arange(cfg.context_length,dtype=torch.long),persistent=False)
        self.apply(self._init_weights)
        for b in self.blocks: b.attn.out_proj.weight.data.mul_(1/math.sqrt(2*cfg.num_layers)); b.mlp.fc2.weight.data.mul_(1/math.sqrt(2*cfg.num_layers))
    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,tokens):
        B,T=tokens.shape; tok=self.tok_embed(tokens); pos=self.pos_embed(self.pos_ids[:T])
        x=self.embed_drop(tok+pos)
        for b in self.blocks: x=b(x)
        return self.lm_head(self.final_ln(x))

def count_unique(m):
    seen={}; [seen.update({id(p):p.numel()}) for p in m.parameters()]
    return sum(seen.values())

def top_k_filter(logits,k):
    if not k: return logits
    thresh=torch.topk(logits,k,dim=-1)[0][...,-1:]
    return torch.where(logits<thresh,torch.full_like(logits,float("-inf")),logits)

def generate(model,prompt,max_new,t=1.0,top_k=None):
    model.eval(); tokens=prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits=model(tokens[:,-model.cfg.context_length:])
            next_l=logits[:,-1,:]/t
            if top_k: next_l=top_k_filter(next_l,top_k)
            tokens=torch.cat([tokens,torch.multinomial(F.softmax(next_l,-1),1)],1)
    return tokens

def main():
    torch.manual_seed(0)
    cfg=GPTConfig(); m=GPTModel(cfg)
    print(f"参考124M参数: {count_unique(m):,}")
    head_tied=m.lm_head.weight.data_ptr()==m.tok_embed.weight.data_ptr()
    print(f"权重绑定: {head_tied}")
    tiny=GPTConfig(vocab_size=512,context_length=64,d_model=64,num_heads=4,num_layers=2,dropout=0)
    tm=GPTModel(tiny); print(f"小型模型参数: {count_unique(tm):,}")
    gen=generate(tm,torch.tensor([[1,2,3,4,5]]),12,t=0.8,top_k=20,seed=42)
    print(f"生成: {gen.tolist()[0]}")
    print("Demo OK.")

if __name__=="__main__": raise SystemExit(main())
