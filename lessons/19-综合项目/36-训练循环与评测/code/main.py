"""训练循环与评测——AdamW+权重衰减分离+余弦预热+JSONL日志。"""
import json, math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

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
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok_embed=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.pos_embed=nn.Embedding(cfg.context_length,cfg.d_model)
        self.embed_drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln=LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
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

def build_param_groups(model,wd):
    decay,no_decay=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim()<2 or name.endswith(".bias") or name.endswith(".shift") or name.endswith(".scale"): no_decay.append(p)
        else: decay.append(p)
    return [{"params":decay,"weight_decay":wd},{"params":no_decay,"weight_decay":0}]

def cosine_warmup(step,warmup,total,max_lr,min_lr):
    if step<warmup: return max_lr*(step+1)/max(warmup,1)
    prog=min(max((step-warmup)/max(total-warmup,1),0),1)
    return min_lr+(max_lr-min_lr)*0.5*(1+math.cos(math.pi*prog))

def calc_loss_batch(model,inputs,targets):
    logits=model(inputs); return F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))

@torch.no_grad()
def evaluate(model,val_loader,n):
    was=model.training; model.eval(); total=0.0; count=0
    for inp,tgt in val_loader:
        if count>=n: break
        total+=calc_loss_batch(model,inp,tgt).item(); count+=1
    if was: model.train()
    return total/max(count,1)

def make_batches(ids,bs,ctx,seed=0):
    gen=torch.Generator().manual_seed(seed); mx=ids.numel()-ctx-1
    while True:
        starts=torch.randint(0,mx+1,(bs,),generator=gen)
        yield (torch.stack([ids[s:s+ctx] for s in starts.tolist()]),
               torch.stack([ids[s+1:s+1+ctx] for s in starts.tolist()]))

def generate(model,prompt,max_new,t=1.0,top_k=20,seed=0):
    was=model.training; model.eval(); tokens=prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits=model(tokens[:,-model.cfg.context_length:])[:,-1,:]/t
            if top_k>0:
                thresh=torch.topk(logits,top_k,dim=-1)[0][...,-1:]
                logits=torch.where(logits<thresh,torch.full_like(logits,float("-inf")),logits)
            tokens=torch.cat([tokens,torch.multinomial(F.softmax(logits,-1),1)],1)
    if was: model.train()
    return tokens.tolist()[0]

class Cfg:
    vocab_size=256; context_length=32; d_model=64; num_heads=4; num_layers=2
    mlp_expansion=4; dropout=0.1; use_bias=True; weight_tying=True

def main():
    torch.manual_seed(0); cfg=Cfg()
    model=GPTModel(cfg); print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    ids=torch.randint(0,cfg.vocab_size,(4096,)); val_ids=torch.randint(0,cfg.vocab_size,(1024,))
    train_loader=make_batches(ids,4,32,seed=1); val_loader=make_batches(val_ids,4,32,seed=2)
    optimizer=torch.optim.AdamW(build_param_groups(model,0.01),lr=3e-3,betas=(0.9,0.95))
    prompt=torch.tensor([[7,11,13,17]]); log=[]
    Path("outputs").mkdir(exist_ok=True)
    for step in range(80):
        lr=cosine_warmup(step,10,80,3e-3,3e-4)
        for g in optimizer.param_groups: g["lr"]=lr
        inp,tgt=next(train_loader); optimizer.zero_grad(set_to_none=True)
        loss=calc_loss_batch(model,inp,tgt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        rec={"step":step,"train_loss":loss.item(),"lr":lr}
        if (step+1)%20==0:
            rec["val_loss"]=evaluate(model,val_loader,4)
            print(f"step {step:3d} | lr {lr:.5f} | train {loss.item():.4f} | val {rec['val_loss']:.4f}")
            generate(model,prompt,16,t=0.8,top_k=20,seed=step)
        log.append(rec)
    Path("outputs/losses.jsonl").write_text("\n".join(json.dumps(r) for r in log)+"\n")
    print(f"首步损失: {log[0]['train_loss']:.4f}  末步损失: {log[-1]['train_loss']:.4f}")
    print(f"日志: outputs/losses.jsonl ({len(log)}行)")
    print("Demo OK.")

if __name__=="__main__": raise SystemExit(main())
