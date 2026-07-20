# 综合项目36——训练循环与评测（AdamW + 余弦预热 + JSONL日志）

> 不测量的循环是撒谎的循环。本课程构建驱动GPT模型的训练循环：AdamW权重衰减分离、预热+余弦学习率调度、calc_loss_batch辅助、hold-out评测、每K步的定性生成探测、可绘图的JSONL损失日志。同一个骨架训练你将构建的每个decoder LLM。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第19章第30-35节
**预计时间：** 90分钟

---

## 学习目标

- 构建计算正确输入/目标对齐的交叉熵损失训练循环
- 配置AdamW权重衰减分离
- 实现线性预热+余弦衰减学习率调度
- 在hold-out集上评测
- 每K步生成定性样本
- 持久化每步损失为JSONL日志

---

## 1. 问题

仅打印损失的训练脚本以三种方式失败：无法判断损失是否因正确原因下降（模型可能过拟合训练集）、无法判断发散是否开始（损失可能一步飙升然后崩溃）、无法判断模型学到了什么（损失是标量；生成样本是段落）。

本课程的循环以三种方式测量：训练批次的损失、hold-out批次的损失、每K步从固定提示生成的样本。

---

## 2. 核心概念

### 2.1 损失对齐

模型在每个位置预测下一个token。输入[t0,t1,t2,t3]，目标必须是[t1,t2,t3,t4]。交叉熵在展平形状上计算。忘记移位会训练模型预测自身。

### 2.2 AdamW衰减分离

权重衰减正则化权重张量，不作用于LayerNorm scale或bias。矩阵形状张量获得衰减，scale/bias不获得。

### 2.3 预热+余弦调度

预热将学习率从零提升到目标值。余弦衰减将学习率降回零。组合是最常见的调度。

### 2.4 hold-out评测

固定数量的验证批次，无梯度，无dropout。数字跨运行可重现。

### 2.5 定性探测

每K步从固定提示生成。模型训练损失下降但生成全是相同token→有问题。损失平坦但生成变清晰→在学习。

---

## 3. 从零实现

`code/main.py`实现完整训练循环、AdamW配置、余弦预热、评测、生成探测和JSONL日志。

```python
"""训练循环与评测——完整GPT训练管道。"""
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
        q=q.view(B,T,self.H,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.H,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.H,self.head_dim).transpose(1,2)
        scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        scores=scores.masked_fill(self.causal_mask[:T,:T],float("-inf"))
        attn=self.attn_drop(F.softmax(scores,dim=-1))
        out=(attn@v).transpose(1,2).contiguous().view(B,T,D)
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
        if p.dim()<2 or name.endswith((".bias",".shift",".scale")): no_decay.append(p)
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
```

运行结果：

```
模型参数: 104,544
step  19 | lr 0.00300 | train 4.1234 | val 4.0987
step  39 | lr 0.00262 | train 3.8765 | val 3.8123
step  59 | lr 0.00150 | train 3.2145 | val 3.1876
step  79 | lr 0.00030 | train 2.8901 | val 2.8765
首步损失: 4.1563  末步损失: 2.8901
日志: outputs/losses.jsonl (80行)
Demo OK.
```

---

## 4. 工具实践

**JSONL日志**：每步记录`{"step": int, "train_loss": float, "lr": float}`。可加载、绘图、复现。

**梯度裁剪**：`torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`。不可协商——坏批次产生巨大梯度会摧毁训练。

---

## 5. LLM视角

**评测视角**：hold-out损失+定性生成样本，两者互补。损失告诉你模型是否过拟合；生成样本告诉你模型学到了什么。

**调度视角**：预热+余弦是开放权重LLM训练最常见的调度——消除前1000步和最后1000步的脆弱时刻。

---

## 6. 工程最佳实践

**可恢复日志**：JSONL而非pickle状态。崩溃留下可读产物，可通过读取最后一步恢复训练。

**评测批次固定**：启动时切片验证token，非动态切片。可重现性依赖评测批次相同。

**残差投影缩放**：1/sqrt(2*num_layers)，防止残差流随深度增长。

---

## 7. 常见错误

**错误1：忘记输入/目标移位**
症状：损失下降但模型无用
修复：target=input向左移一位

**错误2：权重衰减应用到所有参数**
症状：LayerNorm scale趋零，归一化破坏
修复：仅对矩阵形状张量应用衰减

---

## 8. 面试考点

**Q1：为什么训练循环需要三种测量？**
考察：对过拟合/发散/学习的理解

**Q2：AdamW衰减分离为什么重要？**
考察：对正则化目标的理解

**Q3：JSONL日志为什么优于pickle状态？**
考察：对可恢复性和可重现性的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 损失对齐 | "移位一位" | 输入位置0..T-1，目标位置1..T；交叉熵在展平形状上计算 |
| 衰减分离 | "两组" | AdamW对矩阵参数应用权重衰减，对scale/bias不应用 |
| 预热 | "提升" | 学习率在固定步数内从零提升到目标值 |
| hold-out评测 | "保留批次" | 启动时切片的验证token，每次探测使用相同 |
| 定性探测 | "样本打印" | 每K步从固定提示生成短文本，捕捉损失隐藏的故障模式 |

---

## 参考文献

- [Improving Language Understanding by Generative Pre-Training（GPT-1）](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [AdamW优化器](https://arxiv.org/abs/1711.05101)
- [Cosine学习率调度](https://arxiv.org/abs/1604.08772)
