# 综合项目37——加载预训练权重（safetensors权重映射）

> 从头训练124M参数模型是预算决策；加载已发布的检查点是周二日常。本课程从safetensors文件加载预训练GPT-2风格权重到第35节的精确架构中，逐个映射参数名称，验证形状，转置conv1d风格权重布局，用加载权重生成文本以确认加载成功。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-36节
**预计时间：** 90分钟

---

## 学习目标

- 用safetensors库读取文件并检查张量名称和形状
- 将每个预训练参数名称映射到本地GPT模型中的参数
- 处理发布的GPT-2权重和本地模型之间的两种名称约定差异
- 在任何权重赋值前检测并拒绝形状不匹配
- 加载权重后生成短文本确认加载有效

---

## 1. 问题

发布的权重不是为你的架构打包的。它们携带原始实现使用的名称。预训练文件有`transformer.h.0.attn.c_attn.weight`，你的模型期望`blocks.0.attn.qkv.weight`（相同矩阵的不同布局约定）。

盲目复制的加载器将正确张量放在错误位置。只在形状不匹配时拒绝但不记录的加载器让你猜测哪个张量没落地。

---

## 2. 核心概念

### 2.1 GPT-2命名约定

| 预训练名称 | 形状 | 含义 |
|------------|------|------|
| `wte.weight` | (50257, 768) | Token嵌入 |
| `wpe.weight` | (1024, 768) | 位置嵌入 |
| `h.N.attn.c_attn.weight` | (768, 2304) | 融合QKV线性权重 |
| `h.N.mlp.c_fc.weight` | (768, 3072) | MLP fc1权重 |
| `ln_f.weight` | (768,) | 最终LayerNorm scale |

### 2.2 本地命名约定

| 本地名称 | 含义 |
|----------|------|
| `tok_embed.weight` | Token嵌入 |
| `pos_embed.weight` | 位置嵌入 |
| `blocks.N.attn.qkv.weight` | 融合QKV |
| `blocks.N.mlp.fc1.weight` | MLP fc1 |
| `final_ln.scale` | 最终LayerNorm scale |

### 2.3 转置加载

发布的GPT-2权重以conv1d布局存储（与nn.Linear期望的转置）。加载器在赋值期间转置。LM头不在文件中——模型依赖与`wte`的权重绑定。

### 2.4 LoadReport

跟踪加载、缺失、意外和形状不匹配列表。打印它就是告诉加载是否成功。

---

## 3. 从零实现

`code/main.py`实现名称映射、形状检查、转置加载和演示。

```python
"""加载预训练权重——safetensors映射+形状验证。"""
import math
from dataclasses import dataclass, field
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

@dataclass
class ModelConfig:
    vocab_size:int=256; context_length:int=64; d_model:int=192
    num_heads:int=6; num_layers:int=4; mlp_expansion:int=4
    dropout:float=0.0; use_bias:bool=True; weight_tying:bool=True

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
        self.tok_embed=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos_embed=nn.Embedding(cfg.context_length,cfg.d_model)
        self.embed_drop=nn.Dropout(cfg.dropout); self.blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln=LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        if cfg.weight_tying: self.lm_head.weight=self.tok_embed.weight
        self.register_buffer("pos_ids",torch.arange(cfg.context_length,dtype=torch.long),persistent=False)
    def forward(self,tokens):
        B,T=tokens.shape; tok=self.tok_embed(tokens); pos=self.pos_embed(self.pos_ids[:T])
        x=self.embed_drop(tok+pos)
        for b in self.blocks: x=b(x)
        return self.lm_head(self.final_ln(x))

@dataclass
class LoadReport:
    loaded:list=field(default_factory=list); missing:list=field(default_factory=list)
    unexpected:list=field(default_factory=list); shape_mismatch:list=field(default_factory=list)
    def summary(self): return f"loaded={len(self.loaded)} missing={len(self.missing)} unexpected={len(self.unexpected)} shape_mismatch={len(self.shape_mismatch)}"
    def ok(self): return not self.missing and not self.shape_mismatch

CONV1D_SUFFIXES=("c_attn.weight","c_proj.weight","c_fc.weight")

def make_name_map(num_layers):
    m={"wte.weight":"tok_embed.weight","wpe.weight":"pos_embed.weight","ln_f.weight":"final_ln.scale","ln_f.bias":"final_ln.shift"}
    for i in range(num_layers):
        p=f"h.{i}"; d=f"blocks.{i}"
        m[f"{p}.ln_1.weight"]=f"{d}.ln1.scale"; m[f"{p}.ln_1.bias"]=f"{d}.ln1.shift"
        m[f"{p}.ln_2.weight"]=f"{d}.ln2.scale"; m[f"{p}.ln_2.bias"]=f"{d}.ln2.shift"
        m[f"{p}.attn.c_attn.weight"]=f"{d}.attn.qkv.weight"; m[f"{p}.attn.c_attn.bias"]=f"{d}.attn.qkv.bias"
        m[f"{p}.attn.c_proj.weight"]=f"{d}.attn.out_proj.weight"; m[f"{p}.attn.c_proj.bias"]=f"{d}.attn.out_proj.bias"
        m[f"{p}.mlp.c_fc.weight"]=f"{d}.mlp.fc1.weight"; m[f"{p}.mlp.c_fc.bias"]=f"{d}.mlp.fc1.bias"
        m[f"{p}.mlp.c_proj.weight"]=f"{d}.mlp.fc2.weight"; m[f"{p}.mlp.c_proj.bias"]=f"{d}.mlp.fc2.bias"
    return m

def load_safetensors(model,path):
    mapping=make_name_map(model.cfg.num_layers); local=dict(model.named_parameters())
    report=LoadReport(); pending=[]
    with safe_open(str(path),framework="pt") as r:
        for src in r.keys():
            dst=mapping.get(src)
            if dst is None: report.unexpected.append(src); continue
            if dst not in local: report.unexpected.append(src); continue
            t=r.get_tensor(src)
            if any(src.endswith(s) for s in CONV1D_SUFFIXES): t=t.t().contiguous()
            if tuple(t.shape)!=tuple(local[dst].shape):
                report.shape_mismatch.append((src,tuple(t.shape),tuple(local[dst].shape))); continue
            pending.append((src,dst,t))
    with torch.no_grad():
        for src,dst,t in pending:
            local[dst].copy_(t.to(device=local[dst].device,dtype=local[dst].dtype))
            report.loaded.append((src,dst,tuple(t.shape)))
    return report

def make_stub(path,cfg,seed=42):
    gen=torch.Generator().manual_seed(seed)
    def randn(*s): return torch.randn(*s,generator=gen)*0.02
    t={}
    t["wte.weight"]=randn(cfg.vocab_size,cfg.d_model); t["wpe.weight"]=randn(cfg.context_length,cfg.d_model)
    t["ln_f.weight"]=torch.ones(cfg.d_model); t["ln_f.bias"]=torch.zeros(cfg.d_model)
    for i in range(cfg.num_layers):
        t[f"h.{i}.ln_1.weight"]=torch.ones(cfg.d_model); t[f"h.{i}.ln_1.bias"]=torch.zeros(cfg.d_model)
        t[f"h.{i}.ln_2.weight"]=torch.ones(cfg.d_model); t[f"h.{i}.ln_2.bias"]=torch.zeros(cfg.d_model)
        t[f"h.{i}.attn.c_attn.weight"]=randn(3*cfg.d_model,cfg.d_model).t(); t[f"h.{i}.attn.c_attn.bias"]=torch.zeros(3*cfg.d_model)
        t[f"h.{i}.attn.c_proj.weight"]=randn(cfg.d_model,cfg.d_model).t(); t[f"h.{i}.attn.c_proj.bias"]=torch.zeros(cfg.d_model)
        t[f"h.{i}.mlp.c_fc.weight"]=randn(cfg.mlp_expansion*cfg.d_model,cfg.d_model).t(); t[f"h.{i}.mlp.c_fc.bias"]=torch.zeros(cfg.mlp_expansion*cfg.d_model)
        t[f"h.{i}.mlp.c_proj.weight"]=randn(cfg.d_model,cfg.mlp_expansion*cfg.d_model).t(); t[f"h.{i}.mlp.c_proj.bias"]=torch.zeros(cfg.d_model)
    save_file(t,str(path))

@torch.no_grad()
def quick_gen(model,prompt,n=8,seed=0):
    torch.manual_seed(seed); model.eval(); tokens=prompt.clone()
    for _ in range(n):
        logits=model(tokens[:,-model.cfg.context_length:])
        tokens=torch.cat([tokens,torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)],1)
    return tokens.tolist()[0]

def main():
    torch.manual_seed(0); cfg=ModelConfig(); path=Path("outputs/gpt2-stub.safetensors")
    path.parent.mkdir(exist_ok=True)
    if not path.exists(): make_stub(path,cfg,seed=42)
    print(f"文件: {path} ({path.stat().st_size:,}字节)")
    m=GPTModel(cfg); fp_before=sum(p.detach().norm().item() for p in m.parameters())
    prompt=torch.tensor([[7,11,13,17]]); tokens_before=quick_gen(m,prompt)
    report=load_safetensors(m,path)
    fp_after=sum(p.detach().norm().item() for p in m.parameters())
    tokens_after=quick_gen(m,prompt)
    print(f"报告: {report.summary()}")
    print(f"指纹变化: {fp_before:.4f} -> {fp_after:.4f}")
    print(f"加载前: {tokens_before} 加载后: {tokens_after}")
    assert tokens_before!=tokens_after; print("权重绑定: "+str(m.lm_head.weight.data_ptr()==m.tok_embed.weight.data_ptr()))
    return 0

if __name__=="__main__": raise SystemExit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 名称映射 | 从预训练张量名到本地参数名的函数 |
| 形状不匹配 | 预训练张量存在但维度与本地参数不一致 |
| 转置加载 | Conv1d布局权重在赋值时转置 |
| 权重绑定别名 | lm_head.weight = tok_embed.weight |
| 加载报告 | 跟踪loaded/missing/unexpected/shape_mismatch |

---

## 5. 面试考点

**Q1：为什么预训练权重需要转置？**
**Q2：权重绑定如何减少参数量？**
