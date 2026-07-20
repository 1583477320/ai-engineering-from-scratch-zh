# 综合项目38——分类器微调（头部替换）

> 预训练语言模型是自注意力块的堆叠，末端是token预测头。当你想要垃圾邮件/非垃圾邮件分类时，头部是错的但身体大致是对的。本课程切掉头部，粘贴一个二类线性层到池化表示上，并用两种方式训练分类器：仅最终层和全量微调。评测是精确率、召回率和F1。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 在不重新初始化身体的情况下替换语言模型头为分类头
- 实现两种训练策略：冻结身体（仅头部）和全量微调
- 构建分词器感知的数据管道，填充掩码并池化注意力输出
- 从原始logits计算精确率、召回率、F1和混淆矩阵
- 理解参数量、训练时间和准确率之间的权衡

---

## 1. 问题

三种选择：从零训练分类器（错误——浪费预训练的结构）、头部替换+冻结身体（快、内存便宜、不易过拟合）、头部替换+全量微调（慢、可能过拟合小数据、但能达到更高准确率）。

本课程构建两种策略，让你在同一固定数据集上比较它们。

---

## 2. 核心概念

### 2.1 头部替换

模型是函数f_theta(tokens) -> hidden_states。头部是g_phi(hidden) -> logits。替换头部意味着保持theta，替换g_phi。头部只是一个线性层。

### 2.2 冻结 vs 全量微调

冻结：requires_grad=False在身体参数上，优化器只看到头部。全量：梯度流回整个堆栈。

### 2.3 池化

分类器需要每个序列一个向量。本课程使用均值池化：按注意力掩码加权平均隐藏状态。

---

## 3. 从零实现

`code/main.py`实现分类器、冻结/解冻、训练循环和评测。

```python
"""分类器微调——头部替换+冻结/全量对比。"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import List, Sequence, Tuple

class ByteTokenizer:
    PAD_ID=256; VOCAB=260
    def encode(self,text,max_len):
        raw=list(text.encode("utf-8",errors="ignore"))[:max_len]
        attn=[1]*len(raw)
        while len(raw)<max_len: raw.append(self.PAD_ID); attn.append(0)
        return raw,attn

class MultiHeadAttention(nn.Module):
    def __init__(self,D,H):
        super().__init__(); self.H=H; self.dh=D//H
        self.qkv=nn.Linear(D,D*3,bias=False); self.out=nn.Linear(D,D,bias=False)
    def forward(self,x,mask):
        B,T,D=x.shape; q,k,v=self.qkv(x).view(B,T,3,self.H,self.dh).permute(2,0,3,1,4)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        att=att.masked_fill(mask.view(B,1,1,T)==0,float("-inf"))
        w=F.softmax(att,dim=-1); w=torch.nan_to_num(w,nan=0.0)
        return self.out((w@v).transpose(1,2).contiguous().view(B,T,D))

class FeedForward(nn.Module):
    def __init__(self,D): super().__init__(); self.fc1=nn.Linear(D,D*4); self.fc2=nn.Linear(D*4,D)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self,D,H):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=MultiHeadAttention(D,H); self.ln2=nn.LayerNorm(D); self.ff=FeedForward(D)
    def forward(self,x,m): x=x+self.attn(self.ln1(x),m); return x+self.ff(self.ln2(x))

class LMBody(nn.Module):
    def __init__(self,V,D,H,depth,L):
        super().__init__()
        self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(L,D); self.blocks=nn.ModuleList([Block(D,H) for _ in range(depth)]); self.ln=nn.LayerNorm(D); self.L=L
    def forward(self,ids,m):
        B,T=ids.shape; pos=torch.arange(T,device=ids.device).unsqueeze(0).expand(B,T)
        x=self.tok(ids)+self.pos(pos)
        for b in self.blocks: x=b(x,m)
        return self.ln(x)

class Classifier(nn.Module):
    def __init__(self,body,nc=2):
        super().__init__(); self.body=body; self.head=nn.Linear(body.ln.normalized_shape[0],nc)
    def forward(self,ids,m):
        h=self.body(ids,m); m2=m.unsqueeze(-1).to(h.dtype)
        return self.head((h*m2).sum(1)/m2.sum(1).clamp(min=1))

def freeze_body(m):
    for p in m.body.parameters(): p.requires_grad=False

def unfreeze_body(m):
    for p in m.body.parameters(): p.requires_grad=True

SPAM=["FREE entry in {n}wkly comp to win FA Cup","URGENT call {p} to claim your prize","WINNER claim your {a} pound award now"]
HAM=["are you home for dinner tonight","see you at {t} tomorrow","can you pick up {i} on the way"]
SLOTS={"n":["1","2","3"],"p":["09061701461","08000839402"],"a":["100","250","500"],"t":["6pm","7pm"],"i":["milk","bread"]}

def fill(tpl,rng):
    s=tpl
    for k,opts in SLOTS.items():
        if "{"+k+"}" in s: s=s.replace("{"+k+"}",rng.choice(opts))
    return s

def make_data(n=800,seed=0):
    rng=random.Random(seed); texts=[]; labels=[]
    for _ in range(n//2):
        texts.append(fill(rng.choice(SPAM),rng)); labels.append(1)
        texts.append(fill(rng.choice(HAM),rng)); labels.append(0)
    order=list(range(len(texts))); rng.shuffle(order)
    return [texts[i] for i in order],[labels[i] for i in order]

class DS(Dataset):
    def __init__(self,texts,labels,tok,L):
        self.texts=texts; self.labels=labels; self.tok=tok; self.L=L
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        ids,mask=self.tok.encode(self.texts[i],self.L)
        return torch.tensor(ids),torch.tensor(mask),torch.tensor(self.labels[i])

def evaluate(model,dl):
    tp=fp=fn=tn=0; model.eval()
    with torch.no_grad():
        for ids,mask,y in dl:
            pred=model(ids,mask).argmax(-1)
            for p,yi in zip(pred.tolist(),y.tolist()):
                if p==1 and yi==1: tp+=1
                elif p==1: fp+=1
                elif yi==1: fn+=1
                else: tn+=1
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    f1=2*p*r/(p+r) if p+r else 0
    return p,r,f1

@dataclass
class Config:
    V:int=260; D:int=64; H:int=4; depth:int=2; L:int=32
    bs:int=32; head_ep:int=20; full_ep:int=20; head_lr:float=5e-3; full_lr:float=1e-3

import math, random
def main():
    import torch.nn.functional as F
    torch.manual_seed(0); cfg=Config(); rng=random.Random(0)
    tok=ByteTokenizer(); texts,labels=make_data(800,0)
    split=int(len(texts)*0.8); tr_t,tr_y=texts[:split],labels[:split]; te_t,te_y=texts[split:],labels[split:]
    train_dl=DataLoader(DS(tr_t,tr_y,tok,cfg.L),cfg.bs,shuffle=True)
    test_dl=DataLoader(DS(te_t,te_y,tok,cfg.L),cfg.bs)

    body=LMBody(cfg.V,cfg.D,cfg.H,cfg.depth,cfg.L)
    # Warm-up pretrain
    head_proj=nn.Linear(cfg.D,cfg.V,bias=False)
    opt=torch.optim.Adam(list(body.parameters())+list(head_proj.parameters()),lr=3e-3)
    ds_lm=list(zip(tr_t,tr_y))
    for _ in range(5):
        for txt,_ in ds_lm[:32]:
            ids,mask=tok.encode(txt,cfg.L)
            ids_t,mask_t=torch.tensor(ids),torch.tensor(mask)
            logits=head_proj(body(ids_t.unsqueeze(0),mask_t.unsqueeze(0)))
            loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,cfg.V),ids_t[1:])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"预训练损失: {loss.item():.4f}")

    head_only=Classifier(LMBody(cfg.V,cfg.D,cfg.H,cfg.depth,cfg.L)); head_only.body.load_state_dict(body.state_dict())
    unfreeze_body(head_only); freeze_body(head_only)
    opt_h=torch.optim.Adam([p for p in head_only.parameters() if p.requires_grad],lr=cfg.head_lr)
    for ep in range(cfg.head_ep):
        for ids,mask,y in train_dl:
            loss=F.cross_entropy(head_only(ids,mask),y); opt_h.zero_grad(); loss.backward(); opt_h.step()
    p,r,f1=head_only(test_dl); print(f"仅头部: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    full=Classifier(LMBody(cfg.V,cfg.D,cfg.H,cfg.depth,cfg.L)); full.body.load_state_dict(body.state_dict())
    opt_f=torch.optim.Adam(full.parameters(),lr=cfg.full_lr)
    for ep in range(cfg.full_ep):
        for ids,mask,y in train_dl:
            loss=F.cross_entropy(full(ids,mask),y); opt_f.zero_grad(); loss.backward(); opt_f.step()
    p,r,f1=full(test_dl); print(f"全量微调: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    return 0

if __name__=="__main__": sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 头部替换 | 保持预训练身体，替换LM头为分类线性层 |
| 冻结微调 | requires_grad=False在身体上，仅训练头部 |
| 全量微调 | 梯度流回整个堆栈 |
| 均值池化 | 按注意力掩码加权平均序列隐藏状态 |
| F1分数 | 精确率和召回率的调和平均 |

---

## 5. 面试考点

**Q1：头部替换与全量微调的权衡？**
**Q2：为什么冻结身体在小数据集上更安全？**
