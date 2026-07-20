# 综合项目41——完整评测管道

> 训练是可以用损失曲线监控的部分。评测是你必须设计的部分。本课程构建统一的评测管道，对任何训练好的语言模型运行四种异构评测，聚合为每任务报告，并内置本地mock LLM评判器。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 计算带掩码token计数的hold-out困惑度
- 在短形式事实提示上运行精确匹配评测
- 计算预测和参考字符串间的token F1
- 构建本地mock LLM评判器（1-5分）
- 将四种评测聚合为加权报告

---

## 1. 问题

单一指标从不描述一个语言模型。困惑度说模型拟合语言分布多好，但不说它是否回答问题。精确匹配说模型是否产生gold字符串，但惩罚正确改写。token F1原谅改写但被错误内容的词重叠欺骗。你实际想要的管道四种都有。

---

## 2. 核心概念

### 2.1 困惑度（正确计数）

困惑度 = exp(平均每个token的负对数似然)。两个陷阱：均值必须在实际token位置上（排除padding）；模型在位置i预测位置i+1。

### 2.2 精确匹配

规范化：小写、去空白、折叠双空格、去除尾标点。规范化使精确匹配在实践中可用。

### 2.3 Token F1

1. 规范化预测和参考
2. 空格分词
3. 计算多集交集
4. 精确率 = 交集/预测token数，召回率 = 交集/参考token数，F1 = 调和平均

### 2.4 Mock LLM评判器

5分制：5=精确匹配，4=F1≥0.8，3=F1∈[0.5,0.8)，2=F1∈[0.2,0.5)，1=其他。

### 2.5 聚合

加权平均归一化分数。默认权重：0.2困惑度、0.3精确匹配、0.3 token F1、0.2评判。

---

## 3. 从零实现

```python
"""完整评测管道——困惑度+精确匹配+token F1+评判器。"""
import json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

class InstructionTokenizer:
    INST=256; RESP=257; PAD=258; VOCAB=260; IGNORE=-100

class CausalSelfAttn(nn.Module):
    def __init__(self,D,H,L):
        super().__init__(); self.H=H; self.dh=D//H
        self.qkv=nn.Linear(D,D*3,bias=False); self.out=nn.Linear(D,D,bias=False)
        self.register_buffer("mask",torch.tril(torch.ones(L,L,dtype=torch.bool)),persistent=False)
    def forward(self,x):
        B,T,D=x.shape; q,k,v=self.qkv(x).view(B,T,3,self.H,self.dh).permute(2,0,3,1,4)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.dh); att=att.masked_fill(~self.mask[:T,:T],float("-inf"))
        return self.out((F.softmax(att,-1)@v).transpose(1,2).contiguous().view(B,T,D))

class Block(nn.Module):
    def __init__(self,D,H,L):
        super().__init__(); self.ln1=nn.LayerNorm(D); self.attn=CausalSelfAttn(D,H,L)
        self.ln2=nn.LayerNorm(D); self.fc1=nn.Linear(D,D*4); self.fc2=nn.Linear(D*4,D)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))+x

class TinyGPT(nn.Module):
    def __init__(self,V,D,H,depth,L):
        super().__init__()
        self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(L,D)
        self.blocks=nn.ModuleList([Block(D,H,L) for _ in range(depth)])
        self.ln=nn.LayerNorm(D); self.head=nn.Linear(D,V,bias=False); self.L=L
    def forward(self,ids):
        B,T=ids.shape; pos=torch.arange(T,device=ids.device).unsqueeze(0).expand(B,T)
        x=self.tok(ids)+self.pos(pos)
        for b in self.blocks: x=b(x)
        return self.head(self.ln(x))

def perplexity_eval(model,ids,mask):
    with torch.no_grad():
        logits=model(ids); shifted_logits=logits[:,:-1,:].reshape(-1,logits.size(-1))
        targets=ids[:,1:].reshape(-1); loss=F.cross_entropy(shifted_logits,targets,ignore_index=258,reduction="none")
        mask_flat=mask[:,1:].reshape(-1).float()
        total=(loss*mask_flat).sum(); tokens=mask_flat.sum().clamp(min=1)
    return {"perplexity":float(math.exp(total/tokens.item())), "tokens":int(tokens.item())}

def normalise(text):
    s=text.lower().strip(); s=" ".join(s.split())
    if s and s[-1] in ".!?": s=s[:-1]
    return s

def exact_match_eval(pairs):
    hits=sum(1 for p,r in pairs if normalise(p)==normalise(r))
    return {"exact_match":hits/max(len(pairs),1),"total":len(pairs),"hits":hits}

def token_f1(pred,ref):
    p=set(normalise(pred).split()); r=set(normalise(ref).split())
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    inter=len(p&r); prec=inter/max(len(p),1); rec=inter/max(len(r),1)
    return 2*prec*rec/(prec+rec) if prec+rec else 0

def token_f1_eval(pairs):
    scores=[token_f1(p,r) for p,r in pairs]
    return {"token_f1":float(np.mean(scores)),"per_example":scores}

def mock_judge(inst,pred,ref):
    if normalise(pred)==normalise(ref): return 5,"exact match"
    f1=token_f1(pred,ref)
    if f1>=0.8: return 4,"high overlap"
    if f1>=0.5: return 3,"moderate overlap"
    if f1>=0.2: return 2,"low overlap"
    return 1,"minimal match"

def judge_eval(pairs):
    scores=[mock_judge(i,p,r)[0] for i,p,r in pairs]
    return {"judge_score":float(np.mean(scores))/5,"per_example":scores}

def aggregate(results,weights):
    norm=lambda k,v: min(1,1/(1+math.log(max(v,1e-10)))) if k=="perplexity" else v
    total=sum(weights[k]*norm(k,results[k]) for k in weights)
    return {"aggregate":total,"details":{k:norm(k,v) for k,v in results.items()}}

def main():
    torch.manual_seed(0); tok=InstructionTokenizer(); model=TinyGPT(260,96,4,2,96)
    # Brief train
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    for _ in range(10):
        ids=torch.randint(0,260,(4,64)); loss=F.cross_entropy(model(ids)[:,:-1,:].reshape(-1,260),ids[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval
    ids=torch.randint(0,260,(2,32)); mask=torch.ones(2,32)
    ppl=perplexity_eval(model,ids,mask); print(f"困惑度: {ppl['perplexity']:.2f}")
    em=exact_match_eval([("Paris","Paris."),("Madrid","Madrid")]); print(f"精确匹配: {em['exact_match']:.2f}")
    f1=token_f1_eval([("Paris is capital","Paris"),("Tokyo is capital","Tokyo")]); print(f"Token F1: {f1['token_f1']:.2f}")
    jd=judge_eval([("Capital?","Paris","Paris."),("Capital?","Paris is in France","Tokyo")]); print(f"评判: {jd['judge_score']:.2f}")
    agg=aggregate({"perplexity":ppl["perplexity"],"exact_match":em["exact_match"],"token_f1":f1["token_f1"],"judge_score":jd["judge_score"]},
                   {"perplexity":0.2,"exact_match":0.3,"token_f1":0.3,"judge_score":0.2})
    print(f"聚合: {agg['aggregate']:.3f}")
    json.dump(agg,open("outputs/report.json","w"))
    return 0

if __name__=="__main__": import sys; sys.exit(main())
