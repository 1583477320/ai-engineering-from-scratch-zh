# 综合项目39——指令微调（SFT监督微调）

> 预训练基础模型可以扩展序列但不能遵循指令。监督微调是最小的修复：向模型喂入指令和期望响应的配对示例，训练身体预测响应token。关键是只让损失计算响应而非指令。本课程构建Alpaca风格SFT循环，用`ignore_index=-100`掩码指令token，在200个指令-响应对上训练，用精确匹配评测。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将配对指令-响应数据格式化为带边界token的单一因果序列
- 构建掩码指令token的collate函数，使交叉熵只计算响应token
- 在SFT目标下训练小型Transformer身体
- 实现尊重响应起始边界的贪心和温度采样生成
- 计算hold-out集上的精确匹配

---

## 1. 问题

基础模型不知道什么是指令。给它字符串"What is the capital of France?"它会继续问题或发明新句子。模型有语言但没有格式契约。

SFT契约是一个字符串模板：`<INST> 指令 <RESP> 响应`。边界token在训练时预留。但有一个陷阱：如果对整个序列用普通交叉熵，你是在训练模型预测指令token。指令是给定的，你想要零梯度。修复是掩码。

---

## 2. 核心概念

### 2.1 掩码目标

`ignore_index`是`torch.nn.functional.cross_entropy`的特性。任何等于`ignore_index`的目标位置贡献零损失和零梯度。约定在PyTorch中是`-100`。

collate函数为每个示例构建两个张量：`input_ids`（完整序列）和`labels`（input_ids的副本，指令位置被`-100`覆盖）。

模型在前向传递时看到整个序列（注意力可关注指令），损失只计算响应token。这正是你想要的：以指令为条件，预测响应。

### 2.2 精确匹配

最严格的文本指标。预测响应字符串被规范化（小写、去除空格、折叠双空格）后与参考响应比较。要么1要么0。

---

## 3. 从零实现

`code/main.py`实现指令分词器、SFT数据集、collate、TinyGPT、训练循环和精确匹配评测。

```python
"""指令微调SFT——掩码指令token+精确匹配评测。"""
import sys, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

class InstructionTokenizer:
    INST=256; RESP=257; PAD=258; VOCAB=260; IGNORE=-100
    def encode_pair(self,inst,resp,L):
        ib=list(inst.encode("utf-8",errors="ignore"))[:L-3]
        ids=[self.INST]+ib+[self.RESP]; rs=len(ids)
        ids.extend(list(resp.encode("utf-8",errors="ignore"))[:L-len(ids)])
        return ids,rs
    def encode_prefix(self,inst,L):
        return [self.INST]+list(inst.encode("utf-8",errors="ignore"))[:L-2]+[self.RESP]
    def decode_response(self,ids):
        return bytes(i for i in ids if i<256).decode("utf-8",errors="replace")

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
        self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(L,D); self.blocks=nn.ModuleList([Block(D,H,L) for _ in range(depth)])
        self.ln=nn.LayerNorm(D); self.head=nn.Linear(D,V,bias=False); self.L=L
    def forward(self,ids):
        B,T=ids.shape; pos=torch.arange(T,device=ids.device).unsqueeze(0).expand(B,T)
        x=self.tok(ids)+self.pos(pos)
        for b in self.blocks: x=b(x)
        return self.head(self.ln(x))

def sft_collate(batch,pad=258,ign=-100):
    max_t=max(len(x[0]) for x in batch)
    il,la,am=[],[],[]
    for ids,rs in batch:
        p=ids+[pad]*(max_t-len(ids)); lbl=list(p)
        for i in range(len(lbl)):
            if i<rs or i>=len(ids): lbl[i]=ign
        il.append(p); la.append(lbl); am.append([1]*len(ids)+[0]*(max_t-len(ids)))
    return torch.tensor(il),torch.tensor(la),torch.tensor(am)

def shifted_loss(logits,labels,ign=-100):
    return F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1),ignore_index=ign)

def generate(model,tok,inst,max_len,T=0.0,seed=0):
    model.eval(); rng=torch.Generator(); rng.manual_seed(seed)
    ids=tok.encode_prefix(inst,max_len); outs=[]
    with torch.no_grad():
        for _ in range(max_len):
            if len(ids)>=max_len: break
            x=torch.tensor([ids]); logits=model(x)[:,-1,:]
            if T<=0: nid=int(logits.argmax().item())
            else: nid=int(torch.multinomial(F.softmax(logits/T,-1),1,generator=rng).item())
            if nid in (256,257,258): break
            ids.append(nid); outs.append(nid)
            if len(outs)>=2 and outs[-1] in (46,33,63) and outs[-2] in (46,33,63): break
    return tok.decode_response(outs)

def exact_match(pred,gold):
    return 1 if pred.lower().strip()==gold.lower().strip() else 0

# 数据集：6类任务，共200对
CAPITALS=[("France","Paris"),("Spain","Madrid"),("Italy","Rome"),("Japan","Tokyo"),("Egypt","Cairo"),("Brazil","Brasilia"),("Canada","Ottawa"),("Australia","Canberra"),("Kenya","Nairobi"),("Sweden","Stockholm")]
ARITH=[(2,3,"+"),(5,4,"*"),(9,7,"-"),(12,4,"/"),(7,6,"+"),(11,3,"-"),(8,8,"+"),(15,5,"/"),(6,9,"*"),(20,4,"/")]
LISTS=[("colors",["red","green","blue"]),("planets",["mercury","venus","earth"]),("vowels",["a","e","i"]),("seasons",["spring","summer","autumn"]),("metals",["iron","gold","silver"])]
CODES=[("print hello world","print('hello world')"),("print the number 42","print(42)"),("sort a list named items","items.sort()"),("reverse a list","items.reverse()"),("get the length of items","len(items)")]

def make_data(seed=0):
    rng=random.Random(seed); pairs=[]; cats=[]
    for c,ci in CAPITALS:
        for t in [f"What is the capital of {c}?","Name the capital city of {c}.",f"Capital of {c}?",f"Tell me the capital of {c}."]:
            pairs.append({"instruction":t,"response":f"the capital of {c} is {ci}."}); cats.append("capitals")
    for a,b,op in ARITH:
        r=f"{a} {op} {b} = {eval(f'{a}{op}{b}')" if op!="/" else f"{a} {op} {b} = {a//b}"
        for t in [f"Compute {a} {op} {b}.",f"What is {a} {op} {b}?"]:
            pairs.append({"instruction":t,"response":r}); cats.append("arithmetic")
    for nm,items in LISTS:
        for t in [f"List three {nm}.",f"Give me three {nm}."]:
            pairs.append({"instruction":t,"response":", ".join(items)+"."}); cats.append("lists")
    for task,code in CODES:
        for t in [f"Write python code to {task}.",f"Python: {task}."]:
            pairs.append({"instruction":t,"response":code}); cats.append("code")
    rng.shuffle(pairs); return pairs,cats

class SFTDS(Dataset):
    def __init__(self,pairs,tok,L):
        self.pairs=pairs; self.tok=tok; self.L=L
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ids,rs=self.tok.encode_pair(self.pairs[i]["instruction"],self.pairs[i]["response"],self.L)
        return ids,rs

def main():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    tok=InstructionTokenizer(); pairs,cats=make_data(0)
    split=int(len(pairs)*0.8); tr,te=pairs[:split],pairs[split:]
    train_dl=DataLoader(SFTDS(tr,tok,96),16,shuffle=True,collate_fn=sft_collate)
    model=TinyGPT(260,96,4,2,96); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    for ep in range(20):
        model.train(); el=0; nb=0
        for il,la,am in train_dl:
            loss=shifted_loss(model(il),la); opt.zero_grad(); loss.backward(); opt.step()
            el+=loss.item(); nb+=1
        if (ep+1)%5==0:
            em=sum(1 for p in te if exact_match(generate(model,tok,p["instruction"],96),p["response"]))/max(len(te),1)
            print(f"  epoch {ep+1}: loss={el/nb:.4f} EM={em:.3f}")
    cats_te=list(set(c for _,c in zip(te,[c for _,c in zip(range(len(te)),[c for _,c in zip(range(len(pairs)),cats)[split:]])])))
    em=sum(1 for p in te if exact_match(generate(model,tok,p["instruction"],96),p["response"]))/max(len(te),1)
    print(f"\n最终精确匹配 = {em:.3f}")
    return 0 if em>0 else 1

if __name__=="__main__": sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| ignore_index=-100 | 训练时掩码指令和padding位置 |
| SFT | 监督微调：用指令-响应对训练基础模型 |
| 精确匹配 | 规范化后字符串完全相同 |
| 边界token | INST_ID和RESP_ID分隔指令和响应 |

---

## 5. 面试考点

**Q1：为什么SFT需要掩码指令token？**
**Q2：精确匹配和BLEU/chrF的区别？**
