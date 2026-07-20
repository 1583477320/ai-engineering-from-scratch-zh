# 综合项目40——直接偏好优化（DPO从零实现）

> 奖励模型和PPO是经典RLHF栈。DPO将该栈折叠为单一监督损失，直接在偏好对上拟合策略。本课程从奖励差分恒等式推导DPO损失，实现参考模型+策略模型对，计算序列级对数概率，并在偏好固定数据集上训练小型Transformer。测试固定损失数学和梯度方向。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将DPO损失推导为缩放对数比率差上的sigmoid
- 构建冻结参考模型+可训练策略模型对
- 计算两个模型下的序列级对数概率，掩码prompt token
- 在偏好三元组上训练策略并观察chosen log-prob相对rejected上升
- 用损失数学、梯度符号和参考不变性测试固定行为

---

## 1. 问题

你有一个SFT模型。它遵循指令，但输出参差不齐。你还有一小部分偏好对：同一prompt，人工标记一个completion为chosen，另一个为rejected。

经典RLHF答案是两阶段管道：先训练奖励模型，再用PPO优化策略。DPO将两阶段替换为单一监督损失。奖励模型从不显式存在。

---

## 2. 核心概念

### 2.1 DPO损失推导

从Bradley-Terry偏好模型出发：

```
P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
```

最优策略的闭式解使对数比率差与奖励差成正比。取负对数似然：

```
L_DPO = -E[log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x) - log pi_theta(y_l|x) + log pi_ref(y_l|x)))]
```

无需单独的奖励模型。KL约束嵌入闭式推导中。

### 2.2 梯度符号

对`log pi_theta(y_w|x)`的梯度为负（增加chosen概率降低损失）。对`log pi_theta(y_l|x)`的梯度为正（增加rejected概率增加损失）。训练推高chosen，压低rejected。参考模型冻结。

### 2.3 参考不变性

参考模型（SFT冻结）必须：参数永远不接收梯度、对数概率在epoch间不变、策略从相同权重初始化。

---

## 3. 从零实现

`code/main.py`实现DPO损失、序列对数概率、训练循环和演示。

```python
"""DPO从零实现——偏好对直接优化。"""
import sys, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class InstructionTokenizer:
    INST=256; RESP=257; VOCAB=260
    def encode_prompt(self,p): return [self.INST]+list(p.encode("utf-8",errors="ignore"))+[self.RESP]
    def encode_completion(self,c): return list(c.encode("utf-8",errors="ignore"))

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

def seq_log_prob(model,tok,inst,comp):
    full=list(inst)+list(comp)
    if len(full)>model.L: full=full[-model.L:]; pl=max(0,len(full)-len(comp))
    else: pl=len(inst)
    ids=torch.tensor([full]); logits=model(ids); lp=F.log_softmax(logits,-1)
    targets=torch.tensor(full[pl:]); pos=torch.arange(pl,len(full)-1)
    return lp[0,pos,targets].sum()

def dpo_loss(lp_w_pol,lp_l_pol,lp_w_ref,lp_l_ref,beta):
    margin=(lp_w_pol-lp_w_ref)-(lp_l_pol-lp_l_ref)
    return -F.logsigmoid(beta*margin),margin

def make_prefs():
    return [
        {"prompt":"What is the capital of France?","chosen":"Paris.","rejected":"France is in Europe and has many cities including Paris."},
        {"prompt":"What is the capital of Japan?","chosen":"Tokyo.","rejected":"Japan is an island nation with government in Tokyo."},
        {"prompt":"Compute 2 + 3.","chosen":"5.","rejected":"2 plus 3 is close to 5 I believe."},
        {"prompt":"Compute 7 * 6.","chosen":"42.","rejected":"7 multiplied by 6 is around 42."},
        {"prompt":"Compute 12 / 4.","chosen":"3.","rejected":"Twelve divided by four is roughly three."},
        {"prompt":"List three colors.","chosen":"red, green, blue.","rejected":"Colors include red, green, and blue too."},
        {"prompt":"List three vowels.","chosen":"a, e, i.","rejected":"Vowels produce open mouth sounds like a, e, and i."},
        {"prompt":"Define variable.","chosen":"a name bound to a value.","rejected":"A variable is something you use to store stuff."},
        {"prompt":"Define function.","chosen":"a reusable block of code that returns an output.","rejected":"A function is something that does things on inputs."},
        {"prompt":"Python: print 42.","chosen":"print(42)","rejected":"You can print numbers in python."},
        {"prompt":"Python: sort items.","chosen":"items.sort()","rejected":"Sorting a list in python is easy."},
        {"prompt":"Python: get length.","chosen":"len(items)","rejected":"To get length call len on items."},
    ]

def build_models(cfg):
    torch.manual_seed(cfg.seed)
    ref=TinyGPT(cfg.V,cfg.D,cfg.H,cfg.depth,cfg.L)
    torch.manual_seed(cfg.seed)
    pol=TinyGPT(cfg.V,cfg.D,cfg.H,cfg.depth,cfg.L); pol.load_state_dict(ref.state_dict())
    for p in ref.parameters(): p.requires_grad=False
    ref.eval()
    return ref,pol

def warmup(model,tok,triples,epochs=8,lr=3e-3,seed=0):
    torch.manual_seed(seed); opt=torch.optim.Adam(model.parameters(),lr=lr)
    seqs=[[INST]+list(t["prompt"].encode("utf-8"))+[RESP]+list(t["chosen"].encode("utf-8")) for t in triples]
    INST=256; RESP=257
    for _ in range(epochs):
        for s in seqs:
            ids=torch.tensor([s[:model.L]]); logits=model(ids)
            loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,cfg.V),ids[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

def train_dpo(pol,ref,tok,triples,cfg):
    opt=torch.optim.Adam(pol.parameters(),lr=cfg.lr)
    INST=256; RESP=257
    ref_lps=[(seq_log_prob(ref,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["chosen"].encode("utf-8"))).detach(),
              seq_log_prob(ref,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["rejected"].encode("utf-8"))).detach()) for t in triples]
    init_margins=[]; report={"losses":[],"margins":[]}
    for ep in range(cfg.epochs):
        tl=0; tm=0
        for t,(lw_ref,ll_ref) in zip(triples,ref_lps):
            prompt=[INST]+list(t["prompt"].encode("utf-8"))+[RESP]
            lw_pol=seq_log_prob(pol,tok,prompt,list(t["chosen"].encode("utf-8")))
            ll_pol=seq_log_prob(pol,tok,prompt,list(t["rejected"].encode("utf-8")))
            loss,margin=dpo_loss(lw_pol,ll_pol,lw_ref,ll_ref,cfg.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item(); tm+=margin.item()
        report["losses"].append(tl/len(triples)); report["margins"].append(tm/len(triples))
    with torch.no_grad():
        init_m=[seq_log_prob(pol,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["chosen"].encode("utf-8"))).item()-
                seq_log_prob(pol,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["rejected"].encode("utf-8"))).item() for t in triples]
        final_m=[seq_log_prob(pol,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["chosen"].encode("utf-8"))).item()-
                 seq_log_prob(pol,tok,[INST]+list(t["prompt"].encode("utf-8"))+[RESP],list(t["rejected"].encode("utf-8"))).item() for t in triples]
    report["init_margin"]=np.mean(init_m); report["final_margin"]=np.mean(final_m)
    return report

@dataclass
class Cfg:
    V:int=260; D:int=64; H:int=4; depth:int=2; L:int=96; beta:float=0.2; lr:float=1e-3; epochs:int=30; seed:int=0

import numpy as np
def main():
    cfg=Cfg(); torch.manual_seed(0); tok=InstructionTokenizer(); triples=make_prefs()
    ref,pol=build_models(cfg)
    print("[warmup]"); warmup(ref,tok,triples,8,3e-3,cfg.seed)
    pol.load_state_dict(ref.state_dict()); print("[dpo]"); report=train_dpo(pol,ref,tok,triples,cfg)
    print(f"初始margin: {report['init_margin']:+.4f}  最终margin: {report['final_margin']:+.4f}")
    print(f"初始损失: {report['losses'][0]:.4f}  最终损失: {report['losses'][-1]:.4f}")
    return 0 if report['final_margin']>report['init_margin'] else 1

def make_prefs():
    return [
        {"prompt":"What is the capital of France?","chosen":"Paris.","rejected":"France is in Europe and has many cities including Paris."},
        {"prompt":"What is the capital of Japan?","chosen":"Tokyo.","rejected":"Japan is an island nation with government in Tokyo."},
        {"prompt":"Compute 2 + 3.","chosen":"5.","rejected":"2 plus 3 is close to 5 I believe."},
        {"prompt":"Compute 7 * 6.","chosen":"42.","rejected":"7 multiplied by 6 is around 42."},
        {"prompt":"Compute 12 / 4.","chosen":"3.","rejected":"Twelve divided by four is roughly three."},
        {"prompt":"List three colors.","chosen":"red, green, blue.","rejected":"Colors include red, green, and blue too."},
        {"prompt":"List three vowels.","chosen":"a, e, i.","rejected":"Vowels produce open mouth sounds like a, e, and i."},
        {"prompt":"Define variable.","chosen":"a name bound to a value.","rejected":"A variable is something you use to store stuff."},
        {"prompt":"Define function.","chosen":"a reusable block of code that returns an output.","rejected":"A function is something that does things on inputs."},
        {"prompt":"Python: print 42.","chosen":"print(42)","rejected":"You can print numbers in python."},
        {"prompt":"Python: sort items.","chosen":"items.sort()","rejected":"Sorting a list in python is easy."},
        {"prompt":"Python: get length.","chosen":"len(items)","rejected":"To get length call len on items."},
    ]

if __name__=="__main__": sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| DPO损失 | 直接偏好优化：对偏好的sigmoid损失 |
| 参考模型 | 冻结的SFT模型，对数概率不变 |
| 策略模型 | 从参考初始化，DPO训练中偏离 |
| chosen-rejected margin | chosen与rejected的log-prob差 |
| beta参数 | 控制KL约束强度 |

---

## 5. 面试考点

**Q1：DPO为什么不需要显式奖励模型？**
**Q2：参考模型的三个不变性是什么？**
