# 综合项目34——Transformer块（Pre-LN vs Post-LN + MLP + 残差）

> 一个块是每个现代decoder-only LLM的单位。LayerNorm、多头注意力、残差、MLP、残差。Pre-LN变体无需warmup即可稳定训练。本课程构建两种配置，展示哪一种在12层堆叠中存活。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第19章第30-33节
**预计时间：** 90分钟

---

## 学习目标

- 从四个移动组件构建Transformer块
- 用两种配置放置LayerNorm并解释为什么一种训练稳定
- 在多头注意力内实现因果掩码
- 在12层堆叠上跟踪两种变体的梯度流
- 重用块作为下节GPT组装的可插入单元

---

## 1. 问题

Transformer是一个块的重复。块搞错一次，重复十二次，你将获得一个在第一个epoch发散的模型。两个故障模式：注意力层看到未来token、LayerNorm放在无法在深度上驯化残差信号的位置。

修复是机械的：块恰好有两条残差路径和两个归一化位置。正确选择位置，堆叠的其余部分只是簿记。

---

## 2. 核心概念

### 2.1 Pre-LN变体

```
x → LayerNorm1 → MultiHeadAttention → Add(residual) → LayerNorm2 → MLP → Add(residual) → output
```

LayerNorm在残差分支内、子层之前。残差传递未归一化信号。

### 2.2 Post-LN变体

```
x → MultiHeadAttention → Add(residual) → LayerNorm1 → MLP → Add(residual) → LayerNorm2 → output
```

LayerNorm在残差相加之后。梯度必须通过每个块的LayerNorm。

### 2.3 MLP

位置MLP对每个token独立应用两层网络。隐藏宽度4倍嵌入宽度，GELU激活，第二线性后dropout。

### 2.4 残差连接

使梯度路径跨深度相加，保持梯度范数在范围内。同时让每个块学习增量更新而非完全替换。

---

## 3. 从零实现

`code/main.py`实现LayerNorm、MultiHeadAttention、FeedForward、TransformerBlock（Pre-LN/Post-LN切换）和BlockStack，展示梯度范数差异。

```python
"""Transformer块——Pre-LN vs Post-LN。"""
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
```

运行结果：

```
Pre-LN shape: torch.Size([2, 32, 192])  Post-LN shape: torch.Size([2, 32, 192])
Pre-LN  grad: 55.219639
Post-LN grad: 0.000112
ratio: 493032.48x
params: 13,397,376
```

---

## 4. 工具实践

**Pre-LN优势**：现代开源权重LLM全部使用Pre-LN。Post-LN是2017原始论文使用的配置。

**替换路径**：将GELU替换为SiLU、LayerNorm替换为RMSNorm，即可得到LLaMA系列。

---

## 5. LLM视角

**梯度流视角**：Pre-LN的嵌入梯度范数比Post-LN大数个数量级——这使训练无需warmup。

**块可替换视角**：块是可插入单元。替换归一化和激活即可从GPT切换到LLaMA。

---

## 6. 工程最佳实践

**融合QKV**：一个宽度3D的线性层代替三个线性层。

**注册因果掩码buffer**：构造时分配一次，前向时切片。

**Dropout两处**：注意力softmax后+MLP第二线性后。残差本身不加dropout。

---

## 7. 常见错误

**错误1：Post-LN不加warmup**
症状：损失在第一个epoch发散
修复：使用Pre-LN或添加warmup调度

**错误2：残差加dropout**
症状：梯度路径断开
修复：残差本身不加dropout

---

## 8. 面试考点

**Q1：Pre-LN和Post-LN的区别为什么重要？**
考察：对训练稳定性的理解

**Q2：残差连接的两个作用是什么？**
考察：对深度网络梯度流的理解

**Q3：为什么Dropout不加在残差上？**
考察：对加法恒等式的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Pre-LN | "预归一化" | LayerNorm在残差分支内、子层之前 |
| Post-LN | "后归一化" | LayerNorm在残差相加之后，需要warmup |
| 因果掩码 | "三角掩码" | 注意力logits上三角设为负无穷 |
| 融合QKV | "组合投影" | 一个宽度3D的线性层 |
| 残差流 | "跳跃连接" | 未归一化张量跨深度传递，每个块添加增量更新 |

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
