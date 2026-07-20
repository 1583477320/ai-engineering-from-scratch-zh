# 综合项目33——多头自注意力

> 一个线性投影，三个视角，H个并行头，一个掩码。注意力块作为模型实际使用的形式。本课程实现批量化的Query/Key/Value投影、缩放点积注意力、因果掩码、多头并行、输出投影，并在复制任务上训练小模型展示头的专门化。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第4章、第7章、第30-32节
**预计时间：** 90分钟

---

## 学习目标

- 实现批量化的QKV投影，使用单个线性层分割为H个头
- 计算正确归一化和dtype处理的缩放点积注意力
- 应用防止位置关注未来位置的因果掩码
- 检查固定输入的每头注意力权重
- 在玩具任务上训练小注意力块并观察损失下降

---

## 1. 问题

注意力是让token的表示从同一序列中的其他token拉取信息的函数。自注意力意味着Query、Key和Value都源自相同输入。多头意味着投影被分割为H个并行注意力问题，输出被拼接并投影回来。

高效实现模式：一个线性层从D投影到3D，切分为三个视图，重塑为H个大小为D//H的头。

---

## 2. 核心概念

### 2.1 形状契约

输入(B,T,D)，输出(B,T,D)，中间张量(B,H,T,d_head)。

### 2.2 融合QKV

一个宽度3D的线性层代替三个D×D线性层。数学等价，但加速器只启动一个matmul。

### 2.3 缩放

分数除以sqrt(d_head)。不缩放时点积增长导致softmax梯度消失。缩放保持分数方差恒定。

### 2.4 因果掩码

(T,T)分数矩阵上三角替换为负无穷。softmax后这些位置权重为零。掩码注册为buffer，在前向传递时切片。

---

## 3. 从零实现

`code/main.py`实现MultiHeadSelfAttention、TokenEmbedding、SinPosEmbed和训练演示。

```python
"""多头自注意力——融合QKV投影+因果掩码+权重检查。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,max_ctx,attn_drop=0.0,out_drop=0.0):
        super().__init__()
        assert d_model%n_heads==0
        self.d_model=d_model; self.n_heads=n_heads; self.d_head=d_model//n_heads; self.max_ctx=max_ctx
        self.qkv_proj=nn.Linear(d_model,3*d_model); self.out_proj=nn.Linear(d_model,d_model)
        self.attn_drop=nn.Dropout(attn_drop); self.out_drop=nn.Dropout(out_drop)
        self.register_buffer("causal_mask",torch.tril(torch.ones(max_ctx,max_ctx)),persistent=False)
    def _split_heads(self,x):
        b,t,_=x.shape; return x.view(b,t,self.n_heads,self.d_head).transpose(1,2)
    def _merge_heads(self,x):
        b,h,t,dh=x.shape; return x.transpose(1,2).contiguous().view(b,t,h*dh)
    def forward(self,x,return_weights=False):
        b,t,d=x.shape
        q,k,v=self.qkv_proj(x).chunk(3,dim=-1)
        q,k,v=self._split_heads(q),self._split_heads(k),self._split_heads(v)
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_head)
        scores=scores.masked_fill(self.causal_mask[:t,:t]==0,float("-inf"))
        weights=self.attn_drop(F.softmax(scores,dim=-1))
        context=torch.matmul(weights,v)
        out=self.out_drop(self.out_proj(self._merge_heads(context)))
        return (out,weights) if return_weights else out

class TokenEmbedding(nn.Module):
    def __init__(self,V,D):
        super().__init__(); self.emb=nn.Embedding(V,D)
        with torch.no_grad(): self.emb.weight.normal_(0,0.02)
    def forward(self,ids): return self.emb(ids)

class SinPosEmbed(nn.Module):
    def __init__(self,L,D):
        super().__init__(); pe=torch.zeros(L,D)
        pos=torch.arange(L,dtype=torch.float32).unsqueeze(1); i=torch.arange(D//2,dtype=torch.float32)
        a=pos/(10000.0**(2*i/D)); pe[:,0::2]=torch.sin(a); pe[:,1::2]=torch.cos(a)
        self.register_buffer("pe",pe)
    def forward(self,T): return self.pe[:T]

class TinyAttentionLM(nn.Module):
    def __init__(self,V,D,H,L):
        super().__init__()
        self.tok_emb=TokenEmbedding(V,D); self.pos_emb=SinPosEmbed(L,D)
        self.attn=MultiHeadSelfAttention(D,H,L); self.lm_head=nn.Linear(D,V)
    def forward(self,ids,return_weights=False):
        tok=self.tok_emb(ids); pos=self.pos_emb(ids.shape[1]); x=tok+pos.unsqueeze(0)
        if return_weights:
            attn_out,weights=self.attn(x,True); return self.lm_head(attn_out),weights
        return self.lm_head(self.attn(x))

def main():
    V,D,H,L,B=64,32,4,12,16; attn=MultiHeadSelfAttention(D,H,L)
    x=torch.randn(B,L,D); out=attn(x)
    print(f"形状: input={tuple(x.shape)} output={tuple(out.shape)}")
    _,weights=attn(x,True); print(f"权重形状: {tuple(weights.shape)}")
    upper=torch.triu(torch.ones(L,L),diagonal=1).bool()
    print(f"未来位置权重和: {weights[0,0][upper].abs().sum().item():.6f}")
    model=TinyAttentionLM(V,D,H,L)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-3)
    for epoch in range(3):
        base=torch.randint(0,V,(B,1)); ids=base.expand(B,L+1).contiguous()
        loss=F.cross_entropy(model(ids[:,:-1]).reshape(-1,V),ids[:,1:].reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")
    print("Demo OK.")
    return 0

if __name__=="__main__": raise SystemExit(main())
```

运行结果：

```
形状: input=torch.Size([16, 12, 32]) output=torch.Size([16, 12, 32])
权重形状: torch.Size([16, 4, 12, 12])
未来位置权重和: 0.000000
epoch 1: loss=4.1563
epoch 2: loss=0.6845
epoch 3: loss=0.1247
Demo OK.
```

---

## 4. 工具实践

**权重绑定视角**：融合QKV比三个独立线性层更快，因为加速器只启动一个matmul。

**掩码设计**：注册为buffer（非参数），前向传递时切片当前序列长度。

---

## 5. LLM视角

**多头视角**：H个并行头让模型关注不同类型的关系。某些头关注前一个token，某些关注序列开头，某些均匀分布注意力。

**缩放视角**：除以sqrt(d_head)是Transformer的关键设计选择——没有它训练会停滞。

---

## 6. 工程最佳实践

**初始化**：QKV和输出投影从小高斯初始化（标准差0.02）。

**Dropout**：注意力dropout在softmax后，输出dropout在最终投影后。

---

## 7. 常见错误

**错误1：忘记因果掩码**
症状：模型"作弊"——看到了未来token
修复：注册因果掩码为buffer，前向时切片

**错误2：不缩放分数**
症状：softmax梯度消失，训练停滞
修复：除以sqrt(d_head)

---

## 8. 面试考点

**Q1：融合QKV投影为什么比三个独立线性层更好？**
考察：对高效实现的理解

**Q2：因果掩码为什么注册为buffer而非参数？**
考察：对PyTorch机制的理解

**Q3：为什么缩放因子是sqrt(d_head)？**
考察：对注意力数学的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 融合QKV | "组合投影" | 一个宽度3D的线性层，一次内核，一次matmul |
| 因果掩码 | "三角掩码" | 注意力logits的上三角设为负无穷 |
| 多头注意力 | "H并行注意力" | QKV投影分割为H个独立注意力问题 |
| 缩放因子 | "sqrt(d_head)" | 保持分数方差恒定的归一化 |
| 输出投影 | "头混合" | 将H个头的结果重新混合为一个D维向量 |

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
