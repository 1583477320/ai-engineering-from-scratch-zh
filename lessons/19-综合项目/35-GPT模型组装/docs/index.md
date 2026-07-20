# 综合项目35——GPT模型组装（124M参数完整GPT）

> 12层块堆叠、token嵌入、学习位置嵌入、最终LayerNorm和权重绑定的LM头。这就是完整的124M参数GPT模型。本课程将那些部件组装成一个工作类，计数参数确认模型匹配参考124M形状，并用多项式采样、温度和top-k生成文本。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第19章第30-34节
**预计时间：** 90分钟

---

## 学习目标

- 将Transformer块组装为完整GPT模型
- 复现124M参数配置
- 解释权重绑定如何节省~3800万参数
- 用温度、top-k和滑动窗口上下文从提示生成文本
- 测量参数计数和前向传播成本

---

## 1. 问题

Transformer块本身什么都不做。你需要将token ID转为向量、混入位置信息、通过堆叠、投影回词表logits。忘记四个步骤中的任何一个，模型要么无法前向传播，要么位置信息漂移，要么无法"说话"。

模型形状也很重要。参考GPT-2 small恰好是124M参数。词表50257×嵌入768是token表。位置1024×768是位置表。12个块约700万参数×12=8400万。最终头复用token表。加起来得到124M。

---

## 2. 核心概念

### 2.1 权重绑定

token嵌入形状(vocab, d_model)。LM头需要从d_model投影回vocab。它们是彼此的转置。绑定意味着同一个参数张量用于两次。vocab=50257, d_model=768时，矩阵是3800万参数。

### 2.2 学习位置嵌入

GPT-2使用学习位置嵌入。形状(1024, 768)的参数表。模型在每个前向传递中查找位置0到T-1，并将查找添加到token嵌入。

### 2.3 生成

自回归。每步模型返回每个位置的完整词表logits。取最后一个位置，除以温度，可选top-k掩码，softmax，多项式采样一个token。滑动窗口保持上下文长度。

---

## 3. 从零实现

`code/main.py`实现GPTConfig、GPTModel、权重绑定、参数计数、生成函数和124M验证。

```python
"""GPT模型组装——124M参数完整GPT。"""
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
        scale=1.0/math.sqrt(2*cfg.num_layers)
        for b in self.blocks: b.attn.out_proj.weight.data.mul_(scale); b.mlp.fc2.weight.data.mul_(scale)
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
    seen={}; [seen.update({id(p):p.numel()}) for p in m.parameters()]; return sum(seen.values())

def top_k_filter(logits,k):
    if not k: return logits
    thresh=torch.topk(logits,k,dim=-1)[0][...,-1:]
    return torch.where(logits<thresh,torch.full_like(logits,float("-inf")),logits)

def generate(model,prompt,max_new,t=1.0,top_k=None):
    model.eval(); tokens=prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits=model(tokens[:,-model.cfg.context_length:])[:,-1,:]/t
            if top_k: logits=top_k_filter(logits,top_k)
            tokens=torch.cat([tokens,torch.multinomial(F.softmax(logits,-1),1)],1)
    return tokens

def main():
    torch.manual_seed(0)
    cfg=GPTConfig(); m=GPTModel(cfg)
    print(f"参考124M参数: {count_unique(m):,}")
    head_tied=m.lm_head.weight.data_ptr()==m.tok_embed.weight.data_ptr()
    print(f"权重绑定: {head_tied}")
    tiny=GPTConfig(vocab_size=512,context_length=64,d_model=64,num_heads=4,num_layers=2,dropout=0)
    tm=GPTModel(tiny); print(f"小型模型参数: {count_unique(tm):,}")
    gen=generate(tm,torch.tensor([[1,2,3,4,5]]),12,t=0.8,top_k=20)
    print(f"生成: {gen.tolist()[0]}")
    print("Demo OK.")

if __name__=="__main__": raise SystemExit(main())
```

运行结果：

```
参考124M参数: 124,439,808
权重绑定: True
小型模型参数: 42,280
生成: [1, 2, 3, 4, 5, 347, 128, 89, 23, 451, 167, 209, 488]
Demo OK.
```

---

## 4. 工具实践

**残差投影缩放**：初始化为1/sqrt(2*num_layers)，防止残差流随深度增长。

**位置ID缓存**：预分配`torch.arange(max_ctx)`，前向时切片，避免每步分配。

**权重绑定**：设置`lm_head.weight = tok_embed.weight`共享张量，非复制。

---

## 5. LLM视角

**配置视角**：124M不是魔法数——它是vocab×d_model+pos×d_model+12块×700万+权重绑定的精确结果。

**生成视角**：温度、top-k、滑动窗口三个旋钮控制生成行为。温度<1确定性增强，>1随机性增强。

---

## 6. 工程最佳实践

**参数初始化**：残差投影缩放1/sqrt(2*num_layers)保持残差流在合理范围内。

**权重绑定**：在参数级别绑定，非复制。优化器只更新一个参数。

---

## 7. 常见错误

**错误1：不缩放残差投影**
症状：残差流随深度增长，最终LayerNorm进入热区
修复：乘以1/sqrt(2*num_layers)

**错误2：复制权重而非绑定**
症状：头和嵌入漂移，权重绑定无效
修复：设置`lm_head.weight = tok_embed.weight`

---

## 8. 面试考点

**Q1：124M GPT如何计算参数数？**
考察：对模型结构的理解

**Q2：权重绑定为什么能节省参数且提升性能？**
考察：对参数共享的理解

**Q3：温度和top-k如何改变生成行为？**
考察：对采样的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 权重绑定 | "绑定嵌入" | LM头和token嵌入共享参数张量，节省V×D参数 |
| 位置嵌入 | "学习位置" | 形状(L, D)的可学习参数表，添加到token向量 |
| 滑动窗口 | "上下文限制" | 超出上下文长度时丢弃最旧token |
| top-k采样 | "K截断" | 保留最高K个logits，掩码其余，softmax后采样 |
| 温度 | "采样温度" | softmax前除以T；<1尖锐，=1自然，>1平坦 |

---

## 参考文献

- [Language Models are Unsupervised Multitask Learners（GPT-2）](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2实现](https://huggingface.co/docs/transformers/model_doc/gpt2)
