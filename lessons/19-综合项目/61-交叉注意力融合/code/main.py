"""交叉注意力融合。"""
import torch, torch.nn as nn
class CrossAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h=h; self.d=d//h
        self.q=nn.Linear(d,d); self.kv=nn.Linear(d,d*2); self.out=nn.Linear(d,d)
    def forward(self, text, mem, mask=None):
        B,Nt,D=text.shape; q=self.q(text).view(B,Nt,self.h,self.d).transpose(1,2)
        kv=self.kv(mem); Nv=kv.shape[1]
        k=kv[:,:,:D].view(B,Nv,self.h,self.d).transpose(1,2)
        v=kv[:,:,D:].view(B,Nv,self.h,self.d).transpose(1,2)
        s=(q@k.transpose(-2,-1))*self.d**-.5
        if mask is not None: s=s.masked_fill(mask,float("-inf"))
        return self.out((s.softmax(-1)@v).transpose(1,2).reshape(B,Nt,D))

def causal_mask(n): return torch.triu(torch.ones(n,n))*float("-inf")

class DecoderBlock(nn.Module):
    def __init__(self,d,h):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d); self.ln3=nn.LayerNorm(d)
        self.sa=nn.MultiheadAttention(d,h,batch_first=True)
        self.ca=CrossAttention(d,h)
        self.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
    def forward(self,text,mem):
        B,N,D=text.shape; m=causal_mask(N).to(text.device)
        text=text+self.sa(self.ln1(text),self.ln1(text),self.ln1(text),attn_mask=m)[0]
        text=text+self.ca(self.ln2(text),mem)
        return text+self.ffn(self.ln3(text))

def main():
    d=64; h=4
    block=DecoderBlock(d,h); emb=nn.Embedding(256,d); head=nn.Linear(d,256)
    ids=torch.randint(0,256,(2,10)); mem=torch.randn(2,197,d)
    x=emb(ids); x=block(x,mem)
    print(f"输出: {x.shape} -> logits: {head(x).shape}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
