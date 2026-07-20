"""视觉Transformer编码器——Pre-LN+12层。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self,dim,heads):
        super().__init__(); self.h=heads; self.d=dim//heads
        self.qkv=nn.Linear(dim,dim*3,bias=False); self.out=nn.Linear(dim,dim,bias=False)
    def forward(self,x):
        B,N,D=x.shape
        q,k,v=self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        a=(q@k.transpose(-2,-1))*self.d**-0.5; a=a.softmax(-1)
        return self.out((a@v).transpose(1,2).reshape(B,N,D))

class FFN(nn.Module):
    def __init__(self,dim,exp=4):
        super().__init__(); self.fc1=nn.Linear(dim,dim*exp); self.fc2=nn.Linear(dim*exp,dim)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.ln1=nn.LayerNorm(dim); self.attn=MHSA(dim,heads)
        self.ln2=nn.LayerNorm(dim); self.ffn=FFN(dim)
    def forward(self,x): return x+self.ffn(self.ln2(x+self.attn(self.ln1(x))))

class ViT(nn.Module):
    def __init__(self,dim=768,heads=12,depth=12):
        super().__init__()
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return self.norm(x)

def main():
    vit=ViT()
    x=torch.randn(2,197,768)
    out=vit(x)
    params=sum(p.numel() for p in vit.parameters())
    print(f"输出: {out.shape} 参数: {params/1e6:.2f}M")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
