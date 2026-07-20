"""视觉编码器图块——Conv2d投影+2D正弦位置。"""
import math, torch, torch.nn as nn

def sinusoidal_2d(gh,gw,dim):
    assert dim%4==0
    pe=torch.zeros(gh*gw,dim)
    for r in range(gh):
        for c in range(gw):
            i=r*gw+c; d=dim//2
            div=10000**(torch.arange(0,d,2)/d)
            pe[i,0:d:2]=torch.sin(torch.tensor(r/div))
            pe[i,1:d:2]=torch.cos(torch.tensor(r/div))
            pe[i,d::2]=torch.sin(torch.tensor(c/div))
            pe[i,d+1::2]=torch.cos(torch.tensor(c/div))
    return pe

class PatchEmbed(nn.Module):
    def __init__(self,img=224,patch=16,ch=3,dim=768):
        super().__init__()
        self.patch=patch; self.g=img//patch; self.n=self.g**2
        self.proj=nn.Conv2d(ch,dim,patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class VisionFrontEnd(nn.Module):
    def __init__(self,img=224,patch=16,ch=3,dim=768):
        super().__init__()
        self.pe_=PatchEmbed(img,patch,ch,dim)
        self.cls=nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.register_buffer("pe2d",sinusoidal_2d(self.pe_.g,self.pe_.g,dim))
    def forward(self,x):
        B=x.shape[0]; y=self.pe_(x)
        z=torch.cat([torch.zeros(1,1,self.pe2d.shape[1]),self.pe2d.unsqueeze(0)],dim=1)
        return torch.cat([self.cls.expand(B,-1,-1),y],dim=1)+z

def main():
    m=VisionFrontEnd()
    out=m(torch.randn(2,3,224,224))
    print(f"输出: {out.shape}") # (2,197,768)
    print(f"图块数: {m.pe_.n}")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
