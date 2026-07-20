"""投影层与模态对齐——MLP+余弦损失。"""
import torch, torch.nn as nn, torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(self,in_dim=768,h=1024,out=512):
        super().__init__(); self.fc1=nn.Linear(in_dim,h); self.fc2=nn.Linear(h,out)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class MockTextEmbedding(nn.Module):
    def __init__(self,vocab=128,dim=512,seed=42):
        super().__init__(); torch.manual_seed(seed)
        self.table=nn.Embedding(vocab,dim)
        for p in self.parameters(): p.requires_grad=False
    def forward(self,ids): return self.table(ids)

def cosine_loss(img_emb,txt_emb):
    return (1-F.cosine_similarity(F.normalize(img_emb,-1),F.normalize(txt_emb,-1))).mean()

def main():
    proj=MLPProjector(); te=MockTextEmbedding()
    opt=torch.optim.Adam(proj.parameters(),lr=1e-3)
    pairs=[(torch.randn(768),torch.randint(2,128,(8,))) for _ in range(32)]
    print(f"训练 ({sum(p.numel() for p in proj.parameters()):,} 参数)")
    for step in range(200):
        total=0
        for f,c in pairs:
            loss=cosine_loss(proj(f.unsqueeze(0)),te(c.unsqueeze(0)).mean(1))
            opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
        if step%50==0: print(f"  步{step:3d}: 损失={total/32:.4f}")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
