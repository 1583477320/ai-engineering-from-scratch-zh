"""视觉语言预训练——InfoNCE + LM 损失。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau=nn.Parameter(torch.tensor(math.log(0.07)))
    def forward(self,img_emb,txt_emb):
        tau=self.log_tau.exp()
        sim=F.normalize(img_emb,-1)@F.normalize(txtemb,-1).T/tau
        labs=torch.arange(sim.shape[0])
        return (F.cross_entropy(sim,labs)+F.cross_entropy(sim.T,labs))/2

def lm_loss(logits,targets,pad=0):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=pad)

def main():
    dim=64; vocab=256; enc=nn.Linear(dim,dim); proj=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    txt_emb=nn.Embedding(vocab,dim); dec=nn.TransformerDecoder(nn.TransformerDecoderLayer(dim,4,dim*4,batch_first=True),2)
    head=nn.Linear(dim,vocab); pos_emb=nn.Parameter(torch.randn(1,32,dim)*.02)
    opt=torch.optim.Adam(list(enc.parameters())+list(proj.parameters())+list(txt_emb.parameters())+
                         list(dec.parameters())+list(head.parameters())+[pos_emb]+list(InfoNCE().parameters()),lr=1e-3)
    ic=InfoNCE()
    for step in range(50):
        imgs=torch.randn(16,196,dim); caps=torch.randint(2,vocab,(16,8))
        ie=proj(enc(imgs.mean(1))); te=proj(txt_emb(caps).mean(1))
        cl=ic(ie,te)
        mem=enc(imgs).permute(1,0,2); tgt=caps[:,:-1]; tgt_emb=txt_emb(tgt)+pos_emb[:,:tgt.shape[1]]
        logits=dec(tgt_emb,mem); ll=lm_loss(logits,caps[:,1:])
        loss=cl+0.5*ll; opt.zero_grad(); loss.backward(); opt.step()
        if step%10==0: print(f"  步{step}: 对比={cl.item():.3f} LM={ll.item():.3f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
