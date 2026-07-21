"""端到端分布式训练——DDP+ZeRO-1+检查点。"""
import torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist,torch.multiprocessing as mp,os
class MiniGPT(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(64,32); self.pos=nn.Embedding(16,32)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,128,batch_first=True),2); self.head=nn.Linear(32,64)
    def forward(self,x):
        B,T=x.shape; x=self.emb(x)+self.pos(torch.arange(T)); m=torch.triu(torch.ones(T,T)*float("-inf"),diagonal=1)
        return self.head(self.enc(x,mask=m))
def flatten(p): return torch.cat([q.data.view(-1) for q in p.parameters()])
def unflatten(m,flat):
    i=0
    for p in m.parameters(): n=p.numel(); p.data.view(-1).copy_(flat[i:i+n]); i+=n
class ZeroStep:
    def __init__(self,m,lr=0.01,ws=1,rank=0):
        f=flatten(m); t=f.numel(); ps=t//ws; self.off=rank*ps; self.end=self.off+ps if rank<ws-1 else t
        self.m=torch.zeros(self.end-self.off); self.v=torch.zeros(self.end-self.off)
        self.master=f[self.off:self.end].clone().float(); self.lr=lr; self.t=0
    def step(self):
        f=flatten(m).detach(); g=f[self.off:self.end].float(); self.t+=1
        self.m.mul_(0.9).add_(g,alpha=0.1); self.v.mul_(0.999).addcmul_(g,g,value=0.001)
        mh=self.m/(1-0.9**self.t); vh=self.v/(1-0.999**self.t)
        self.master.sub_(self.lr*mh/(vh.sqrt()+1e-8))
        unflatten(global_model,self.master)
def worker(rank,ws,d):
    global global_model
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]="29502"
    dist.init_process_group("gloo",rank=rank,world_size=ws)
    m=MiniGPT(); global_model=m
    for p in m.parameters(): dist.broadcast(p.data,src=0)
    z=ZeroStep(m,ws=ws,rank=rank); losses=[]
    for step in range(10):
        ids=torch.randint(0,64,(4,16)); loss=F.cross_entropy(m(ids)[:,:-1,:].reshape(-1,64),ids[:,1:].reshape(-1))
        loss.backward(); dist.all_reduce(flatten(m).detach(),op=dist.ReduceOp.SUM)
        flatten(m).div_(ws); z.step()
        losses.append(loss.item())
    if rank==0: d["losses"]=losses; d["final"]=losses[-1]
    dist.destroy_process_group()
def main():
    mgr=mp.Manager(); d=mgr.dict()
    mp.spawn(worker,args=(2,d),nprocs=2,join=True)
    losses=d.get("losses",[]); dr=losses[0]-losses[-1]
    print(f"初始损失={losses[0]:.3f} 最终损失={losses[-1]:.3f} 下降={dr:.3f} {'✓' if dr>0 else '✗'}")
    return 0
if __name__=="__main__":
    import sys; gy=nv.ex
    try: gy()
    except: raise SystemExit(main())
