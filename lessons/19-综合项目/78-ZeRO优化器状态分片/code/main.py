"""ZeRO Stage 1优化器状态分片。"""
import torch,torch.nn as nn
def flatten_params(m): return torch.cat([p.data.view(-1) for p in m.parameters()])
class ZeroOpt:
    def __init__(self,m,lr=0.01,ws=1,rank=0):
        flat=flatten_params(m); total=flat.numel(); ps=total//ws
        self.off=rank*ps; self.end=self.off+ps if rank<ws-1 else total
        self.m=torch.zeros(self.end-self.off); self.v=torch.zeros(self.end-self.off)
        self.master=flat[self.off:self.end].clone().float()
        self.lr=lr; self.t=0
    def step(self,fg):
        g=fg[self.off:self.end].float(); self.t+=1
        self.m.mul_(0.9).add_(g,alpha=0.1); self.v.mul_(0.999).addcmul_(g,g,value=0.001)
        m_h=self.m/(1-0.9**self.t); v_h=self.v/(1-0.999**self.t)
        self.master.sub_(self.lr*m_h/(v_h.sqrt()+1e-8))
        return self.master
def main():
    m=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
    z=ZeroOpt(m,ws=2,rank=0)
    for step in range(5):
        x=torch.randn(4,32); y=torch.randn(4,32); nn.MSELoss()(m(x),y).backward()
        fg=flatten_params(m).detach().clone()
        s=z.step(fg)
        print(f"  step{step}: shard_size={s.numel()}")
    print(f"✓ZeRO-1 {s.numel()}参数")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
