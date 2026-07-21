"""数据并行DDP——广播+allreduce。"""
import torch,torch.nn as nn,torch.optim as optim,torch.distributed as dist,torch.multiprocessing as mp,os
class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))
    def forward(self,x): return self.net(x)
def worker(rank,ws,d):
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]="29500"
    dist.init_process_group("gloo",rank=rank,world_size=ws)
    m=MLP()
    for p in m.parameters(): dist.broadcast(p.data,src=0)
    opt=optim.SGD(m.parameters(),lr=0.01)
    torch.manual_seed(rank)
    for step in range(10):
        x=torch.randn(4,32); y=torch.randn(4,32); opt.zero_grad()
        loss=nn.MSELoss()(m(x),y); loss.backward()
        for p in m.parameters():
            dist.all_reduce(p.grad,op=dist.ReduceOp.SUM); p.grad/=ws
        opt.step()
        if rank==0 and step%5==0: print(f"  step{step}: loss={loss.item():.4f}")
    d["done"]=True; dist.destroy_process_group()
def main():
    d=mp.Manager().dict()
    mp.spawn(worker,args=(2,d),nprocs=2,join=True)
    print(f"✓DDP {d.get('done')}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
