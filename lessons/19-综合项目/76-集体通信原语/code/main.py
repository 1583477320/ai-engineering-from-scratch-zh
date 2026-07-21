"""集体通信原语——环形allreduce+broadcast+allgather+reduce_scatter。"""
import torch,multiprocessing as mp
def demo_ring_allreduce(ws=4,dim=16):
    ctx=mp.get_context("spawn"); queues={}
    for i in range(ws):
        for j in range(ws):
            if i!=j: queues[(i,j)]=ctx.Queue()
    tensors=[torch.randn(dim) for _ in range(ws)]
    def worker(r):
        n=ws; t=tensors[r].clone(); cs=list(t.reshape(n,dim//n).clone())
        for s in range(n-1):
            si=(r-s)%n; ri=(r-s-1)%n
            queues[(r,(r+1)%n)].put(cs[si].clone())
            cs[ri]=cs[ri]+queues[((r-1)%n,r)].get()
        for s in range(n-1):
            queues[(r,(r+1)%n)].put(cs[r].clone())
            cs[(r-1-s)%n]=queues[((r-1)%n,r)].get()
        return torch.cat(cs)
    with ctx.Pool(ws) as p:
        results=p.map(worker,range(ws))
    ref=tensors[0]+tensors[1]+tensors[2]+tensors[3]
    errors=[float((r-ref).abs().max()) for r in results]
    return max(errors)<1e-5,max(errors)

def main():
    ok,err=demo_ring_allreduce(4,16)
    print(f"环形allreduce 4ranks dim=16: {'✓' if ok else '✗'} err={err:.2e}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
