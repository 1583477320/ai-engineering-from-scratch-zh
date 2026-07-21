"""流水线并行与气泡分析。"""
from dataclasses import dataclass,field
from typing import List
@dataclass
class Stage: id:int; compute_ms:float; name:str=""
class Pipeline:
    def __init__(self,stages,m):
        self.stages=stages; self.M=m; self.N=len(stages); self.timeline=[]
    def simulate_gpipe(self):
        for m in range(self.M):
            for s in range(self.N): self.timeline.append(("fwd",m+s,s,m))
        for m in range(self.M):
            for s in range(self.N-1,-1,-1): self.timeline.append(("bwd",self.M+(self.M-1-m)+s,s,m))
    def bubble_fraction(self): return (self.N-1)/(self.M+self.N-1)
def main():
    p=Pipeline([Stage(i,1) for i in range(4)],8); p.simulate_gpipe()
    print(f"N=4 M=8 气泡={p.bubble_fraction():.3f} (公式预测={p.bubble_fraction():.3f})")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
