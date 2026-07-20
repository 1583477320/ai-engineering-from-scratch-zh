"""投机解码服务器——draft/verify调度器脚手架。"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

VOCAB = list("abcdefghij")
def softmax_from(seed):
    rnd=random.Random(seed); w=[rnd.random() for _ in VOCAB]; t=sum(w); return [x/t for x in w]
def sample(dist, rng):
    r=rng.random(); acc=0
    for i,p in enumerate(dist): acc+=p;
    if r<=acc: return i
    return len(dist)-1

@dataclass
class TargetModel:
    calls:int=0; tokens_verified:int=0
    def distribution(self, ctx): return softmax_from(ctx*7+13)
    def verify(self, draft_tokens, ctx, rng):
        self.calls+=1; self.tokens_verified+=len(draft_tokens)+1; acc=[]
        for pos,tok in enumerate(draft_tokens):
            d=self.distribution(ctx+pos)
            if d[tok]>=0.5*max(d): acc.append(tok)
            else: break
        return acc, sample(self.distribution(ctx+len(acc)), rng)

@dataclass
class DraftModel:
    calls:int=0; alignment:float=0.80
    def propose(self, ctx, k, rng, target):
        self.calls+=1; out=[]
        for pos in range(k):
            d=target.distribution(ctx+pos)
            out.append(max(range(len(d)),key=lambda i:d[i]) if rng.random()<self.alignment else sample(d,rng))
        return out

@dataclass
class Metrics:
    generated:int=0; target_calls:int=0; draft_calls:int=0; accepted_sum:int=0
    def acceptance_rate(self, k): return self.accepted_sum/(self.target_calls*k) if self.target_calls else 0
    def tokens_per_call(self): return self.generated/max(1,self.target_calls)

def speculative_decode(n, k, rng, target, draft):
    m=Metrics(); ctx=1
    while m.generated<n:
        dt=draft.propose(ctx,k,rng,target); m.draft_calls+=1
        acc,nxt=target.verify(dt,ctx,rng); m.target_calls+=1; m.accepted_sum+=len(acc)
        for _ in acc: m.generated+=1; ctx+=1;
        if m.generated>=n: break
        m.generated+=1; ctx+=1
    return m

def baseline_decode(n, rng, target):
    m=Metrics(); ctx=1
    while m.generated<n: target.calls+=1; m.target_calls+=1; sample(target.distribution(ctx),rng); m.generated+=1; ctx+=1
    return m

def main():
    n=500; target=TargetModel(); base=baseline_decode(n,random.Random(7),target)
    print(f"基线: {base.target_calls}次调用, {base.tokens_per_call():.2f}tok/call")
    for al in (0.60,0.75,0.90):
        for k in (2,4,6):
            t=TargetModel(); d=DraftModel(alignment=al)
            m=speculative_decode(n,k,random.Random(7),t,d)
            spd=base.target_calls/max(1,m.target_calls)
            print(f"  对齐={al:.2f} k={k} 调用={m.target_calls:3d} 接受率={m.acceptance_rate(k):.2f} tok/call={m.tokens_per_call():.2f} 加速={spd:.2f}x")

if __name__=="__main__": main()
