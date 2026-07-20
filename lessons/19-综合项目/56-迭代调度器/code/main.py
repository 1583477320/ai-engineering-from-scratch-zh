"""迭代调度器——UCB+并行+修剪。"""
from __future__ import annotations
import asyncio, math, random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class Hypothesis:
    id: int; branch: str; payload: Dict[str,Any] = field(default_factory=dict)
@dataclass
class Result:
    hypothesis_id: int; branch: str; reward: float
@dataclass
class BranchStats:
    runs: int = 0; reward_sum: float = 0.0
    @property
    def mean(self): return self.reward_sum/self.runs if self.runs>0 else 0.0
@dataclass
class SchedulerReport:
    per_branch: Dict[str,BranchStats]; total_runs: int
    paper_triggers: List[str]; stop_reason: str

class IterationScheduler:
    def __init__(self,runner,expander=None,n_slots=3,c=math.sqrt(2),pt=0.7,pf=0.2,pa=3,me=50,ms=120):
        self.runner=runner; self.expander=expander; self.n_slots=n_slots; self.c=c
        self.pt=pt; self.pf=pf; self.pa=pa; self.me=me; self.ms=ms
    def _ucb(self,br,stats,total):
        s=stats.get(br)
        if s is None or s.runs==0: return float("inf")
        return s.mean+self.c*math.sqrt(math.log(total)/s.runs)
    async def run(self,seeds):
        queue=list(seeds); stats={}; triggers=[]; fired=set()
        total_runs=0; start=asyncio.get_event_loop().time(); in_flight=set()
        while True:
            elapsed=asyncio.get_event_loop().time()-start
            if total_runs>=self.me or elapsed>=self.ms:
                for t in in_flight: t.cancel(); break
            while len(in_flight)<self.n_slots:
                if not queue: break
                tot=max(1,sum(s.runs for s in stats.values()))
                queue.sort(key=lambda h:-self._ucb(h.branch,stats,tot)); hyp=queue.pop(0)
                s=stats.get(hyp.branch)
                if s and s.runs>=self.pa and s.mean<self.pf: continue
                task=asyncio.create_task(self.runner(hyp)); in_flight.add(task)
            if not in_flight: break
            done,in_flight=await asyncio.wait(in_flight,return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                r=t.result(); total_runs+=1
                if r.branch not in stats: stats[r.branch]=BranchStats()
                stats[r.branch].runs+=1; stats[r.branch].reward_sum+=r.reward
                if r.branch not in fired and stats[r.branch].mean>=self.pt:
                    triggers.append(r.branch); fired.add(r.branch)
                if self.expander and r.reward>=self.pt:
                    for h in self.expander(r): queue.append(h)
        return SchedulerReport(stats,total_runs,triggers,
                               "queue_empty" if not queue else "max_experiments" if total_runs>=self.me else "deadline")

async def demo_runner(hyp):
    await asyncio.sleep(0.01); return Result(hyp.id,hyp.branch,min(1,max(0,random.gauss(0.5,0.2))))

def main():
    seeds=[Hypothesis(i,f"topic_{chr(65+i)}") for i in range(5)]
    r=asyncio.run(IterationScheduler(demo_runner,me=12).run(seeds))
    print(f"停止: {r.stop_reason} 实验: {r.total_runs}")
    for br,s in r.per_branch.items(): print(f"  {br}: {s.runs}次 平均{s.mean:.3f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
