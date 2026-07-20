"""端到端研究演示——组合5个课程。"""
from __future__ import annotations
import json, os, random, sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Hypothesis:
    id: int; branch: str
@dataclass
class BranchStats:
    runs: int = 0; reward_sum: float = 0.0
    @property
    def mean(self): return self.reward_sum/self.runs if self.runs>0 else 0.0

class NoTriggerError(Exception): pass

def run_scheduler(seeds):
    stats={}; triggers=[]
    for h in seeds:
        r=min(1,max(0,random.gauss(0.6,0.2)))
        if h.branch not in stats: stats[h.branch]=BranchStats()
        stats[h.branch].runs+=1; stats[h.branch].reward_sum+=r
        if stats[h.branch].mean>=0.7: triggers.append(h.branch)
    return {"per_branch":stats,"total_runs":len(seeds),"triggers":triggers,"stop":"queue_empty"}

def pick_best(triggers,stats):
    if not triggers: raise NoTriggerError("无论文触发")
    return max(triggers,key=lambda b:stats[b].mean)

def write_paper(branch,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,"paper.tex"),"w") as f:
        f.write(rf"\documentclass{{article}}\begin{{document}}\title{{{branch}}}\section{{引言}}研究{branch}\end{{document}}")
    manifest={"sections":["引言","方法","结果"],"branch":branch}
    with open(os.path.join(out_dir,"manifest.json"),"w") as f: json.dump(manifest,f,indent=2)
    return manifest

def run_demo(out_dir="/tmp/research_demo"):
    random.seed(42)
    seeds=[Hypothesis(i,f"topic_{chr(65+i)}") for i in range(3)]
    sched=run_scheduler(seeds)
    if not sched["triggers"]: raise NoTriggerError("无触发")
    best=pick_best(sched["triggers"],sched["per_branch"])
    return {"best":best,"reward":sched["per_branch"][best].mean,"manifest":write_paper(best,out_dir)}

def main():
    r=run_demo()
    print(f"最佳: {r['best']} 奖励={r['reward']:.3f} 章节={r['manifest']['sections']}")
    return 0

if __name__=="__main__": sys.exit(main())
