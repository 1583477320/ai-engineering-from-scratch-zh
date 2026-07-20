"""GitHub issue到PR异步云智能体脚手架。"""
from __future__ import annotations
import random, time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto

@dataclass
class Task:
    task_id: int; repo: str; issue_num: int; title: str; created_at: float = field(default_factory=time.time)

@dataclass
class BudgetLedger:
    daily_dollar_cap: float = 50.0; daily_pr_cap: int = 5; per_task_dollar_cap: float = 20.0
    spent_today: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prs_today: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    def permit(self, repo, est):
        wc=self.per_task_dollar_cap
        if self.spent_today[repo]+wc>self.daily_dollar_cap: return False,"daily $ cap"
        if self.prs_today[repo]>=self.daily_pr_cap: return False,"daily PR cap"
        return True,"ok"
    def record(self, repo, spent, opened):
        self.spent_today[repo]+=spent
        if opened: self.prs_today[repo]+=1

@dataclass
class InstallationToken:
    repo: str; expires_at: float; permissions: dict = field(default_factory=dict)
    @classmethod
    def mint(cls, r): return cls(repo=r, expires_at=time.time()+3600, permissions={"issues":"rw","pull_requests":"rw","contents":"rw","workflows":"r"})
    def can(self, a):
        return not (a=="force_push" or a.startswith("write:main"))

class SState(Enum):
    CLONE=auto(); INFER=auto(); AGENT=auto(); VERIFY=auto(); PR=auto(); DONE=auto(); FAILED=auto()

@dataclass
class SandboxRun:
    task: Task; state: SState=SState.CLONE; turns:int=0; dollars:float=0; wall_min:float=0
    coverage_delta:float=0; ci_green:bool=False; pr_opened:bool=False; failure:str|None=None

def run_agent(run,diff,rng,tc=20,dc=20.0,mc=30.0):
    run.state=SState.AGENT; tp=max(0.05,0.35*(1-diff)); tm=0.9+diff*0.6; tu=0.25+diff*0.45
    while True:
        run.turns+=1; run.wall_min+=tm; run.dollars+=tu
        if run.turns>=tc: run.failure="turn_cap"; run.state=SState.FAILED; return
        if run.dollars>=dc: run.failure="dollar_cap"; run.state=SState.FAILED; return
        if run.wall_min>=mc: run.failure="minute_cap"; run.state=SState.FAILED; return
        if rng.random()<tp: run.state=SState.VERIFY; return

def run_verify(run,diff,rng):
    if rng.random()<0.05: run.failure="flaky_test"; run.state=SState.FAILED; return
    run.ci_green=True; run.coverage_delta=rng.gauss(0,0.6)
    if run.coverage_delta<-2: run.failure="coverage_regression"; run.state=SState.FAILED; return
    run.state=SState.PR

def open_pr(run,token):
    if time.time()>=token.expires_at: run.failure="token_expired"; run.state=SState.FAILED; return
    run.pr_opened=True; run.state=SState.DONE

def dispatch(task,ledger,rng):
    d=rng.uniform(0.3,0.92); ok,reason=ledger.permit(task.repo,2+d*8)
    if not ok: r=SandboxRun(task); r.failure=f"disp:{reason}"; r.state=SState.FAILED; return r
    t=InstallationToken.mint(task.repo); r=SandboxRun(task); r.state=SState.INFER
    run_agent(r,d,rng)
    if r.state==SState.VERIFY: run_verify(r,d,rng)
    if r.state==SState.PR: open_pr(r,t)
    ledger.record(task.repo,r.dollars,r.pr_opened); return r

def main():
    rng=random.Random(9); ledger=BudgetLedger(); repos=["acme/widget","acme/service","acme/library"]
    runs=[dispatch(Task(i,rng.choice(repos),800+i,f"fix {i}"),ledger,rng) for i in range(20)]
    op=sum(1 for r in runs if r.pr_opened); fa=sum(1 for r in runs if r.state==SState.FAILED)
    print(f"PR打开: {op}  失败: {fa}")
    reasons=defaultdict(int)
    for r in runs:
        if r.failure: reasons[r.failure]+=1
    print("失败:",dict(reasons.items()))
    if op: print(f"通过集: 平均$={sum(r.dollars for r in runs if r.pr_opened)/op:.2f}")

if __name__=="__main__": main()
