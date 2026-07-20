"""评测工具——固定任务、评分样本、pass@k。"""
from __future__ import annotations
import json, os, re, shutil, statistics, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class FixtureTask:
    id:str; goal:str; setup_dir:str; expected_dir:str; verifier_name:str; verifier_args:dict[str,Any]; root:str=""

@dataclass
class SampleResult:
    task_id:str; sample_index:int; latency_ms:float; cost_units:float=0.0; notes:str=""

@dataclass
class VerificationOutcome:
    passed:bool; detail:str

@dataclass
class TaskReport:
    task_id:str; k:int; passes:int; pass_rate:float; pass_at_k:float
    mean_latency_ms:float; p95_latency_ms:float; mean_cost:float; samples:list[dict]=field(default_factory=list)

@dataclass
class EvalReport:
    task_reports:list[TaskReport]; pass_at_1:float; pass_at_k:float; k:int
    mean_latency_ms:float; p95_latency_ms:float; total_cost:float

def pass_at_k(p,k):
    if k<=0: return 0.0; p=max(0.0,min(1.0,p)); return 1.0-(1.0-p)**k

def p95(values):
    if not values: return 0.0
    s=sorted(values); idx=max(0,int(round(0.95*len(s)))-1); return s[min(idx,len(s)-1)]

Verifier=Callable[[FixtureTask,str,dict],VerificationOutcome]

def verify_file_equals(task,scratch,args):
    rel=args.get("path")
    if not isinstance(rel,str): return VerificationOutcome(False,"missing path")
    actual=os.path.join(scratch,rel); expected=os.path.join(task.expected_dir,rel)
    if not os.path.isfile(actual): return VerificationOutcome(False,f"missing: {rel}")
    a=open(actual).read(); e=open(expected).read()
    a=a.rstrip("\n")+"\n"; e=e.rstrip("\n")+"\n"
    return VerificationOutcome(a==e,f"{'matches' if a==e else 'differs'}")

def verify_regex_match(task,scratch,args):
    rel=args.get("path"); pat=args.get("pattern")
    if not rel or not pat: return VerificationOutcome(False,"need path+pattern")
    actual=os.path.join(scratch,rel)
    if not os.path.isfile(actual): return VerificationOutcome(False,f"missing: {rel}")
    return VerificationOutcome(bool(re.search(pat,open(actual).read(),re.MULTILINE)),f"matched {pat!r}")

def verify_shell_exit_zero(task,scratch,args):
    argv=args.get("argv")
    if not argv: return VerificationOutcome(False,"need argv")
    try: proc=subprocess.run(list(argv),cwd=scratch,capture_output=True,timeout=float(args.get("timeout_seconds",10)))
    except (subprocess.TimeoutExpired,FileNotFoundError) as e: return VerificationOutcome(False,str(e))
    return VerificationOutcome(proc.returncode==0,f"exit {proc.returncode}")

VERIFIERS={"file_equals":verify_file_equals,"regex_match":verify_regex_match,"shell_exit_zero":verify_shell_exit_zero}

def load_fixture(task_dir):
    with open(os.path.join(task_dir,"task.json")) as f: spec=json.load(f)
    return FixtureTask(spec["id"],spec["goal"],os.path.join(task_dir,"buggy"),os.path.join(task_dir,"expected"),
        spec["verifier"]["name"],spec["verifier"].get("args",{}),task_dir)

def load_all(root):
    return [load_fixture(os.path.join(root,n)) for n in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root,n)) and os.path.isfile(os.path.join(root,n,"task.json"))]

Candidate=Callable[[FixtureTask,str],SampleResult]

def apply_known_fixes(task,scratch):
    s=time.perf_counter()
    if os.path.isdir(task.expected_dir):
        for dp,_,files in os.walk(task.expected_dir):
            rel=os.path.relpath(dp,task.expected_dir); dst=scratch if rel=="." else os.path.join(scratch,rel)
            os.makedirs(dst,exist_ok=True); [shutil.copy2(os.path.join(dp,f),os.path.join(dst,f)) for f in files]
    return SampleResult(task.id,0,(time.perf_counter()-s)*1000,1.0,"reference")

def noop_candidate(task,scratch): return SampleResult(task.id,0,0,0,"noop")

@dataclass
class EvalHarness:
    tasks:list[FixtureTask]; k:int=1; verifiers:dict[str,Verifier]=field(default_factory=lambda:dict(VERIFIERS))
    def run(self,candidate):
        trs=[]
        for task in self.tasks:
            lats=[]; costs=[]; passes=0; samples=[]
            for si in range(self.k):
                scratch=tempfile.mkdtemp(prefix=f"eval-{task.id}-")
                try:
                    sample=candidate(task,scratch)
                    outcome=self.verifiers[task.verifier_name](task,scratch,task.verifier_args)
                    lats.append(sample.latency_ms); costs.append(sample.cost_units)
                    if outcome.passed: passes+=1
                    samples.append({"i":si,"ms":round(sample.latency_ms,3),"pass":outcome.passed,"detail":outcome.detail})
                finally: shutil.rmtree(scratch,ignore_errors=True)
            pr=passes/self.k if self.k else 0
            trs.append(TaskReport(task.id,self.k,passes,pr,pass_at_k(pr,self.k),
                statistics.mean(lats) if lats else 0,p95(lats),statistics.mean(costs) if costs else 0,samples))
        p1=[min(1.0,r.pass_rate) for r in trs]; pk=[r.pass_at_k for r in trs]
        al=[s["ms"] for r in trs for s in r.samples]; tc=sum(s.get("cu",0) for r in trs for s in r.samples)
        return EvalReport(trs,statistics.mean(p1) if p1 else 0,statistics.mean(pk) if pk else 0,self.k,
            statistics.mean(al) if al else 0,p95(al),tc)

def _demo():
    d=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tasks")
    tasks=load_all(d)
    if not tasks: print("no fixtures"); return 1
    print(f"loaded {len(tasks)} tasks")
    r=EvalHarness(tasks,k=1).run(apply_known_fixes)
    print(json.dumps({"pass@1":round(r.pass_at_1,4),"mean_ms":round(r.mean_latency_ms,3),
        "tasks":[{"id":t.task_id,"passes":t.passes} for t in r.task_reports]},indent=2))
    return 0 if r.pass_at_1>=1.0 else 1

if __name__=="__main__": sys.exit(_demo())
