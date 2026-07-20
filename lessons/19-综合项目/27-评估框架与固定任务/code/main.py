"""评估框架——固定任务+确定性验证+pass@k。"""
import json,os,re,shutil,tempfile,time,math
from dataclasses import dataclass,field
from typing import List,Dict,Callable,Any

@dataclass
class TaskSpec:
    id:str; goal:str; setup:Dict[str,str]=field(default_factory=dict); verifier:Dict[str,Any]=field(default_factory=dict)
@dataclass
class SampleResult:
    success:bool; latency_ms:float; cost:float=0.0
@dataclass
class TaskReport:
    task_id:str; k:int; passes:int; pass_rate:float; mean_latency_ms:float
@dataclass
class EvalReport:
    total_tasks:int; pass_at_1:float; pass_at_k:float; mean_latency_ms:float
    def to_dict(self):
        return {"total_tasks":self.total_tasks,"pass_at_1":round(self.pass_at_1,4),
                "pass_at_k":round(self.pass_at_k,4),"mean_latency_ms":round(self.mean_latency_ms,2)}

def verify_file_equals(scratch,filename,expected):
    p=os.path.join(scratch,filename)
    return os.path.exists(p) and open(p).read().strip()==expected.strip()
def verify_regex_match(scratch,filename,pattern):
    p=os.path.join(scratch,filename)
    return os.path.exists(p) and bool(re.search(pattern,open(p).read()))
VERIFIERS={"file_equals":verify_file_equals,"regex_match":verify_regex_match}

class EvalHarness:
    def __init__(self,k=5): self.k=k
    def run(self,tasks,candidate_fn):
        task_reports=[]; all_lat=[]; total_pass=0
        for task in tasks:
            passes=0; lats=[]
            for _ in range(self.k):
                scratch=tempfile.mkdtemp()
                try:
                    for fn,ct in task.setup.items():
                        os.makedirs(os.path.join(scratch,os.path.dirname(fn)),exist_ok=True)
                        open(os.path.join(scratch,fn),"w").write(ct)
                    t0=time.perf_counter(); result=candidate_fn(task,scratch)
                    lat=(time.perf_counter()-t0)*1000
                    vn=task.verifier.get("type","file_equals")
                    if vn in VERIFIERS:
                        passed=VERIFIERS[vn](scratch,**{k:v for k,v in task.verifier.items() if k!="type"})
                    else: passed=result.success
                    if passed: passes+=1
                    lats.append(lat)
                finally: shutil.rmtree(scratch,ignore_errors=True)
            pr=passes/self.k; lats.sort()
            p95=lats[max(0,int(len(lats)*0.95)-1)] if lats else 0
            task_reports.append(TaskReport(task.id,self.k,passes,pr,sum(lats)/max(len(lats),1)))
            total_pass+=passes
        n=len(tasks); ts=n*self.k
        pa1=total_pass/ts if ts else 0; pak=1-(1-pa1)**self.k if pa1<1 else 1
        all_lat=sum((r.mean_latency_ms for r in task_reports),[]) if False else [r.mean_latency_ms for r in task_reports]
        return EvalReport(n,pa1,pak,sum(r.mean_latency_ms for r in task_reports)/max(len(task_reports),1))

def demo_candidate(task,scratch): return SampleResult(success=True,latency_ms=1.0)

def build_fixtures():
    return [TaskSpec("fizz","Fix off-by-one",{"src/fizz.py":"def fizzbuzz(n):\n    for i in range(1, n):\n        if i%15==0: print('FizzBuzz')\n        elif i%3==0: print('Fizz')\n        elif i%5==0: print('Buzz')\n        else: print(i)"},
                     {"type":"file_equals","filename":"src/fizz.py","expected_content":"def fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i%15==0: print('FizzBuzz')\n        elif i%3==0: print('Fizz')\n        elif i%5==0: print('Buzz')\n        else: print(i)"}),
            TaskSpec("fact","Fix missing return",{"src/fact.py":"def factorial(n):\n    result=1\n    for i in range(2,n+1): result*=i\n    # missing return"},
                     {"type":"regex_match","filename":"src/fact.py","pattern":r"return\s+result"})]

def main():
    r=EvalHarness(k=3).run(build_fixtures(),demo_candidate)
    print(json.dumps(r.to_dict(),indent=2)); return 0
if __name__=="__main__": import sys; sys.exit(main())
