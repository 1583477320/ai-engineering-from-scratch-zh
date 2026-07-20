"""端到端编码智能体——策略+门+沙箱+追踪+评估。"""
import os,re,time,json,uuid,shutil,tempfile,subprocess
from dataclasses import dataclass,field
from typing import Dict,Any

BUGGY='''def fizzbuzz(n):\n    for i in range(1, n):\n        if i%15==0: print("FizzBuzz")\n        elif i%3==0: print("Fizz")\n        elif i%5==0: print("Buzz")\n        else: print(i)\n'''
FIXED='''def fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i%15==0: print("FizzBuzz")\n        elif i%3==0: print("Fizz")\n        elif i%5==0: print("Buzz")\n        else: print(i)\n'''
TEST='''from src.fizz import fizzbuzz\nimport io,sys\ndef test_fizzbuzz():\n    buf=io.StringIO(); sys.stdout=buf; fizzbuzz(16); sys.stdout=sys.__stdout__\n    lines=buf.getvalue().strip().split("\\n")\n    assert len(lines)==15,f"Expected 15 got {len(lines)}"\n    assert lines[14]=="FizzBuzz",f"Got {lines[14]}"\n    print("PASS")\n'''

class GateChain:
    def evaluate(self,tool,args): return "ALLOW","allowed"
class Sandbox:
    def run(self,cmd,cwd):
        r=subprocess.run(cmd,shell=True,cwd=cwd,capture_output=True,text=True,timeout=10)
        return r.stdout,r.stderr,r.returncode
class ObservationLedger:
    def __init__(self,budget=5000): self.entries=[]; self.budget=budget; self.tokens=0
    def append(self,tool,args,result,tokens=0): self.tokens+=tokens; self.entries.append({"tool":tool,"result":result[:200]})

@dataclass
class Span:
    trace_id:str; span_id:str; name:str; attributes:Dict[str,Any]; start_ns:int; end_ns:int=0; status:str="OK"
class SpanBuilder:
    def __init__(self): self.trace_id=uuid.uuid4().hex[:32]; self.spans=[]
    def span(self,name,attrs=None): return SpanCtx(self,name,attrs or {})
class SpanCtx:
    def __init__(self,b,name,attrs):
        self.s=Span(b.trace_id,uuid.uuid4().hex[:32],name,attrs,time.time_ns()); self.b=b
    def __enter__(self): return self.s
    def __exit__(self,*exc):
        self.s.end_ns=time.time_ns()
        if exc[0]: self.s.status="ERROR"
        self.b.spans.append(self.s)

class MetricsRegistry:
    def __init__(self): self.counters={}; self.histograms={}
    def inc(self,name,labels,v=1):
        k=f"{name}{{{','.join(f'{k}={v}' for k,v in labels.items())}}}"
        self.counters[k]=self.counters.get(k,0)+v
    def observe(self,name,labels,v):
        k=f"{name}{{{','.join(f'{k}={v}' for k,v in labels.items())}}}"
        self.histograms.setdefault(k,[]).append(v)
    def exposition(self):
        lines=[]
        for k,c in self.counters.items(): lines.extend([f"# TYPE {k.split('{')[0]} counter",f"{k} {c}",""])
        return "\n".join(lines)

class CodingPolicy:
    def __init__(self,scratch): self.scratch=scratch
    def run(self,gate,sandbox,ledger,spans,metrics):
        with spans.span("survey"):
            files=os.listdir(self.scratch); ledger.append("list",self.scratch,str(files))
            metrics.inc("tools_called_total",{"tool":"read_file"})
        with spans.span("run_tests"):
            t=time.time(); out,err,code=sandbox.run("python -m pytest tests/ -v",self.scratch)
            metrics.observe("tool_latency_ms",{"tool":"run_tests"},(time.time()-t)*1000)
            metrics.inc("tools_called_total",{"tool":"run_tests"}); ledger.append("tests","",out[:200])
        if "PASS" in out and code==0: return True,"通过"
        with spans.span("inspect"):
            content=open(os.path.join(self.scratch,"src/fizz.py")).read()
            metrics.inc("tools_called_total",{"tool":"read_file"})
        if "15" in out or "range(1, n)" in content:
            gs,_=gate.evaluate("write_file",{})
            if gs=="DENY": return False,"门拒绝"
            with spans.span("fix"):
                open(os.path.join(self.scratch,"src/fizz.py"),"w").write(FIXED)
                metrics.inc("tools_called_total",{"tool":"write_file"})
            with spans.span("verify"):
                t=time.time(); out,_,code=sandbox.run("python -m pytest tests/ -v",self.scratch)
                metrics.observe("tool_latency_ms",{"tool":"run_tests"},(time.time()-t)*1000)
                metrics.inc("tools_called_total",{"tool":"run_tests"})
            return code==0,"验证通过" if code==0 else "验证失败"
        return False,"无法检测"

def main():
    gate=GateChain(); sandbox=Sandbox(); ledger=ObservationLedger()
    spans=SpanBuilder(); metrics=MetricsRegistry()
    scratch=tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(scratch,"src"))
        open(os.path.join(scratch,"src/fizz.py"),"w").write(BUGGY)
        open(os.path.join(scratch,"src/__init__.py"),"w").write("")
        open(os.path.join(scratch,"tests/__init__.py"),"w").write("")
        open(os.path.join(scratch,"tests/test_fizz.py"),"w").write(TEST)
        policy=CodingPolicy(scratch)
        ok,msg=policy.run(gate,sandbox,ledger,spans,metrics)
        print(f"结果: {'✓通过' if ok else '✗失败'} ({msg})")
        print(f"Spans: {len(spans.spans)}  观测: {ledger.tokens}")
        print(metrics.exposition())
        return 0 if ok else 1
    finally: shutil.rmtree(scratch,ignore_errors=True)

if __name__=="__main__": import sys; sys.exit(main())
