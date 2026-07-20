"""代码执行指标——子进程隔离+pass@k。"""
import re,json,subprocess,sys,math
from dataclasses import dataclass
from typing import List

def extract_code(text):
    m=re.search(r"```(?:python)?\n(.*?)```",text,re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def run_candidate(code,assertions,timeout=3.0):
    script="import json\n"
    script+="results=[]\n"+code+"\n"
    for a in assertions:
        script+=f"try:\n    assert {a}\n    results.append(True)\nexcept:\n    results.append(False)\n"
    script+="print(json.dumps(results))"
    try:
        r=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=timeout)
        if r.returncode!=0: return {"passed":0,"total":len(assertions),"score":0,"exit":"error"}
        res=json.loads(r.stdout.strip())
        p=sum(1 for x in res if x)
        return {"passed":p,"total":len(assertions),"score":p/max(len(assertions),1),
                "exit":"pass" if p==len(assertions) else "assertion_fail"}
    except subprocess.TimeoutExpired:
        return {"passed":0,"total":len(assertions),"score":0,"exit":"timeout"}
    except Exception as e:
        return {"passed":0,"total":len(assertions),"score":0,"exit":"error"}

def pass_at_k(n,c,k):
    if n-c<k: return 1.0
    return 1.0-math.comb(n-c,k)/math.comb(n,k)

def main():
    for code,asserts in [("def f(x): return x*2",["f(2)==4","f(3)==6"]),
                         ("def f(x): return x+1",["f(2)==4"]),
                         ("def f(x): while True: pass",["f(2)==4"])]:
        r=run_candidate(code,asserts)
        print(f"  {code[:35]}... → {r['exit']} 分数={r['score']:.3f} ({r['passed']}/{r['total']})")
    print(f"\npass@k: n=10,c=5,k=1={pass_at_k(10,5,1):.3f} k=5={pass_at_k(10,5,5):.3f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
