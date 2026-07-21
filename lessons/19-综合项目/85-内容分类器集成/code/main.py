"""内容分类器集成。"""
import re
from dataclasses import dataclass,field
from typing import List

@dataclass
class CV:
    name:str; score:float; severity:str; findings:List[str]=field(default_factory=list)
@dataclass
class Action:
    verb:str; output:str; severity:str

class Tox:
    def __init__(self): self.slurs=["hate","stupid","idiot"]
    def classify(self,t):
        f=[f"slur:{s}" for s in self.slurs if re.search(r'\b'+s+r'\b',t,re.I)]
        sev="high" if len(f)>=2 else "medium" if f else "none"
        return CV("toxicity",min(0.3*len(f),1),sev,f)
    def redact(self,t):
        for s in self.slurs: t=re.sub(r'\b'+s+r'\b',"[redacted]",t,flags=re.I)
        return t

class PII:
    E=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
    def classify(self,t):
        f=[]; sev="none"
        if self.E.search(t): f.append("email"); sev="medium"
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b',t): f.append("ssn"); sev="high"
        return CV("pii",0.5 if f else 0,sev,f)
    def redact(self,t): return self.E.sub("[redacted-email]",t)

class Router:
    def __init__(self,cs): self.cs=cs
    def decide(self,t):
        vs=[c.classify(t) for c in self.cs]; sev_o={"none":0,"low":1,"medium":2,"high":3}
        ms=max(vs,key=lambda v:sev_o.get(v.severity,0))
        if ms.severity=="high": return Action("block","I cannot provide.",ms.severity)
        if ms.severity=="medium":
            o=t
            for c in self.cs: o=c.redact(o)
            return Action("redact",o,ms.severity)
        return Action("log",t,ms.severity)

def main():
    r=Router([Tox(),PII()])
    for t in["Contact support@example.com","I hate this stupid thing","How are you today?"]:
        a=r.decide(t); print(f"  [{a.verb:>6}] {t[:40]}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
