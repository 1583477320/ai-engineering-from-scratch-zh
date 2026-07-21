"""宪法规则引擎。"""
import re,json
from dataclasses import dataclass,field
from typing import List,Dict,Any

@dataclass
class V:
    rule:str; severity:str; explanation:str

CONSTITUTION=[
    {"name":"no-empty-refusal","severity":"medium","must":{"not_":{"contains_regex":r"^I cannot\.?\s*$"}},"explanation":"拒绝必须包含建议","fix":{"append_if_missing":" 我可以帮您做什么？"}},
    {"name":"end-with-runnable","severity":"medium","applies_when":{"contains_regex":r"```"},"must":{"any_of":[{"ends_with_regex":r"```\s*$"},{"contains_regex":"assumption:"}]},"explanation":"代码必须以围栏或假设结束","fix":{"append_if_missing":"\n\n假设：输入有效。"}},
    {"name":"no-pii","severity":"high","must":{"not_":{"contains_regex":r"[\w.+-]+@[\w-]+\.[\w.]+"}},"explanation":"不能包含邮箱","fix":{"replace_regex":[r"[\w.+-]+@[\w-]+\.[\w.]+","[redacted-email]"]}},
]

def check(pred,t):
    if isinstance(pred,dict):
        for k,v in pred.items():
            if k=="all_of": return all(check(x,t) for x in v)
            if k=="any_of": return any(check(x,t) for x in v)
            if k=="not_": return not check(v,t)
            if k=="contains_regex": return bool(re.search(v,t,re.I))
            if k=="ends_with_regex": return bool(re.search(v+r"\s*$",t,re.I))
            if k=="max_words": return len(t.split())<=v
    return False

class Engine:
    def __init__(self,rules=None): self.rules=rules or CONSTITUTION
    def evaluate(self,t):
        vs=[]
        for r in self.rules:
            if "applies_when" in r and not check(r["applies_when"],t): continue
            if not check(r["must"],t): vs.append(V(r["name"],r["severity"],r.get("explanation","")))
        return vs
    def fix(self,t,vs):
        for v in vs:
            r=next(x for x in self.rules if x["name"]==v.rule)
            f=r.get("fix",{})
            if "append_if_missing" in f and f["append_if_missing"] not in t: t+=f["append_if_missing"]
            if "replace_regex" in f: t=re.sub(f["replace_regex"][0],f["replace_regex"][1],t,flags=re.I)
        return t

def main():
    e=Engine()
    t="I cannot.\n```python\nx=1\n```"
    vs=e.evaluate(t); print(f"违规: {len(vs)}")
    for v in vs: print(f"  [{v.severity}] {v.rule}: {v.explanation}")
    f=e.fix(t,vs)
    if f!=t: print(f"修正后: {f[:100]}...")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
