"""端到端评估运行器——组合全部评估组件。"""
import json,random,math,re,time
from dataclasses import dataclass,field
from typing import List,Dict,Protocol

@dataclass
class TaskSpec:
    task_id:str; category:str; prompt:str; targets:List[str]; metric_name:str; post_process:str

class ModelAdapter(Protocol):
    model_id: str
    def generate(self, prompt: str) -> dict: ...

class RuleBasedAdapter:
    model_id="rule-based"
    def generate(self,p):
        if "2+2" in p: return {"text":"4","confidence":0.99}
        if "7*6" in p: return {"text":"42","confidence":0.99}
        if "3+3" in p: return {"text":"B","confidence":0.95}
        if "cat" in p.lower(): return {"text":"cat sat","confidence":0.9}
        return {"text":"unknown","confidence":0.3}

class RandomAdapter:
    model_id="random"
    def generate(self,p):
        return {"text":random.choice(["4","7","B","A","cat","unknown"]),"confidence":0.5}

def tokenize(t): return re.findall(r"\w+",t.lower())
def exact_match(p,ts): return float(any(p.strip()==t.strip() for t in ts))
def rouge_l(p,t):
    a,b=tokenize(p),tokenize(t)
    if not a or not b: return 0
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1]=dp[i][j]+1 if a[i]==b[j] else max(dp[i+1][j],dp[i][j+1])
    lcs=dp[n][m]; pr,rc=lcs/len(a),lcs/len(b)
    return 2*pr*rc/(pr+rc) if pr+rc else 0

def score_one(task,pred):
    p=pred.strip()
    if task.post_process=="lower": p=p.lower()
    if task.post_process=="strip_whitespace": p=p.strip()
    if task.post_process=="extract_letter":
        for c in p.upper():
            if c in "ABCDE": p=c; break
    if task.metric_name=="exact_match": return exact_match(p,task.targets)
    if task.metric_name=="rouge_l": return max(rouge_l(p,t) for t in task.targets)
    return 0

def build_fixtures():
    return [TaskSpec("arith_001","arithmetic","Compute: 2+2",["4"],"exact_match","strip_whitespace"),
            TaskSpec("arith_002","arithmetic","Compute: 7*6",["42"],"exact_match","strip_whitespace"),
            TaskSpec("mcq_001","mcq","2+2=? A:3 B:4 C:5",["B"],"exact_match","extract_letter"),
            TaskSpec("mcq_002","mcq","3+3=? A:5 B:6 C:7",["B"],"exact_match","extract_letter"),
            TaskSpec("sum_001","summary","Summarize: The cat sat.",["cat sat"],"rouge_l","strip_whitespace")]

def run_eval(adapters,tasks):
    runs=[]
    for a in adapters:
        for t in tasks:
            gen=a.generate(t.prompt); s=score_one(t,gen["text"])
            runs.append({"model":a.model_id,"task":t.task_id,"score":s,"cat":t.category})
    return runs

def aggregate(runs,b=100):
    by_m=defaultdict(list)
    for r in runs: by_m[r["model"]].append(r["score"])
    rows=[]
    for mid,scores in by_m.items():
        means=sorted([sum(random.choices(scores,k=len(scores)))/len(scores) for _ in range(b)])
        mean=sum(scores)/len(scores); lo=means[5]; hi=means[94]
        rows.append({"model_id":mid,"mean":round(mean,3),"ci":f"{lo:.3f}-{hi:.3f}","n":len(scores)})
    rows.sort(key=lambda r:-r["mean"]); return rows

def render_md(rows):
    lines=["| Rank | Model | Mean | 95% CI | Tasks |","|------|-------|------|--------|-------|"]
    for i,r in enumerate(rows,1):
        lines.append(f"| {i} | {r['model_id']} | {r['mean']} | {r['ci']} | {r['n']} |")
    return "\n".join(lines)

def main():
    random.seed(42); tasks=build_fixtures()
    adapters=[RuleBasedAdapter(),RandomAdapter()]
    runs=run_eval(adapters,tasks)
    rows=aggregate(runs)
    print("排行榜:"); print(render_md(rows))
    rule_idx=next(i for i,r in enumerate(rows) if r["model_id"]=="rule-based")
    rand_idx=next(i for i,r in enumerate(rows) if r["model_id"]=="random")
    ok=rule_idx<rand_idx
    print(f"\n规则适配器: {rule_idx+1}  随机适配器: {rand_idx+1}  {'✓达标' if ok else '✗未达标'}")
    return 0 if ok else 1

if __name__=="__main__":
    import sys; sys.exit(main())
