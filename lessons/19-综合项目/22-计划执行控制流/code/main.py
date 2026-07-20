"""计划执行智能体——失败重新规划、计划差异、双预算。"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class Step:
    id:int; tool_name:str; args:dict; expected_outcome:str; result:Any=None; error:str|None=None
    def signature(self): return (self.tool_name,json.dumps(self.args,sort_keys=True))

@dataclass
class PlanDiff:
    revision:int; removed:list[int]; added:list[int]; revised:list[int]
    def to_dict(self): return {"revision":self.revision,"removed":self.removed,"added":self.added,"revised":self.revised}

@dataclass
class Event:
    type:str; payload:dict; ts:float=field(default_factory=time.time)

@dataclass
class SessionResult:
    status:str; reason:str; history:list[Step]; revisions:list[PlanDiff]; events:list[Event]

Planner=Callable[[str,list[Step],str|None],list[Step]]
ToolExecutor=Callable[[str,dict],Any]

class ToolFailure(Exception): pass

def _diff_plans(old,new,rev):
    oi={s.id for s in old}; ni={s.id for s in new}
    removed=sorted(oi-ni); added=sorted(ni-oi); revised=[]
    ob={s.id:s for s in old}
    for s in new:
        if s.id in oi and ob[s.id].signature()!=s.signature(): revised.append(s.id)
    return PlanDiff(revision=rev,removed=removed,added=added,revised=revised)

class PlanExecuteAgent:
    def __init__(self,planner,executor,*,max_steps=12,max_replans=5):
        self._planner=planner; self._executor=executor; self.max_steps=max_steps; self.max_replans=max_replans; self._events=[]
    def _emit(self,t,p): self._events.append(Event(type=t,payload=p))
    def run(self,goal):
        self._events=[]; history=[]; revisions=[]; st=0; rp=0; le=None
        plan=self._planner(goal,history,None)
        self._emit("plan.commit",{"rev":0,"steps":[s.expected_outcome for s in plan]})
        if not plan: return SessionResult("failed","no_plan",history,revisions,list(self._events))
        cursor=0; rev=0
        while cursor<len(plan):
            if st>=self.max_steps: return SessionResult("failed","step_budget",history,revisions,list(self._events))
            step=plan[cursor]; self._emit("step.start",{"id":step.id,"tool":step.tool_name})
            try:
                step.result=self._executor(step.tool_name,step.args)
                self._emit("step.end",{"id":step.id,"outcome":"ok"})
                history.append(step); cursor+=1; st+=1; continue
            except Exception as exc:
                step.error=f"{type(exc).__name__}: {exc}"
                self._emit("step.end",{"id":step.id,"outcome":"error","error":step.error})
                history.append(step); st+=1; le=step.error
            if rp>=self.max_replans: return SessionResult("failed","replan_budget",history,revisions,list(self._events))
            rp+=1; rev+=1; new_plan=self._planner(goal,history,le)
            self._emit("plan.draft",{"rev":rev})
            if not new_plan: return SessionResult("failed","no_plan",history,revisions,list(self._events))
            diff=_diff_plans(plan[cursor:],new_plan,rev); revisions.append(diff); self._emit("plan.diff",diff.to_dict())
            plan=new_plan; cursor=0; self._emit("plan.commit",{"rev":rev,"steps":[s.expected_outcome for s in plan]})
        return SessionResult("completed","goal_met",history,revisions,list(self._events))

def make_planner(fail_id=None,recovery="route_around"):
    def planner(goal,hist,le):
        if le is None:
            init=[Step(1,"fetch",{"k":"in"},"loaded"),Step(2,"transform",{"mode":"v1"},"v1"),
                  Step(3,"render",{},"rendered"),Step(4,"submit",{},"submitted")]
            if fail_id:
                for s in init:
                    if s.id==fail_id: s.args={**s.args,"_fail":True}
            return init
        if recovery=="route_around" and "transform" in le:
            return [Step(2,"transform",{"mode":"v2"},"fallback"),Step(3,"render",{},"rendered"),Step(4,"submit",{},"submitted")]
        return [Step(98,"log",{"why":le or ""},"logged"),Step(99,"notify",{},"notified")]
    return planner

def _demo():
    def exe(tool,args):
        if args.get("_fail"): raise ToolFailure(f"{tool} forced")
        if tool=="fetch": return {"k":"v"}
        if tool=="transform":
            if args.get("mode")=="v1": raise ToolFailure("v1 down")
            return {"ok":True}
        if tool=="render": return "html"
        if tool=="submit": return {"id":1}
        if tool in ("log","notify"): return "ok"
        raise ToolFailure(f"unknown {tool}")
    agent=PlanExecuteAgent(planner=make_planner(fail_id=2,recovery="route_around"),executor=exe,max_steps=12,max_replans=5)
    res=agent.run("ship report")
    print(json.dumps({"status":res.status,"reason":res.reason,"history":[(s.id,s.tool_name,bool(s.error)) for s in res.history],
        "revisions":[r.to_dict() for r in res.revisions]},indent=2))

if __name__=="__main__": _demo()
