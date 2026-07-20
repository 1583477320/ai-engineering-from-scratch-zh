"""智能体循环契约——确定性状态机、钩子、拉取点。"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

class State(str, Enum):
    IDLE="idle"; PLANNING="planning"; EXECUTING="executing"
    AWAITING_TOOL="awaiting_tool"; REFLECTING="reflecting"; DONE="done"

HOOK_TOPICS=("before_plan","after_plan","before_step","after_step",
             "before_tool_call","after_tool_call","on_error","on_pause",
             "on_budget_exceeded","on_complete")
EVENT_TYPES=("session.start","plan.draft","plan.commit","step.start","step.end",
             "tool.call","tool.result","tool.error","budget.warn","session.pause","session.complete")

class HookAbort(Exception): pass

@dataclass
class Event:
    type:str; payload:dict; ts:float
    def to_dict(self): return {"type":self.type,"payload":self.payload,"ts":self.ts}

@dataclass
class Budget:
    max_turns:int=8; max_tool_calls:int=16; max_wall_seconds:float=30.0
    turns:int=0; tool_calls:int=0; started_at:float=field(default_factory=time.time)
    def remaining_seconds(self): return max(0,self.max_wall_seconds-(time.time()-self.started_at))
    def exceeded(self):
        if self.turns>=self.max_turns: return "turns"
        if self.tool_calls>=self.max_tool_calls: return "tool_calls"
        if self.remaining_seconds()<=0: return "wall_clock"
        return None

@dataclass
class Step:
    id:int; description:str; requires_tool:bool
    tool_name:str|None=None; tool_args:dict=field(default_factory=dict)
    result:Any=None; error:str|None=None

@dataclass
class PullRequest:
    reason:str; state:State; payload:dict

@dataclass
class SessionResult:
    state:State; reason:str; steps:list[Step]; events:list[Event]

class HookRegistry:
    def __init__(self): self._subs={t:[] for t in HOOK_TOPICS}
    def on(self,topic,fn): self._subs[topic].append(fn)
    def fire(self,topic,payload): return [fn(payload) for fn in self._subs[topic]]

Planner=Callable[[str,list[Step]],list[Step]]

def _default_planner(goal,history):
    if history: return []
    return [Step(1,f"interpret: {goal}",False),
            Step(2,"fetch record",True,"db.get_user",{"id":42}),
            Step(3,"summarize",True,"format.summary",{"style":"short"})]

class HarnessLoop:
    def __init__(self,planner=None,budget=None):
        self.state=State.IDLE; self.hooks=HookRegistry(); self.budget=budget or Budget()
        self._planner=planner or _default_planner; self._goal=""
        self._plan=[]; self._cursor=0; self._events=[]; self._history=[]
        self._reason=""; self._prev_state=None

    @property
    def events(self): return list(self._events)

    def _emit(self,etype,payload):
        self._events.append(Event(type=etype,payload=payload,ts=time.time()))

    def _transition(self,target):
        legal={State.IDLE:{State.PLANNING},State.PLANNING:{State.EXECUTING,State.IDLE,State.DONE},
               State.EXECUTING:{State.AWAITING_TOOL,State.REFLECTING,State.IDLE},
               State.AWAITING_TOOL:{State.REFLECTING,State.IDLE},
               State.REFLECTING:{State.PLANNING,State.EXECUTING,State.DONE,State.IDLE},State.DONE:set()}
        if target not in legal[self.state]: raise RuntimeError(f"illegal {self.state}->{target}")
        self.state=target

    def _check_budget(self):
        w=self.budget.exceeded()
        if w is None: return None
        self._emit("budget.warn",{"limit":w}); self.hooks.fire("on_budget_exceeded",{"limit":w})
        self._reason=f"budget_exceeded:{w}"; self._prev_state=self.state; return self._pause(self._reason)

    def _pause(self,reason):
        self._emit("session.pause",{"reason":reason}); self.hooks.fire("on_pause",{"reason":reason})
        self._transition(State.IDLE); return PullRequest(reason=reason,state=self.state,payload={"reason":reason})

    def run(self,goal):
        if self.state!=State.IDLE: raise RuntimeError(f"run requires IDLE, got {self.state}")
        self._goal=goal; self.budget.started_at=time.time()
        self._emit("session.start",{"goal":goal}); return self._step()

    def resume(self,payload=None):
        if self.state==State.IDLE and self._reason.startswith("budget_exceeded"):
            self.budget.turns=0; self.budget.tool_calls=0; self.budget.started_at=time.time()
            self._reason=""; prev=self._prev_state; self._prev_state=None
            if not self._plan: return self._begin_plan()
            self.state=State.EXECUTING if prev==State.EXECUTING else State.REFLECTING; return self._step()
        if self.state==State.AWAITING_TOOL:
            if payload is None: raise ValueError("resume needs payload")
            cur=self._plan[self._cursor]
            if "error" in payload: cur.error=str(payload["error"]); self._emit("tool.error",{"step":cur.id,"error":cur.error})
            else: cur.result=payload.get("result"); self._emit("tool.result",{"step":cur.id,"result":cur.result})
            self.hooks.fire("after_tool_call",{"step":cur}); self._transition(State.REFLECTING); return self._step()
        raise RuntimeError(f"resume unsupported from {self.state}")

    def _begin_plan(self):
        self._transition(State.PLANNING); self.hooks.fire("before_plan",{"goal":self._goal})
        draft=self._planner(self._goal,list(self._history))
        self._emit("plan.draft",{"steps":[s.description for s in draft]})
        self.hooks.fire("after_plan",{"steps":draft}); self._plan=draft; self._cursor=0
        self._emit("plan.commit",{"count":len(draft)})
        if not draft: return self._complete("no_plan")
        self._transition(State.EXECUTING); return self._step()

    def _step(self):
        if self.state==State.IDLE: return self._begin_plan()
        b=self._check_budget()
        if b is not None: return b
        if self.state==State.REFLECTING:
            self._cursor+=1; self.budget.turns+=1
            if self._cursor>=len(self._plan): return self._complete("goal_met")
            self._transition(State.EXECUTING); return self._step()
        step=self._plan[self._cursor]; self.hooks.fire("before_step",{"step":step})
        self._emit("step.start",{"step_id":step.id,"desc":step.description})
        if step.requires_tool:
            try: self.hooks.fire("before_tool_call",{"step":step})
            except HookAbort as e: step.error=f"abort:{e}"; self._emit("tool.error",{"step":step.id,"error":step.error}); self._transition(State.REFLECTING); return self._step()
            self.budget.tool_calls+=1; self._emit("tool.call",{"step":step.id,"tool":step.tool_name})
            self._transition(State.AWAITING_TOOL); self._emit("step.end",{"step_id":step.id,"outcome":"awaiting"})
            return PullRequest("tool_call",self.state,{"tool":step.tool_name,"args":step.tool_args})
        step.result=f"ok:{step.description}"; self._emit("step.end",{"step_id":step.id,"outcome":"ok"})
        self.hooks.fire("after_step",{"step":step,"outcome":"ok"}); self._transition(State.REFLECTING); return self._step()

    def _complete(self,reason):
        self._emit("session.complete",{"reason":reason}); self.hooks.fire("on_complete",{"reason":reason})
        self._transition(State.DONE); self._reason=reason
        return SessionResult(state=self.state,reason=reason,steps=list(self._plan),events=list(self._events))

def _demo():
    loop=HarnessLoop(); fired=[]
    for t in HOOK_TOPICS: loop.hooks.on(t lambda p,t=t: fired.append(t))
    out=loop.run("ship notes"); assert isinstance(out,PullRequest) and out.reason=="tool_call"
    out=loop.resume({"result":{"id":42}}); assert isinstance(out,PullRequest) and out.reason=="tool_call"
    final=loop.resume({"result":"done"}); assert isinstance(final,SessionResult) and final.state==State.DONE
    print(json.dumps({"events":[e.type for e in final.events],"hooks":fired,"state":final.state.value,"reason":final.reason},indent=2))

if __name__=="__main__": _demo()
