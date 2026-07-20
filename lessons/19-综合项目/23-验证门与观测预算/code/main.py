"""验证门与观测预算。"""
from __future__ import annotations
import json, re, sys
from dataclasses import dataclass, field
from typing import Callable, Protocol

@dataclass(frozen=True)
class ToolCall:
    turn:int; tool:str; argv:tuple[str,...]; payload:str=""

@dataclass(frozen=True)
class Observation:
    turn:int; tool:str; text:str; tokens:int

@dataclass(frozen=True)
class GateDecision:
    allow:bool; gate:str; reason:str

def estimate_tokens(text:str)->int:
    return max(1,len(text)//4) if text else 0

@dataclass
class ObservationLedger:
    rows:list[Observation]=field(default_factory=list)
    def record(self,obs): self.rows.append(obs)
    def cumulative(self): return sum(r.tokens for r in self.rows)
    def per_tool(self,name): return sum(r.tokens for r in self.rows if r.tool==name)
    def latest_turn(self): return self.rows[-1].turn if self.rows else -1

@dataclass
class GateContext:
    ledger:ObservationLedger; current_turn:int; history:tuple[ToolCall,...]=()

class VerificationGate(Protocol):
    name:str
    def evaluate(self,call:ToolCall,ctx:GateContext)->GateDecision: ...

@dataclass
class WhitelistGate:
    allowed:frozenset[str]; name:str="whitelist"
    def evaluate(self,call,ctx):
        if call.tool in self.allowed: return GateDecision(True,self.name,"ok")
        return GateDecision(False,self.name,f"{call.tool!r} not in {sorted(self.allowed)}")

@dataclass
class RegexGate:
    refuse_patterns:tuple[re.Pattern,...]; name:str="regex"
    @classmethod
    def from_strings(cls,pats,name="regex"): return cls(tuple(re.compile(p) for p in pats),name)
    def evaluate(self,call,ctx):
        hay=" ".join(call.argv)+" "+call.payload
        for p in self.refuse_patterns:
            if p.search(hay): return GateDecision(False,self.name,f"matched {p.pattern!r}")
        return GateDecision(True,self.name,"no match")

@dataclass
class RecencyGate:
    window:int; name:str="recency"
    def evaluate(self,call,ctx):
        last=ctx.ledger.latest_turn()
        if last<0: return GateDecision(True,self.name,"no prior")
        gap=call.turn-last
        if gap>self.window: return GateDecision(False,self.name,f"gap {gap}>{self.window}")
        return GateDecision(True,self.name,f"gap {gap} ok")

@dataclass
class BudgetGate:
    max_tokens:int; name:str="budget"
    def evaluate(self,call,ctx):
        used=ctx.ledger.cumulative()
        if used>=self.max_tokens: return GateDecision(False,self.name,f"{used}/{self.max_tokens} exhausted")
        return GateDecision(True,self.name,f"{self.max_tokens-used} left")

@dataclass
class ChainOutcome:
    decisions:list[GateDecision]
    @property
    def allow(self): return all(d.allow for d in self.decisions)
    @property
    def deny_reason(self):
        for d in self.decisions:
            if not d.allow: return f"[{d.gate}] {d.reason}"
        return None

@dataclass
class GateChain:
    gates:tuple[VerificationGate,...]
    def evaluate(self,call,ctx):
        decisions=[]
        for g in self.gates:
            d=g.evaluate(call,ctx); decisions.append(d)
            if not d.allow: return ChainOutcome(decisions=decisions)
        return ChainOutcome(decisions=decisions)

ToolFn=Callable[[ToolCall],str]

def run_loop(calls,chain,tool_fns):
    ledger=ObservationLedger(); decisions=[]; obs=[]; allowed=0; refused=0; history=[]
    for call in calls:
        ctx=GateContext(ledger=ledger,current_turn=call.turn,history=tuple(history))
        outcome=chain.evaluate(call,ctx); decisions.append(outcome); history.append(call)
        if not outcome.allow: refused+=1; continue
        fn=tool_fns.get(call.tool)
        if fn is None: refused+=1; continue
        result=fn(call)
        o=Observation(call.turn,call.tool,result,estimate_tokens(result))
        ledger.record(o); obs.append(o); allowed+=1
    return {"turns":len(calls),"allowed":allowed,"refused":refused,
            "tokens":sum(o.tokens for o in obs),
            "decisions":[{"tool":c.tool,"allow":d.allow,"reason":d.deny_reason or "ok"} for c,d in zip(calls,decisions)]}

def _demo():
    tools={"read_file":lambda c:"# fake\n"+("line "*12)*12,"list_dir":lambda c:"a.py\nb.py\n","run_tests":lambda c:'{"ok":true}'}
    chain=GateChain(gates=(WhitelistGate(frozenset({"read_file","list_dir","run_tests"})),
        RegexGate.from_strings([r"\brm\s+-rf\b",r"\bsudo\b"]),RecencyGate(window=3),BudgetGate(max_tokens=200)))
    calls=[ToolCall(1,"list_dir",("./",)),ToolCall(2,"read_file",("main.py",)),ToolCall(3,"read_file",("README.md",)),
           ToolCall(4,"run_tests",("./",)),ToolCall(5,"shell",("rm","-rf","/"))]
    r=run_loop(calls,chain,tools)
    print(f"turns={r['turns']} allowed={r['allowed']} refused={r['refused']}")
    for d in r["decisions"]: print(f"  {d['tool']:12s} {'ALLOW' if d['allow'] else 'DENY':6s} {d['reason']}")
    print(f"\ntokens: {r['tokens']}")
    return 0 if r["refused"]>=1 else 1

if __name__=="__main__": sys.exit(_demo())
