"""OTel GenAI Span + Prometheus指标。"""
from __future__ import annotations
import json, math, os, sys, tempfile, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

GEN_AI_SYSTEM="gen_ai.system"; GEN_AI_REQUEST_MODEL="gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS="gen_ai.usage.input_tokens"; GEN_AI_USAGE_OUTPUT_TOKENS="gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_MODEL="gen_ai.response.model"; GEN_AI_RESPONSE_ID="gen_ai.response.id"
GEN_AI_TOOL_NAME="gen_ai.tool.name"; GEN_AI_TOOL_CALL_ID="gen_ai.tool.call.id"
GEN_AI_TOOL_RESULT_BYTES="gen_ai.tool.result.bytes"
STATUS_UNSET="UNSET"; STATUS_OK="OK"; STATUS_ERROR="ERROR"

@dataclass
class SpanEvent:
    name:str; timestamp_unix_nano:int; attributes:dict[str,Any]=field(default_factory=dict)

@dataclass
class GenAISpan:
    trace_id:str; span_id:str; name:str; start_unix_nano:int; end_unix_nano:int=0
    parent_span_id:str=""; attributes:dict[str,Any]=field(default_factory=dict)
    events:list[SpanEvent]=field(default_factory=list); status:str=STATUS_UNSET; status_message:str=""; kind:str="INTERNAL"
    @property
    def duration_ms(self): return (self.end_unix_nano-self.start_unix_nano)/1e6 if self.end_unix_nano>0 else 0
    def to_dict(self):
        return {"trace_id":self.trace_id,"span_id":self.span_id,"parent":self.parent_span_id,
            "name":self.name,"kind":self.kind,"start":self.start_unix_nano,"end":self.end_unix_nano,
            "dur_ms":round(self.duration_ms,4),"attrs":dict(self.attributes),
            "status":{"code":self.status,"msg":self.status_message}}

def new_trace_id(): return uuid.uuid4().hex
def new_span_id(): return uuid.uuid4().hex[:16]
def now_ns(): return time.time_ns()

@dataclass
class JSONLExporter:
    path:str; fh:Any=None; closed:bool=False
    def _ensure_open(self):
        if self.fh is None: os.makedirs(os.path.dirname(self.path) or ".",exist_ok=True); self.fh=open(self.path,"a")
    def export(self,span):
        self._ensure_open(); self.fh.write(json.dumps(span.to_dict(),separators=(",",":"))+"\n"); self.fh.flush()
    def close(self):
        if self.fh and not self.closed: self.fh.close(); self.closed=True

class InMemoryExporter:
    def __init__(self): self.spans=[]
    def export(self,span): self.spans.append(span)
    def close(self): pass

@dataclass
class Counter:
    name:str; values:dict[tuple,float]=field(default_factory=dict)
    def inc(self,labels=None,by=1.0):
        k=tuple(sorted((labels or {}).items())); self.values[k]=self.values.get(k,0)+by

@dataclass
class Histogram:
    name:str; buckets:tuple[float,...]=(5,10,25,50,100,250,500,1000,2500,5000,10000)
    samples:dict[tuple,list[float]]=field(default_factory=dict)
    def observe(self,v,labels=None):
        k=tuple(sorted((labels or {}).items())); self.samples.setdefault(k,[]).append(float(v))
    def bucket_counts(self,labels=None):
        k=tuple(sorted((labels or {}).items())); vs=self.samples.get(k,[])
        c={b:sum(1 for v in vs if v<=b) for b in self.buckets}; c[math.inf]=len(vs); return c
    def total_count(self,labels=None): return len(self.samples.get(tuple(sorted((labels or {}).items())),[]))
    def total_sum(self,labels=None): return sum(self.samples.get(tuple(sorted((labels or {}).items())),[]))

@dataclass
class MetricsRegistry:
    counters:dict[str,Counter]=field(default_factory=dict); histograms:dict[str,Histogram]=field(default_factory=dict)
    def counter(self,name): return self.counters.setdefault(name,Counter(name))
    def histogram(self,name): return self.histograms.setdefault(name,Histogram(name))

def prometheus_exposition(reg):
    lines=[]
    for name in sorted(reg.counters):
        c=reg.counters[name]; lines.append(f"# TYPE {c.name} counter")
        for k,v in sorted(c.items()):
            lbl="{"+",".join(f'{k_}="{v_}"' for k_,v_ in sorted(k))+"}" if k else ""
            lines.append(f"{c.name}{lbl} {v}")
    for name in sorted(reg.histograms):
        h=reg.histograms[name]; lines.append(f"# TYPE {h.name} histogram")
        for k in sorted(h.samples.keys()):
            counts=h.bucket_counts(dict(k)); d=dict(k)
            le_vals=[(f'le="{b}"',b) for b in h.buckets]+[('le="+Inf"',math.inf)]
            for le_s,b in le_vals:
                extra=",".join([le_s]+[f'{k_}="{v_}"' for k_,v_ in sorted(d)])
                lines.append(f'{h.name}_bucket{{{extra}}} {counts[b]}')
            lbl="{"+",".join(f'{k_}="{v_}"' for k_,v_ in sorted(d))+"}" if d else ""
            lines.append(f"{h.name}_sum{lbl} {h.total_sum(dict(k))}")
            lines.append(f"{h.name}_count{lbl} {h.total_count(dict(k))}")
    return "\n".join(lines)+"\n"

@dataclass
class SpanBuilder:
    trace_id:str=field(default_factory=new_trace_id); exporters:list=field(default_factory=list); metrics:MetricsRegistry|None=None
    @contextmanager
    def span(self,name,attributes=None,parent=None,kind="INTERNAL"):
        s=GenAISpan(self.trace_id,new_span_id(),name,now_ns(),parent_span_id=parent.span_id if parent else "",kind=kind,attributes=dict(attributes or {}))
        try: yield s; s.status=STATUS_OK
        except BaseException as e:
            s.status=STATUS_ERROR; s.status_message=str(e); s.events.append(SpanEvent("exception",now_ns(),{"type":type(e).__name__,"msg":str(e)})); raise
        finally:
            s.end_unix_nano=now_ns()
            for exp in self.exporters: exp.export(s)
            if self.metrics:
                tool=s.attributes.get(GEN_AI_TOOL_NAME)
                if tool:
                    self.metrics.counter("tools_called_total").inc({"tool":str(tool)})
                    self.metrics.histogram("tool_latency_ms").observe(s.duration_ms,{"tool":str(tool)})

def _demo():
    tmp=tempfile.mkdtemp(); tp=os.path.join(tmp,"traces.jsonl")
    jsonl=JSONLExporter(tp); mem=InMemoryExporter(); m=MetricsRegistry()
    b=SpanBuilder(exporters=[jsonl,mem],metrics=m)
    with b.span("gen_ai.chat",{GEN_AI_SYSTEM:"anthropic",GEN_AI_REQUEST_MODEL:"claude"},kind="CLIENT") as chat:
        chat.attributes[GEN_AI_USAGE_INPUT_TOKENS]=412; chat.attributes[GEN_AI_USAGE_OUTPUT_TOKENS]=96
        with b.span("gen_ai.tool.execution",parent=chat,attributes={GEN_AI_TOOL_NAME:"read_file"}):
            time.sleep(0.005)
    print(f"emitted {len(mem.spans)} span(s)")
    for s in mem.spans: print(f"  {s.name:30s} dur={s.duration_ms:.2f}ms status={s.status}")
    print("\n--- prometheus ---\n"+prometheus_exposition(m))
    jsonl.close()
    with open(tp) as f: print(f"roundtrip: {len([json.loads(l) for l in f if l.strip()])} spans")

if __name__=="__main__": sys.exit(_demo())
