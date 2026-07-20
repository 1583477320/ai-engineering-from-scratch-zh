"""OTel 可观测性——span+JSONL导出+Prometheus指标。"""
import json,time,uuid
from dataclasses import dataclass,field
from typing import Dict,List,Any

@dataclass
class GenAISpan:
    trace_id:str; span_id:str; name:str; attributes:Dict[str,Any]=field(default_factory=dict)
    start_ns:int=0; end_ns:int=0; status:str="OK"; status_message:str=""

class SpanBuilder:
    def __init__(self): self.trace_id=uuid.uuid4().hex[:32]; self.spans=[]
    def span(self,name,attrs=None): return SpanCtx(self,name,attrs or {})
class SpanCtx:
    def __init__(self,b,name,attrs):
        self.s=GenAISpan(b.trace_id,uuid.uuid4().hex[:32],name,dict(attrs),time.time_ns()); self.b=b
    def __enter__(self): return self.s
    def __exit__(self,*exc):
        self.s.end_ns=time.time_ns()
        if exc[0]: self.s.status="ERROR"; self.s.status_message=str(exc[1])
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
        for k,c in self.counters.items():
            m=k.split("{")[0]; lines.extend([f"# TYPE {m} counter",f"{k} {c}",""])
        buckets=[5,10,25,50,100,250,500,1000]
        for k,vals in self.histograms.items():
            m=k.split("{")[0]; l=k.split("{")[1]
            lines.extend([f"# TYPE {m} histogram",""])
            for b in buckets:
                lines.append(f'{m}_bucket{{le="{b}",{l}}} {sum(1 for v in vals if v<=b)}')
            lines.append(f'{m}_bucket{{le="+Inf",{l}}} {len(vals)}')
            lines.append(f'{m}_sum{{ {l}}} {sum(vals):.2f}')
            lines.append(f'{m}_count{{ {l}}} {len(vals)}')
        return "\n".join(lines)

def main():
    b=SpanBuilder(); m=MetricsRegistry()
    with b.span("gen_ai.chat",{"gen_ai.system":"anthropic","gen_ai.request.model":"claude-3"}):
        for tool in ["read_file","write_file","run_tests"]:
            t0=time.time_ns()
            with b.span(f"gen_ai.tool.{tool}",{"gen_ai.tool.name":tool}):
                time.sleep(0.001)
            m.inc("tools_called_total",{"tool":tool})
            m.observe("tool_latency_ms",{"tool":tool},(time.time_ns()-t0)/1e6)
    print(f"Span数: {len(b.spans)}")
    for s in b.spans[:3]: print(f"  {s.name}: {s.status}")
    print(f"\nPrometheus:\n{m.exposition()[:300]}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
