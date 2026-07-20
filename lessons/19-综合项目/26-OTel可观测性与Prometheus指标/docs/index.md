# 综合项目26——OTel GenAI Span与Prometheus指标可观测性

> 没有可观测性的智能体主循环是花费金钱的黑盒。本课程手写一个span构建器，发出符合OpenTelemetry GenAI语义约定的记录，写入JSONL文件每行一个span，并以Prometheus文本格式暴露计数器和直方图。全部标准库Python，离线运行。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** 第19章 · 第23-25节（门/沙箱/工具）
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 构建符合OpenTelemetry GenAI语义约定的span数据类
- 实现每行一个自包含span的JSONL导出器
- 构建带标签的计数器和直方图，以Prometheus文本格式导出
- 将任何可调用包装在span上下管理器中
- 验证发出的span可往返json.loads并匹配规范形状

---

## 1. 问题

生产中的编码智能体每轮产生三类工件：模型调用、工具执行、验证门决策。没有结构化遥测，这些都无用。

三类失败模式：**缺失追踪**（周二出错但唯一记录是500行聊天日志）、**不可解析追踪**（智能体用了自定义字段名，OTel后端无法读取）、**未聚合指标**（可以看到一个慢工具调用，但无法回答"过去一小时read_file调用的p95延迟是多少"）。

---

## 2. 核心概念

### 2.1 GenAI语义约定

标准属性键：
- `gen_ai.system`（提供商，如`anthropic`）
- `gen_ai.request.model`（模型ID）
- `gen_ai.usage.input_tokens` / `output_tokens`
- `gen_ai.tool.name` / `gen_ai.tool.call.id`

### 2.2 Span结构

```
GenAISpan
  trace_id: str     (16字节hex，W3C追踪上下文)
  span_id: str      (16字节hex)
  parent_span_id: str
  name: str
  attributes: dict
  start_unix_nano: int
  end_unix_nano: int
  status: str       (UNSET/OK/ERROR)
  events: list[SpanEvent]
```

### 2.3 JSONL导出器

每行一个JSON对象。最简单的可流式、grep和导入的格式。真实OTel导出器说OTLP gRPC；JSONL是离线等价物。

### 2.4 Prometheus指标

- **计数器**：`tools_called_total{tool="read_file"}`，每次工具调用递增
- **直方图**：`tool_latency_ms{tool="read_file"}`，记录延迟分布
- 紫外线文本格式是拉取式指标的事实标准

---

## 3. 从零实现

`code/main.py`实现`GenAISpan`、`SpanBuilder`、`JSONLExporter`、`Counter`/`Histogram`和`prometheus_exposition`。

```python
"""OTel GenAI Span + Prometheus指标——标准库Python。

核心：符合OpenTelemetry GenAI语义约定的span构建器，
JSONL导出器+Prometheus文本格式计数器/直方图。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, math, os, sys, tempfile, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# GenAI语义约定键
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
        return {"trace_id":self.trace_id,"span_id":self.span_id,"parent_span_id":self.parent_span_id,
            "name":self.name,"kind":self.kind,"start":self.start_unix_nano,"end":self.end_unix_nano,
            "duration_ms":round(self.duration_ms,4),"attributes":dict(self.attributes),
            "status":{"code":self.status,"message":self.status_message}}

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
    name:str; values:dict[tuple,tuple,float]=field(default_factory=dict)
    def inc(self,labels=None,by=1.0):
        k=tuple(sorted((labels or {}).items())); self.values[k]=self.values.get(k,0)+by
    def get(self,labels=None): return self.values.get(tuple(sorted((labels or {}).items())),0)

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
            counts=h.bucket_counts(dict(k)); lbl="{"+",".join(f'{k_}="{v_}"' for k_,v_ in sorted(k))+"}" if k else ""
            for b in h.buckets: lines.append(f"{h.name}_bucket{{{','.join([f'le="{b}"']+[f'{k_}="{v_}"' for k_,v_ in sorted(k)])}}} {counts[b]}")
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
    tmp=tempfile.mkdtemp(); trace_path=os.path.join(tmp,"traces.jsonl")
    jsonl=JSONLExporter(trace_path); mem=InMemoryExporter(); m=MetricsRegistry()
    b=SpanBuilder(exporters=[jsonl,mem],metrics=m)
    with b.span("gen_ai.chat",{GEN_AI_SYSTEM:"anthropic",GEN_AI_REQUEST_MODEL:"claude",GEN_AI_TOOL_NAME:"read_file"},kind="CLIENT") as chat:
        chat.attributes[GEN_AI_USAGE_INPUT_TOKENS]=412; chat.attributes[GEN_AI_USAGE_OUTPUT_TOKENS]=96
        with b.span("gen_ai.tool.execution",parent=chat,attributes={GEN_AI_TOOL_NAME:"read_file"}):
            time.sleep(0.005)
    print(f"emitted {len(mem.spans)} span(s)")
    for s in mem.spans: print(f"  {s.name:30s} dur={s.duration_ms:.2f}ms status={s.status}")
    print("\n--- prometheus ---\n"+prometheus_exposition(m))
    jsonl.close()
    with open(trace_path) as f: print(f"roundtrip: {len([json.loads(l) for l in f if l.strip()])} spans")

if __name__=="__main__": sys.exit(_demo())
```

---

## 4. 工具实践

**与第23-25节的集成**：
- 门决策记录为span事件
- 沙箱执行包装在`gen_ai.tool.execution` span中
- 评测运行包装在`eval.run` span中

**OTel SDK替代**：手写版教授线格式。生产中将相同属性接入OTel SDK，获得OTLP导出器、批处理和资源检测。

---

## 5. LLM视角

**结构化遥测视角**：OTel GenAI语义约定是标准——如果智能体写这些属性，每个OTel兼容后端都能读取。

**指标视角**：追踪回答"发生了什么"，指标回答"多久发生一次"。两者都需要——追踪看单次调用，指标看聚合趋势。

---

## 6. 工程最佳实践

**ID生成**：trace_id 16字节hex（W3C追踪上下文），span_id 8字节hex。

**直方图桶**：使用OTel默认毫秒桶集（5/10/25/50/100/250/500/1000/2500/5000/10000/+Inf）。

**错误处理**：导出器从不抛出——IO错误被表面化但智能体继续运行。

---

## 7. 常见错误

**错误1：使用自定义字段名**
症状：Grafana/Honeycomb无法解析
修复：使用GenAI语义约定标准键

**错误2：不聚合指标**
症状：只能看单次调用，无法看趋势
修复：Counter + Histogram + Prometheus文本格式

---

## 8. 面试考点

**Q1：为什么OTel GenAI语义约定比自定义字段名更好？**
考察：对标准化的理解

**Q2：追踪和指标的区别是什么？**
考察：对可观测性支柱的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI语义约定 | "OTel标准属性" | gen_ai.*键的标准集合，跨LLM框架共享 |
| JSONL导出器 | "每行一个span" | 最简单的可流式追踪格式 |
| Prometheus文本格式 | "指标导出" | 计数器+直方图的标准拉取式格式 |
| SpanBuilder | "追踪构建器" | 持有trace_id，通过上下文管理器发出span |
| 计数器 | "事件计数" | 每次工具调用递增的标记计数器 |
| 直方图 | "延迟分布" | 记录延迟样本并按桶聚合的度量 |

---

## 参考文献

- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Prometheus文本格式](https://prometheus.io/docs/instrumenting/exposition_formats/)
- [W3C追踪上下文](https://www.w3.org/TR/trace-context/)
