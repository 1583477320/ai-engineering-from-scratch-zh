# OTel 可观测性配方

## Span 属性

```
gen_ai.system, gen_ai.request.model
gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
gen_ai.tool.name, gen_ai.tool.call.id
```

## Prometheus 指标

- 计数器: tools_called_total{tool="..."}
- 直方图: tool_latency_ms{tool="..."} (OTel 默认桶: 5,10,25,50,100,250,500,1000)
