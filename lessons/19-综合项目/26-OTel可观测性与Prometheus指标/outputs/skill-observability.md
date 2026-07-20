# OTel可观测性技能

## 目标
构建符合OTel GenAI语义约定的span构建器、JSONL导出器和Prometheus指标。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | Span结构正确 | 符合OTel GenAI语义约定 |
| 20 | JSONL往返 | json.loads成功解析所有span |
| 20 | 计数器和直方图 | 工具调用计数和延迟分布正确 |
| 20 | Prometheus文本格式 | 标准导出格式正确 |
| 15 | 上下文管理器 | span()正确记录开始/结束/异常 |

## 构建检查清单

- [ ] GenAISpan数据类（符合OTel规范）
- [ ] SpanBuilder（trace_id/span/context manager）
- [ ] JSONLExporter（每行一个span）
- [ ] InMemoryExporter（测试用）
- [ ] Counter和Histogram
- [ ] MetricsRegistry
- [ ] prometheus_exposition函数
- [ ] 演示：chat span + tool span + 指标 + Prometheus输出
