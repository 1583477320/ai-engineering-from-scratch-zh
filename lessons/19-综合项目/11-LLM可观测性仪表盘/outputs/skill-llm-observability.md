# LLM可观测性技能

## 目标
给定LLM应用，仪表盘导入追踪、运行评测、告警漂移、在Next.js中展示每用户成本分解。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 追踪模式覆盖 | 产生规范GenAI span的SDK家族数（目标6+） |
| 20 | 评测正确性 | DeepEval/RAGAS分数 vs 手动标注集 |
| 20 | 仪表盘UX | 注入回归的MTTR（目标<5分钟） |
| 20 | 成本/规模 | 持续1k span/s无积压 |
| 15 | 告警+漂移检测 | Prometheus/Alertmanager链路端到端 |

## 构建检查清单

- [ ] OpenTelemetry Collector配置
- [ ] 尾采样器（100%错误+10%成功）
- [ ] ClickHouse span存储
- [ ] 6个SDK家族覆盖（OpenAI、Anthropic、Google等）
- [ ] DeepEval评测作业（忠诚度、毒性）
- [ ] 自定义PII-leak LLM-judge
- [ ] PSI漂移检测
- [ ] Prometheus Alertmanager告警链
- [ ] Next.js仪表盘（概览、追踪、评测、漂移）
