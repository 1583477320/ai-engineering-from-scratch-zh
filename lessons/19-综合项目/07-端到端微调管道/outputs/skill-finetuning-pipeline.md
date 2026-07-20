# 微调管道技能

## 目标
单条命令运行数据→SFT→DPO→量化→部署→评测，输出模型卡+服务端点。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 与基线相比的评测提升 | MMLU-Pro、MT-Bench-v2等目标任务的提升 |
| 20 | 管道可复现性 | 一条命令端到端重跑，相同种子 |
| 20 | 数据卫生 | 去重率、PII清洗覆盖、污染检查通过 |
| 20 | 服务效率 | 不同批大小的token/s、EAGLE-3接受率、$/100万token |
| 15 | 模型卡+安全评测 | MOF 2026完整性+Llama Guard 4通过率 |

## 构建检查清单

- [ ] 数据去重（Datatrove MinHash）
- [ ] 质量过滤（Nemotron-CC风格）
- [ ] PII清洗（Presidio）
- [ ] 污染检查（MinHash-LSH）
- [ ] Axolotl SFT配置
- [ ] TRL DPO/GRPO配置
- [ ] GPTQ+AWQ+GGUF量化
- [ ] vLLM+EAGLE-3部署
- [ ] 评测矩阵（4个基准）
- [ ] MOF 2026模型卡
- [ ] 安全评测（Llama Guard 4）
