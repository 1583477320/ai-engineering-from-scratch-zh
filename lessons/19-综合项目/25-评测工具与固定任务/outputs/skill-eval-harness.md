# 评测工具技能

## 目标
构建接受固定任务文件夹、运行候选智能体、通过确定性验证器评分、聚合结果的评测工具。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 验证器正确 | file_equals/regex_match/shell_exit_zero全部工作 |
| 20 | pass@k数学 | pass@1和pass@k正确计算 |
| 20 | 固定任务加载 | JSON+buggy/expected正确加载 |
| 20 | 候选协议 | Candidate Callable接口正确 |
| 15 | 报告格式 | EvalReport含所有聚合指标 |

## 构建检查清单

- [ ] FixtureTask数据类
- [ ] SampleResult数据类
- [ ] VerificationOutcome数据类
- [ ] 三种验证器函数
- [ ] load_fixture/load_all加载
- [ ] EvalHarness.run方法
- [ ] 参考候选（apply_known_fixes）
- [ ] 无操作候选（noop_candidate）
- [ ] 演示：pass@1=1.0
