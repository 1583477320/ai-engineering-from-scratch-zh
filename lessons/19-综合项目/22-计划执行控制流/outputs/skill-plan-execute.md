# 计划执行控制流技能

## 目标
实现带重新规划、计划差异和双预算的结构化计划执行。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 重新规划正确 | 失败后带错误上下文从游标获取新计划 |
| 20 | 计划差异准确 | removed/added/revised正确计算 |
| 20 | 步骤预算执行 | max_steps正确触发FAILED |
| 20 | 重新规划预算 | max_replans正确触发FAILED |
| 15 | 事件流完整 | plan.commit/draft/diff/start/end正确发射 |

## 构建检查清单

- [ ] Step数据类（id/tool_name/args/expected_outcome/result/error）
- [ ] PlanDiff数据类（revision/removed/added/revised）
- [ ] PlanExecuteAgent类（run方法）
- [ ] 确定性计划器（根据last_error选择计划）
- [ ] 计划差异计算（_diff_plans）
- [ ] 双预算执行（max_steps + max_replans）
- [ ] 事件发射（plan.commit/draft/diff/step.start/end/session.complete）
- [ ] 演示：线性成功+一次重新规划+重新规划耗尽
