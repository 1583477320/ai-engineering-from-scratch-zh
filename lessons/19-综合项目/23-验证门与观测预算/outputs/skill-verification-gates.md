# 验证门技能

## 目标
构建带短路语义的门链和观测账本，控制工具调用的可见性和预算。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 门链短路 | 第一个拒绝即终止 |
| 20 | 四门正确 | 白名单/正则/新鲜度/预算全部工作 |
| 20 | 观测账本 | 累积token和per-tool token正确 |
| 20 | GateDecision结构化 | allow/gate/reason正确输出 |
| 15 | 合成循环端到端 | 三轮循环+拒绝正确触发 |

## 构建检查清单

- [ ] ToolCall/Observation/GateDecision数据类
- [ ] ObservationLedger（record/cumulative/per_tool/latest_turn）
- [ ] GateContext（ledger+current_turn+history）
- [ ] WhitelistGate（显式允许集合）
- [ ] RegexGate（正则拒绝模式）
- [ ] RecencyGate（最近N轮窗口）
- [ ] BudgetGate（累积token上限）
- [ ] GateChain（有序短路评估）
- [ ] 合成循环演示
