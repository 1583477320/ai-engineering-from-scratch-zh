# DevOps故障排查智能体技能

## 目标
给定K8s集群和告警源，智能体产出排名根因假设和Slack审批门的修复流程。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | RCA准确率 | 20个合成故障中≥80%正确根因 |
| 20 | 安全性 | 破坏性操作未经Slack审批绝不执行 |
| 20 | 假设产出时间 | 从告警到Slack简报，p50 < 5分钟 |
| 20 | 可解释性 | 每个假设有图谱路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus端到端 |

## 构建检查清单

- [ ] K8s知识图谱（kube-state-metrics同步）
- [ ] 告警接收器（PagerDuty/Alertmanager webhook）
- [ ] 只读工具表面（kubectl get/describe, promql, logql）
- [ ] 根因假设排名
- [ ] Slack审批卡
- [ ] 修复审批门（破坏性操作）
- [ ] 审计日志（考虑过 vs 已执行）
- [ ] 20个合成故障场景
