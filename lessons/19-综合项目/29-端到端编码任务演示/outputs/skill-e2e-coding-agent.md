# 端到端编码智能体配方

## 五状态策略

SURVEY → RUN_TESTS → INSPECT → FIX → VERIFY → 成功/失败

## 五个断言

1. 步数 < 12
2. 观测预算未超
3. 零门拒绝
4. 每步有 span
5. Prometheus 包含计数器和直方图

## 替换为真实 LLM

修改策略层的 policy 函数即可，其他基础设施不变
