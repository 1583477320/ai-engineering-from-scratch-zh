# 端到端安全门配方

## 三检查点

生成前(detector) → 生成中(stream filter) → 生成后(classifier+rules)

## 聚合表

| 信号 | 动作 |
|:----|:-----|
| any high | block |
| any medium | redact |
| any low | warn |
| 全部 none | allow |

## 输出

每次请求的 RequestTrace（审计轨迹）
