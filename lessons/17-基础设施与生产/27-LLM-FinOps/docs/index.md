# LLM FinOps——单位经济性与多租户归因

> 传统 FinOps 在 LLM 支出上失效。成本是词元交易，不是资源占用时间。标签不适用——API 调用是一个交易，不是资源。工程决策（提示词设计、上下文窗口、输出长度）就是财务决策。2026 年的实践指南有三个归因维度：按用户、按任务、按租户。四个词元层——提示词、工具、记忆、响应——单一桶隐藏成本。多租户产品的执法阶梯：速率限制→每日支出上限→终止开关。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 13（可观测性）、阶段 17 · 14（缓存）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么传统 FinOps（标签+层级）在 LLM 支出上失效，并说出三个新的归因维度
- [ ] 列举四个词元层（提示词、工具、记忆、响应）以及为什么单一桶计费隐藏成本
- [ ] 为多租户产品设计执法阶梯（速率限制→支出上限→终止开关）
- [ ] 选择单位指标（每解决问题成本/每产出成本）而不是 $/M 词元

---

## 1. 问题

你的账单显示 $40,000。你不知道：
- 哪个租户花的
- 哪个产品功能驱动的
- 是否有某个用户滥用
- 是提示词膨胀、工具调用还是记忆放大是罪魁祸首

标签+聚合在云资源（EC2、S3）上有效——标签传播到账单行项目。LLM API 调用不自动打标签——你必须在调用处标记 user/task/tenant 并携带。事后再做归因总是遗漏边界情况。

---

## 2. 概念

### 2.1 三个归因维度

**按用户（`user_id`）：** 谁花了多少。驱动座位定价和扩增对话，识别高消耗用户。

**按任务（`task_id` + `route`）：** 哪个产品面成本多少。驱动功能优先级排序和高成本功能停用决策。

**按租户（`tenant_id`）：** 哪个客户是盈利的。驱动单位经济学、续约定价、层级阈值。

三个都在调用处从第一天开始检测。事后方案总是更差。

### 2.2 四个词元层

| 层 | 举例 | 占总量的典型比例 |
|---|---|---|
| 提示词 | 系统+用户输入 | 40-60% |
| 工具 | 工具调用结果反馈 | 20-40%（智能体工作负载） |
| 记忆 | 之前对话/检索文档 | 10-30% |
| 响应 | 模型输出 | 10-30% |

将四层混在一起使优化盲目——在归因 schema 中分拆它们。

### 2.3 执法阶梯

1. **速率限制** 每租户。预期峰值的 2-3 倍。返回 429 + `Retry-After`。租户感到摩擦；没有意外账单。
2. **每日支出上限** 每租户。合同上限的 1.5-3 倍。触发：收紧速率限制 + 通知客户成功团队。
3. **终止开关** 支出 z-score > 4（相对于租户基线）。自动暂停租户；通知值班人员；升级到运维+客户成功。

### 2.4 单位指标

$/M 词元是供应商语言。产品指标：

- 每解决支持工单成本
- 每生成文章成本
- 每成功智能体任务成本
- 每用户会话分钟成本

将成本绑定到产品结果。否则优化没有锚点。

### 2.5 归因跟踪形状

```python
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
```

每次调用都发出。存储在数据湖中。按维度聚合。

### 2.6 叠加优化栈

栈：缓存 + 批处理 + 路由 + 网关。四个都启用时：
- L2 缓存（第 17 · 14 课）：输入便宜约 10 倍
- 批处理（第 17 · 15 课）：50% 折扣
- 路由到廉价模型（第 17 · 16 课）：60% 成本降低
- 网关效率（第 17 · 19 课）：冗余+重试

最佳情况叠加：~5-10% 的朴素基线。大多数团队有 2-3 个杠杆；很少四个都用。

---

## 3. 从零实现

### 第 1 步：归因跟踪和终止开关

```python
import statistics


class CostTracker:
    """多租户成本跟踪和终止开关。"""

    def __init__(self):
        self.tenant_usage = {}
        self.tenant_baselines = {}

    def record_call(self, tenant_id, cost):
        if tenant_id not in self.tenant_usage:
            self.tenant_usage[tenant_id] = []
        self.tenant_usage[tenant_id].append(cost)

    def check_kill_switch(self, tenant_id, z_threshold=4):
        if tenant_id not in self.tenant_usage or len(self.tenant_usage[tenant_id]) < 10:
            return {"triggered": False}
        costs = self.tenant_usage[tenant_id]
        recent = costs[-5:]
        mu = statistics.mean(costs)
        sigma = max(statistics.stdev(costs), 0.001)
        z_score = (sum(recent) / len(recent) - mu) / sigma
        return {"triggered": z_score > z_threshold, "z_score": z_score,
                "action": "暂停租户" if z_score > z_threshold else "继续"}


# 演示
tracker = CostTracker()
for i in range(100):
    cost = 0.01 * (1 + 0.1 * (i > 90))  # 最后 10 次突发
    tracker.record_call("tenant-1", cost)

result = tracker.check_kill_switch("tenant-1")
print(f"z-score: {result['z_score']:.2f}  触发: {result['triggered']}  动作: {result['action']}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 归因模式对照

| 模式 | 精度 | 复杂度 | 最佳场景 |
|---|---|---|---|
| 标签+聚合 | 低 | 低 | 粗略估算 |
| 遥测连接器 | 最高 | 高 | 成熟团队 |
| 采样+外推 | 中 | 中 | 费用粗略估算 |
| 基于模型 | 低 | 中 | 无标签的历史数据 |
| 事件溯源 | 高 | 高 | 实时追踪 |

---

## 5. 工程最佳实践

### 5.1 归因从第一天开始

第一天就在每个 API 调用上打 tag——`user_id`、`task_id`、`tenant_id`。事后归因总是遗漏边界情况且精度低。

### 5.2 单位指标选业务相关

$/M 词元是供应商语言。产品决策需要"每解决工单成本"或"每生成文章成本"。

### 5.3 中文场景特别建议

- **国内 LLM 计费模式不同。** 国内 LLM API 的计费模式与 OpenAI/Anthropic 不同（有些按字符计费而非词元）。归因系统需要适配
- **多租户场景更有价值。** 国内 SaaS 企业的多租户 LLM 归因需求强烈——不同租户使用量差异大。终止开关可以防止租户滥用
- **叠加优化栈在国内同样有效。** 缓存+批处理+路由+网关的组合在国内 LLM API 上同样适用

---

## 6. 常见错误

### 错误 1：不用词元层拆分成本

**现象：** 看到成本飙升但不知道是提示词太长还是工具调用太多。

**原因：** 四个词元层（提示词、工具、记忆、响应）混在一个桶中。

**修复：** 在归因 schema 中分拆四层——每层的词元数和成本单独跟踪。

### 错误 2：用 $/M 词元做产品决策

**现象：** 团队在优化 $/M 词元，但产品指标（每解决工单成本）没有改善。

**原因：** $/M 词元是供应商指标，不是产品指标。

**修复：** 用"每解决工单成本"或"每生成文章成本"等业务相关指标。

---

## 7. 面试考点

### Q1：传统的 FinOps 标签+聚合为什么在 LLM 支出上失效？（难度：⭐⭐）

**参考答案：**
传统云资源的成本是资源占用时间（EC2 实例运行 X 小时），标签可以自动传播到账单。LLM 的支出是词元交易（一个 API 调用），不是资源占用——标签不会自动传播到 API 调用。你需要手动在调用处标记 user_id/task_id/tenant_id 并在整个追踪链中携带。传统工具（AWS Cost Explorer、Azure Cost Management）无法按词元交易维度查看成本。

### Q2：执法阶梯的三个层级是什么？（难度：⭐⭐）

**参考答案：**
第一层：速率限制（2-3x 预期峰值，429+Retry-After）。第二层：每日支出上限（1.5-3x 合同上限，触发时收紧速率限制+通知客户成功）。第三层：终止开关（支出 z-score > 4 相对基线，自动暂停租户+通知值班）。三层逐级递进——先摩擦（速率限制）、再告警（支出上限）、最后切断（终止开关）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 按用户归因 | "用户级成本" | 每次调用标记 user_id |
| 按任务归因 | "功能成本" | task_id + route 标识产品面 |
| 四个词元层 | "成本层" | 提示词+工具+记忆+响应 |
| 速率限制 | "429 守卫" | 网关逐租户上限 |
| 终止开关 | "自动暂停" | 支出 z-score > 4 触发自动暂停 |
| 单位指标 | "产品单位指标" | 成本绑定到产品结果，非词元 |

---

## 📚 小结

LLM FinOps 的三个核心实践：从第一天开始做三纬度归因（用户/任务/租户）、分四个词元层跟踪成本（提示词/工具/记忆/响应）、设计三层执法阶梯（速率限制/支出上限/终止开关）。单位指标选业务相关的"每解决工单成本"而非供应商指标 $/M 词元。叠加优化（缓存+批处理+路由+网关）可将成本降到基线的约 5-10%。

---

## ✏️ 练习

1. 运行 `code/main.py`。终止开关在什么 z-score 下触发？如何选择阈值？
2. 设计一个按租户+按任务的成本仪表盘。你优先构建哪 5 个视图？
3. 你的最大租户是单位经济性负的。提出三个按客户影响排序的干预措施。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 成本跟踪器 | `code/main.py` | 多租户成本跟踪和终止开关 |
| FinOps 方案 | `outputs/skill-finops-plan.md` | 归因 schema 和执法阶梯设计 |

---

## 📖 参考资料

1. [官方文档] FinOps Foundation — FinOps for AI Overview. https://www.finops.org/wg/finops-for-ai-overview/
2. [博客] Digital Applied — LLM Agent Cost Attribution 2026. https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026
3. [博客] FinOps School — Cost per Unit 2026 Guide. https://finopsschool.com/blog/cost-per-unit/
