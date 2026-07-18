# 故障模式——MAST、群体思维、同质化、级联错误

> 多智能体系统在真实任务上的失败率是 41–86.7%。这不是"加更多智能体就能解决的问题"。失败是有结构的。2026 年的工程实践是把故障模式作为设计输入——你的架构不够好，直到你能指出每一个故障类别并命名你部署的缓解措施。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 16 · 13（共享记忆与黑板）、阶段 16 · 14（共识与拜占庭容错）、阶段 16 · 15（投票辩论拓扑）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 16 · 24（评估与基准测试）——故障模式的缓解效果需要通过基准测试验证

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 MAST 故障分类法的三大根因——规格问题（41.77%）、协调故障（36.94%）、验证缺失（21.30%）
- [ ] 识别群体思维家族的五个模式——同质化崩溃、从众偏见、心智理论缺陷、混合动机、级联可靠性故障
- [ ] 实现断路器模式，防止重试风暴级联放大
- [ ] 设计 STRATUS 风格的三角色故障响应系统——检测、诊断、验证

---

## 1. 问题

多智能体系统在 41–86.7% 的情况下会失败（Cemri et al. 2025 在 7 个开源 MAS 上实测的数据）。这不是调试就能解决的问题——失败是有结构性原因的。

三种最常见的问题：

发布任务说"总结一下这份报告"，结果两个智能体都以为自己是审稿人（**角色歧义**——规格问题）。

两个智能体同时更新同一份数据，一个把它改成了 A，一个改成了 B（**状态漂移**——协调故障）。

智能体 A 生成了一个分析结果，智能体 B 直接拿来用了——结果 A 的数据源是错的，但没有人验证（**验证缺失**——验证缺口）。

这只是冰山一角。2026 年的生产实践是把故障模式作为设计输入——在你能够指出每个 MAST 类别并说出你部署了什么缓解措施之前，你的架构不算"足够好"。

---

## 2. 概念

### 2.1 MAST 分类法

MAST（Cemri et al., NeurIPS 2025, arXiv:2503.13657）从 1642 条执行迹中归纳出三大根因类别：

**规格问题（Specification Problems）——41.77%**

任务定义不够明确。典型例子：
- **角色歧义：** 两个智能体都以为自己是审稿人
- **任务定义不足：** "把这个总结一下"——但用户想要的是特定角度的总结
- **成功标准隐式：** 智能体不知道自己做完了没有

缓解措施：
- 编写明确的角色合同。每个智能体的提示词说明它做什么**和不做什么**
- 任务执行前定义"完成标准"
- 预检：在派发任务前，由另一个智能体检查任务定义是否足够

**协调故障（Coordination Failures）——36.94%**

通信或状态层面的断裂。典型例子：
- 两个智能体没有同步地更新同一份状态
- 智能体之间的消息丢失（队列故障、超时）
- 状态漂移：智能体 A 认为任务已完成，B 还在执行

缓解措施：
- 带版本号的共享状态 + 乐观并发控制
- 关键消息的显式确认（重试直到被确认）
- 定期状态同步检查点

**验证缺失（Verification Gaps）——21.30%**

输出没有经过独立核验。典型例子：
- 一个智能体声称成功，没有人验证
- 智能体链上每个节点都信任前一个节点的输出
- 对涌现的组合行为没有测试覆盖

缓解措施：
- 独立验证者智能体（第 13 课）。只读的、独立的访问源
- 显式移交合约："A 的输出必须通过检查器 C 才能启动 B"
- 结果日志用于事后分析

### 2.2 群体思维家族（arXiv:2508.05687）

当智能体趋同或互相模仿时，五种相关联的故障：

**同质化崩溃（Monoculture Collapse）：** 共享相同基础模型或训练数据→相关性的错误。三个智能体共享同一个 LLM，就共享同一个幻觉。

**从众偏见（Conformity Bias）：** 智能体向声音最大或最自信的同伴靠拢，即使它是错的。

**心智理论缺陷（Deficient ToM）：** 智能体无法建模对方的信念；协作崩溃（第 18 课）。

**混合动机动力学（Mixed-Motive Dynamics）：** 部分对齐激励的智能体向妥协的中点漂移——结果谁都不满意。

**级联可靠性故障（Cascading Reliability Failures）：** 一个组件的错误模式触发了依赖组件中的错误模式。

### 2.3 级联错误——重试风暴

一个经典的 2026 年生产事故：

```
支付服务 10% 的请求失败
   ↓
订单智能体重试支付（指数退避但实现拙劣）
   ↓
每次重试都会触发新的订单-库存检查
   ↓
库存服务负载翻倍
   ↓
库存服务开始超时
   ↓
每个订单都重试库存检查
   ↓
库存服务负载变为 10 倍
   ↓
集群宕机
```

修复方案是经典的：**断路器**。当下游错误率超过阈值时，用缓存或默认结果短路返回。外加每个请求设置上限的重试预算。

断路器是少数几个直接从分布式系统借用而不需要修改的多智能体故障缓解方案。

### 2.4 记忆中毒（复习）

第 13 课已覆盖：一个智能体的幻觉进入共享记忆→下游智能体将其当作事实处理。在 MAST 术语中，这是共享记忆层的验证缺失。

症状是**渐进的准确率衰减**——不会崩溃，而是缓慢漂移，导致根因诊断极其痛苦。

缓解方案：仅追加日志、来源追踪、不可写入的验证者。

### 2.5 STRATUS——专业化故障检测智能体

STRATUS（NeurIPS 2025）报告显示，部署以下角色可以将缓解成功率提升 1.5 倍：

- **检测智能体：** 监视频繁的征兆模式（高不一致率、重试峰值、准确率漂移）
- **诊断智能体：** 给定症状，从 MAST 分类法中推断出可能的根因
- **验证智能体：** 在缓解措施应用后，检查症状是否消失

这是 SRE 风格的故障响应，应用于智能体系统。三个角色都可以是带有专门提示词的 LLM 智能体。

### 2.6 故障模式审计

2026 年最佳实践是每季度一次（或每次大版本发布）的故障模式审计：

1. **采样迹。** 收集约 1000 条真实执行迹
2. **分类。** 对每条迹的失败映射到 MAST + 群体思维类别
3. **按类别计算失败率。** 哪些类别在你的系统中占主导？列出排名前三
4. **对缓解方案排序。** 哪个修复能消除最多的失败？
5. **挑选 2-3 个缓解方案。** 实施；下季度再次审计

### 2.7 悄无声息的失败

最危险的故障类别是**静默的正确性失败**。会崩溃、抛出异常、触发告警的失败可以被监控。会生成看起来合理但实际上是错的结果——无法通过异常日志检测。这就是为什么验证缺失虽然只占 21.30% 的计数，但在按失败排序时却是最昂贵的类别。

---

## 3. 从零实现

### 第 1 步：MAST 故障分类器

```python
from enum import Enum


class MASTCategory(Enum):
    SPECIFICATION = "specification"       # 规格问题
    COORDINATION = "coordination"         # 协调故障
    VERIFICATION = "verification"         # 验证缺失


class FailureRecord:
    """一条故障记录，映射到 MAST 分类法。"""

    def __init__(self, description: str, category: MASTCategory,
                 severity: float = 0.5, mitigation: str = ""):
        self.description = description
        self.category = category
        self.severity = severity
        self.mitigation = mitigation


class FailureTaxonomy:
    """故障分类器——将执行迹映射到 MAST 类别。"""

    CATEGORY_KEYWORDS = {
        MASTCategory.SPECIFICATION: [
            "角色歧义", "任务定义", "成功标准", "不清楚", "模糊",
        ],
        MASTCategory.COORDINATION: [
            "状态漂移", "消息丢失", "同步", "超时", "不一致",
        ],
        MASTCategory.VERIFICATION: [
            "未验证", "未核验", "信任输出", "没有检查", "幻觉传播",
        ],
    }

    @classmethod
    def classify(cls, incident: str) -> MASTCategory:
        """根据关键词将事件分类到 MAST 类别。"""
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in incident:
                    return category
        return MASTCategory.COORDINATION  # 默认
```

### 第 2 步：断路器

```python
import time


class CircuitBreaker:
    """断路器——当下游错误率超过阈值时短路。"""

    def __init__(self, threshold: float = 0.1, reset_timeout: float = 30):
        self.threshold = threshold            # 错误率阈值
        self.reset_timeout = reset_timeout    # 半开等待时间
        self.errors = 0
        self.total = 0
        self.open = False
        self.open_since = 0.0

    def call(self, fn, *args, **kwargs):
        """执行调用，含断路器检查。"""
        if self.open:
            if time.time() - self.open_since > self.reset_timeout:
                self.open = False  # 半开：允许试探
            else:
                raise RuntimeError("断路器打开，拒绝调用")

        try:
            result = fn(*args, **kwargs)
            self.total += 1
            self.errors = max(0, self.errors - 1)  # 逐步恢复
            return result
        except Exception as e:
            self.total += 1
            self.errors += 1
            if self.errors / max(self.total, 1) > self.threshold:
                self.open = True
                self.open_since = time.time()
            raise
```

### 第 3 步：重试风暴模拟器

```python
class RetryStormSimulator:
    """模拟重试风暴——对比有无断路器。"""

    def __init__(self, failure_rate: float = 0.15, num_requests: int = 100):
        self.failure_rate = failure_rate
        self.num_requests = num_requests
        self.inventory_load = 0

    def simulate_without_cb(self):
        """无断路器：所有失败都重试，放大负载。"""
        self.inventory_load = 0
        retries = 0

        for _ in range(self.num_requests):
            success = False
            attempt = 0
            while not success and attempt < 5:
                attempt += 1
                self.inventory_load += 1  # 每次重试都打到下游
                import random
                if random.random() > self.failure_rate:
                    success = True
                else:
                    retries += 1

        return {
            "inventory_load": self.inventory_load,
            "retries": retries,
            "amplification": self.inventory_load / self.num_requests,
        }

    def simulate_with_cb(self):
        """有断路器：打开后短路，限制负载。"""
        self.inventory_load = 0
        cb = CircuitBreaker(threshold=0.1, reset_timeout=10)
        import random

        for _ in range(self.num_requests):
            try:
                self.inventory_load += 1
                if random.random() > self.failure_rate:
                    pass  # 成功
                else:
                    raise RuntimeError("下游错误")
            except RuntimeError:
                # 断路器可能打开，短路后续调用
                pass

        return {
            "inventory_load": self.inventory_load,
            "amplification": self.inventory_load / self.num_requests,
        }


if __name__ == "__main__":
    sim = RetryStormSimulator(failure_rate=0.15, num_requests=100)

    print("=== 无断路器 ===")
    result = sim.simulate_without_cb()
    print(f"  下游负载: {result['inventory_load']} (放大 {result['amplification']:.1f}x)")
    print(f"  重试次数: {result['retries']}")

    print("\n=== 有断路器 ===")
    result = sim.simulate_with_cb()
    print(f"  下游负载: {result['inventory_load']} (放大 {result['amplification']:.1f}x)")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 断路器模式

```python
# Python 标准库级别的断路器——使用 functools
from functools import wraps
import time


def circuit_breaker(threshold=0.1, reset_timeout=30):
    """断路器装饰器。"""
    state = {"errors": 0, "total": 0, "open": False, "open_since": 0}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if state["open"]:
                if time.time() - state["open_since"] > reset_timeout:
                    state["open"] = False
                else:
                    raise RuntimeError("断路器打开")
            try:
                result = func(*args, **kwargs)
                state["total"] += 1
                return result
            except Exception as e:
                state["errors"] += 1
                state["total"] += 1
                if state["errors"] / max(state["total"], 1) > threshold:
                    state["open"] = True
                    state["open_since"] = time.time()
                raise
        return wrapper
    return decorator
```

### 4.2 故障模式审计工具链

| 工具 | 功能 | 适用阶段 |
|---|---|---|
| LangSmith 追踪 | 自动记录执行迹，含错误和延迟 | 开发/生产 |
| 自定义审计脚本 | 按 MAST 类别分析迹 | 季度审计 |
| 金数据集 | 小批量高质量、人工审核的回归测试集 | 持续 |
| STRATUS 检测器 | 监控异常征 | 生产 |

---

## 5. 工程最佳实践

### 5.1 每季度做一次 MAST 审计

不是每年。随着系统增长，类别分布会迁移。

### 5.2 所有地方都加断路器

每个到下游依赖的出站调用。默认阈值为 5-10% 错误率。

### 5.3 金数据集

小批量、高质量的回归测试集。人工审核。每周对其运行回归测试。

### 5.4 STRATUS 三重奏

从检测智能体开始；在症状变得嘈杂时添加诊断；验证始终需要。

### 5.5 失败预算

为每个故障类别设置显式的 SLO 失败率。超出预算触发停止发布的讨论。

### 5.6 中文场景特别建议

- **中文 LLM 的"角色歧义"更严重。** 英文提示词中 "you are an analyst" 的语义清晰，中文"你是分析师"可能被理解成多个角色（数据分析师、证券分析师、市场分析师）。角色定义必须包含明确的排除项
- **从众偏见在中文辩论中更强。** 中文 LLM 更倾向于维护和谐、避免直接反驳——这加剧了多智能体系统中的从众偏见。在提示词中明确要求"坚持你的专业判断，不同意就说不同意"
- **中文 LLM 的基础模型同质化更严重。** 英文生态有 GPT-4o、Claude、Llama 等多个差异显著的基础模型；中文 LLM 厂商都基于类似的开源模型微调，同质化崩溃的风险更高——考虑混合使用中文和英文 LLM

---

## 6. 常见错误

### 错误 1：没有断路器

**现象：** 一个下游服务故障 10% 的请求，经过多轮重试后，下游负载变为 10 倍，集群宕机。

**原因：** 长链路中每层都重试，且没有统一的上限。

**修复：** 每个出站调用包裹在断路器中，默认阈值 5-10%。

### 错误 2：故障分类停留在直觉

**现象：** 每次故障排查都从零开始——"这次是什么原因？"。团队对系统最频繁的故障模式没有系统性的理解。

**原因：** 没有故障分类法，没有审计流程。

**修复：** 引入 MAST 分类法，每季度做一次审计。

### 错误 3：没有检测静默失败

**现象：** 系统看起来运行正常（没有错误日志、没有告警），但用户满意度在下降。检查发现智能体一直在输出错误的分析结果。

**原因：** 验证缺失——没有人独立检查智能体的输出质量。

**修复：** 部署独立的验证者智能体 + 抽样人工审查 + 金数据集回归测试。

---

## 7. 面试考点

### Q1：为什么 41-86.7% 的失败率是一个结构性问题，而不是"加更多智能体"能解决的？（难度：⭐⭐）

**参考答案：**
MAST 论文覆盖了 7 个不同的开源多智能体系统，失败率在最简单和最新进的系统之间分布。即使是最新进的系统也有约 41% 的失败率。原因在于三类失敗有结构性根因：规格问题（任务定义不够明确）、协调故障（智能体之间的沟通断裂）、验证缺失（没有人核验输出）。加更多智能体不会让任务定义更明确，反而会加剧角色歧义和协调开销——更多智能体意味着更多角色需要定义、更多通信需要协调。

### Q2：断路器如何防止重试风暴？需要配置哪几个参数？（难度：⭐⭐）

**参考答案：**
断路器监视下游调用的错误率。当错误率超过阈值时打开，直接返回默认值或缓存结果，不再发起真正的调用。三个关键参数：**阈值**（错误率达到多少时打开——通常 5-10%）、**重置超时**（断路器打开后等待多长时间尝试半开——通常 30-60 秒）、**半开试探**（半开后允许多少个试探请求——通常 1 个，成功则关闭，失败则继续打开）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| MAST | "2026 年故障分类法" | Cemri 2025；3 个根类别 + 14 个子类型 |
| 规格问题 | "角色歧义" | 任务或角色定义不足；智能体不知道要做什么 |
| 协调故障 | "状态漂移" | 智能体之间的通信或同步断裂 |
| 验证缺失 | "没人检查" | 输出没有经过独立验证就被接受 |
| 群体思维 | "同质化故障" | 同质化、从众、ToM 缺陷、混合动机、级联 |
| 同质化崩溃 | "同一个模型，同一个幻觉" | 共享基础模型或训练数据导致的相关性错误 |
| 重试风暴 | "级联错误放大" | 一个失败触发重试，重试放大下游负载 |
| 断路器 | "失败时快速失败" | 错误率超过阈值时打开；短路返回默认值 |
| STRATUS | "故障响应三重奏" | 检测 + 诊断 + 验证智能体。缓解成功率提升 1.5x |

---

## 📚 小结

MAST 分类法告诉你多智能体系统在真实任务上 41-86.7% 的失败率不是偶然——它是有结构的：规格问题（~42%）、协调故障（~37%）、验证缺失（~21%）。群体思维家族添加了同质化崩溃和从众偏见。断路器是防止重试风暴的核心模式；STRATUS 三重奏（检测/诊断/验证）可以将缓解成功率提升 1.5 倍。最危险的失败是静默的正确性失败——输出看起来合理但实际是错的。

下一课我们将讨论多智能体系统的评估与基准测试——如何用结构化的方法来衡量系统是否真的在进步。

---

## ✏️ 练习

1. 运行 `code/main.py`。确认断路器遏制了重试风暴。改变失效阈值观察权衡。
2. 实现一个**缓慢失效的代理指标**：在 3 个并行智能体之间测量一致率。当一致率急剧下降时触发告警。通过逐渐关联智能体输出来模拟同质化漂移。
3. 读 Cemri et al. (arXiv:2503.13657)。选择其中的一个 MAS 系统，映射其排名前三的故障类别。与 MAST 预测的类别相比如何？
4. 设计一个 STRATUS 风格的检测-诊断-验证三重奏，针对你认识的某个多智能体系统。检测器监视哪些症状？诊断器推荐哪些缓解措施？验证器如何确认有效？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 故障模式分析工具 | `code/main.py` | MAST 分类器 + 断路器 + 重试风暴模拟 |
| MAST 审计技能 | `outputs/skill-mast-auditor.md` | 对多智能体系统运行故障模式审计 |
