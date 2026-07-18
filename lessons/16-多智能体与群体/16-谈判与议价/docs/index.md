# 谈判与议价

> 智能体协商资源、价格、任务分配和条款。2026 年基准集清楚：NegotiationArena（arXiv:2402.05863）显示 LLM 通过角色操纵（"绝望"）可将收益提高 ~20%；"Measuring Bargaining Abilities"（arXiv:2402.15813）显示买方比卖方更难，规模没有帮助——他们的 OG-Narrator（确定性报价生成器 + LLM 叙述）将成交率从 26.67% 提高到 88.88%；大规模自主谈判竞赛（arXiv:2503.06416）运行了约 18 万次谈判，发现**思维链隐藏**的智能体通过向对手隐藏推理而获胜；Bhattacharya 等人 2025 年关于哈佛谈判项目指标的研究将 Llama-3 排为最有效，Claude-3 最具攻击性，GPT-4 最公平。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 02（FIPA-ACL 遗产）、阶段 16 · 09（并行群体网络）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现合同网协议（FIPA 祖先）——管理者广播 cfp、投标者投标、管理者颁奖
- [ ] 对比朴素 LLM 议价和 OG-Narrator（确定性报价 + LLM 叙述）——理解为什么分解机制和叙述层能提高成交率
- [ ] 理解思维链隐藏——赢家如何隐藏推理以获取竞争优势
- [ ] 理解为什么"让 LLM 叙述，不要让 LLM 计算报价"是关键原则

---

## 1. 问题

两个智能体需要就价格达成一致。用纯语言提示，2024-2026 年的 LLM 在紧密参数化议价中成交率低得惊人（arXiv:2402.15813 中约 27%）。规模没有帮助：GPT-4 在议价结构上并不比 GPT-3.5 好；它在议价的*语言*上更好。

根本问题是 LLM 混淆了两项工作——决定报价和叙述报价。OG-Narrator 将这两者分离：确定性报价生成器计算数字动作；LLM 只叙述。成交率跃升至 ~89%。

---

## 2. 概念

### 2.1 合同网——一段话总结

Smith 1980 年的合同网协议：**管理者**广播**征求提案 (cfp)**；**投标者**用 **propose** 消息回复其报价；管理者选择赢家并发送 **accept-proposal** 给赢家，**reject-proposal** 给输家。FIPA 将其编纂为 `fipa-contract-net` 交互协议。

### 2.2 为什么 OG-Narrator 胜出

| 问题 | 朴素 LLM | OG-Narrator |
|------|---------|-------------|
| 报价 | 方差高，经常在 ZOPA 之外 | 确定性，始终在 ZOPA 内 |
| 锚点 | 情绪化的，接受坏的首次报价 | 战略性的，基于 Rubinstein/Zeuthen |
| 语言 | 差——策略错误 | 好——LLM 做它擅长的事：写作 |

### 2.3 思维链隐藏

大规模自主谈判竞赛（arXiv:2503.06416）运行了约 18 万次谈判。赢家向对手隐藏推理：

- 如果智能体打印"我只会接受到 $75；我的保留价格是 $70"到公开可见的草稿本，对手会读到它
- 赢家私下计算策略；输出通道只包含报价和最少必要叙述

工程要点：**分离私有草稿本上下文与公开消息上下文。** 不是可选的。

### 2.4 合同网 + LLM

```
管理者分解任务 → 广播 cfp → 投标者投标 (价格, ETA, 置信度)
管理者选择赢家 → 颁奖 → 被拒绝者自由投标其他任务
```

### 2.5 叙述 vs 机制规则

> 让 LLM 叙述。不要让 LLM 计算报价。

如果报价需要是数字（价格、ETA、数量），从谈判状态确定性地生成它，让 LLM 产生框架。如果报价需要是提案结构（任务分解、角色分配），让 LLM 起草，但对 Schema 和约束检查进行验证再发送。

---

## 3. 从零实现

### 第 1 步：定义谈判状态和策略

```python
@dataclass
class BargainState:
    buyer_max: int
    seller_min: int
    buyer_offer: int | None = None
    seller_offer: int | None = None
    rounds: int = 0

def naive_llm_bargain(state, rng):
    """朴素 LLM——方差高，经常在 ZOPA 外。"""
    if state.seller_offer is None:
        return rng.randint(state.buyer_max - 60, state.buyer_max + 30)
    r = rng.random()
    if r < 0.35:
        return state.seller_offer + rng.randint(-8, 3)
    elif r < 0.65:
        return rng.randint(state.seller_min - 30, state.buyer_max + 30)
    else:
        return rng.randint(state.seller_min - 60, state.buyer_max + 60)

def og_narrator_bargain(state, rng, concession=0.35):
    """OG-Narrator——确定性 Zeuthen 风格让步。"""
    if state.seller_offer is None and state.buyer_offer is None:
        return state.buyer_max - max(1, int((state.buyer_max - state.seller_min) * 0.2))
    if state.seller_offer is None:
        return state.buyer_offer
    prior = state.buyer_offer if state.buyer_offer is not None else state.buyer_max
    move = max(1, int(concession * (state.seller_offer - prior)))
    return min(prior + move, state.buyer_max)
```

### 第 2 步：实现合同网

```python
@dataclass
class Bid:
    bidder: str
    price: int
    eta_minutes: int
    confidence: float

@dataclass
class ContractNetTask:
    task_id: str
    description: str
    deadline_minutes: int
    budget: int

class ContractNetManager:
    def __init__(self, bidders):
        self.bidders = bidders
        self.proposals = {}

    def broadcast_cfp(self, task):
        self.proposals[task.task_id] = []

    def receive_proposal(self, task_id, bid):
        self.proposals[task_id].append(bid)

    def award(self, task):
        props = self.proposals.get(task.task_id, [])
        feasible = [b for b in props if b.price <= task.budget and b.eta_minutes <= task.deadline_minutes]
        if not feasible:
            return None
        winner = max(feasible, key=lambda b: b.confidence / max(b.price, 1))
        return winner
```

### 第 3 步：运行基准

```python
def bench_deal_rate(buyer_fn, label, trials=1000):
    rng = random.Random(42)
    deals = 0
    for _ in range(trials):
        seller_min = rng.randint(50, 80)
        buyer_max = rng.randint(max(seller_min + 5, 75), 115)
        if simulate_bargain(buyer_fn, rng, buyer_max=buyer_max, seller_min=seller_min):
            deals += 1
    print(f"  {label:20s} deal rate: {deals / trials:.2%}")

def main():
    bench_deal_rate(naive_llm_bargain, "naive LLM")
    bench_deal_rate(og_narrator_bargain, "OG-Narrator")
    demo_contract_net()
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 谈判架构模式

| 模式 | 机制 | 优势 |
|------|------|------|
| OG-Narrator | 确定性报价 + LLM 叙述 | 成交率从 27% 提高到 89% |
| 合同网 | cfp → propose → award | 扩展到 100+ 工作者 |
| 多方可评分博弈 | 秘密分数 + 最低接受阈值 | N 方联盟形成 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 私有草稿本与公开消息分离 | 不是可选的——赢家隐藏推理 |
| 确定性报价生成 | 价格、数量、ETA：计算，不提示 |
| 所有传入报价验证 Schema | 拒绝超出 ZOPA 的报价 |
| 限制轮数（3-5 轮） | 超出后升级到调解员 |

---

## 6. 常见错误

### 错误 1：让 LLM 计算报价

**现象：** LLM 在 ZOPA 之外报价，或锚点情绪化而非战略性。

**修复：** "让 LLM 叙述，不要让 LLM 计算报价。"

### 错误 2：不隐藏推理

**现象：** 智能体在公开草稿本中打印保留价格，对手读到后利用。

**修复：** 私有草稿本上下文与公开消息上下文分离。

### 错误 3：无界辩论轮次

**现象：** 无限制谈判，每轮增加词元成本但不增加成交率。

**修复：** 限制轮数（3-5 轮）。超出后升级到调解员。

---

## 7. 面试考点

### Q1：为什么 OG-Narrator 比朴素 LLM 谈判更好？（难度：⭐）

**参考答案：**
朴素 LLM 混淆了"决定报价"和"叙述报价"——它在报价时用生成式方法（方差高、经常超出 ZOPA）。

OG-Narrator 将两者分离：确定性报价生成器（Rubinstein/Zeuthen 风格）计算数字，LLM 只叙述。报价保持在谈判区间内，锚点是战略性的而非情绪性的。

### Q2：思维链隐藏在谈判中为什么重要？（难度：⭐⭐）

**参考答案：**
如果智能体打印保留价格到公开草稿本，对手读到后利用。赢家在私有草稿本中计算策略，公开通道只包含报价和最少叙述。

这是经典博弈论的重述（Aumann 1976）：揭示你的私人估值会损失收益。LLM 不直觉这一点，会在推理追踪中输入保留价格。

### Q3：合同网如何扩展到 100+ 工作者？（难度：⭐⭐⭐）

**参考答案：**
广播-响应模式——管理者广播 cfp，工作者回复报价，管理者颁奖。没有同步聊天——异步的。

扩展性来自协议本身：广播和响应是异步的，不需要智能体互相等待。每个工作者独立响应，管理者独立颁奖。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 合同网 | "任务市场" | Smith 1980、FIPA 1996。cfp + propose + accept/reject |
| ZOPA | "可能协议区间" | 买方最高和卖方最低之间的重叠 |
| BATNA | "谈判协议最佳替代方案" | 这次谈判失败时的后备。设定你的保留价格 |
| OG-Narrator | "报价生成器 + 叙述者" | 分解：确定性报价 + LLM 叙述 |
| 思维链隐藏 | "隐藏你的推理" | 赢家保持私有草稿本；公开通道只显示报价 |
| 合同网 | "FIPA 祖先" | Smith 1980、FIPA 1996。任务市场的经典机制 |

---

## 📚 小结

OG-Narrator 将报价生成和叙述分离，将成交率从 27% 提高到 89%。思维链隐藏是赢家策略——私有草稿本保留推理，公开通道只显示报价。合同网（cfp → propose → award）扩展到 100+ 工作者。关键原则：让 LLM 叙述，不要让 LLM 计算报价。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认 OG-Narrator 在成交率上击败朴素 LLM。差多少？

2. **【实现】** 添加角色操纵（NegotiationArena）——买方在叙述中采用"本周绝望"角色，报价生成器不变。成交率或收益是否变化？

3. **【实现】** 实现思维链隐藏：维护不传给对手的私有草稿本字符串。如果意外泄露会发生什么？

4. **【阅读】** 阅读 Bhattacharya 等人 2025 年关于哈佛谈判项目指标的研究。实现两种不同风格的议价者（攻击性 vs 公平）。测量对称和不对称配对下的收益方差。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 谈判演示 | `code/main.py` | 合同网 + OG-Narrator vs 朴素 LLM |
| 技能提示词 | `outputs/skill-bargainer-designer.md` | 为新任务设计议价协议 |

---

## 📖 参考资料

1. [论文] NegotiationArena. https://arxiv.org/abs/2402.05863 — 基准；角色操纵和利用发现
2. [论文] "Measuring Bargaining Abilities". https://arxiv.org/abs/2402.15813 — OG-Narrator 和买方比卖方更难
3. [论文] Large-Scale Autonomous Negotiation Competition. https://arxiv.org/abs/2503.06416 — ~18 万次谈判；思维链隐藏获胜
4. [论文] Smith 1980. https://ieeexplore.ieee.org/document/1675516 — 经典合同网协议

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
