# 智能体经济、代币激励与声誉

> 长期自主智能体需要经济机构。五层栈：DePIN（物理计算）→ 身份（W3C DID）→ 认知（RAG+MCP）→ 结算（账户抽象）→ 治理（智能体 DAO）。Bittensor、Fetch.ai 和 Gonka 是生产级的代币激励网络。本课构建一个最小智能体市场，应用沙普利值信用归因，运行次价代币拍卖。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 16（谈判与议价）、阶段 16 · 09（并行群体网络）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 学习目标

- 实现沙普利值信用归因——公平分配联合产出
- 运行次价拍卖——对单调聚合而言是真实诚实的
- 构建声誉加权路由——声誉随确认贡献累积
- 理解五层智能体经济栈

---

## 1. 问题

多智能体系统在智能体联合产生价值但需要单独获得奖励时变得复杂。均分或"最后贡献者全得"不公平或可博弈。沙普利值基于构造是公平的但计算出昂贵。次价拍卖是诚实的。声誉让路由决策更好。

---

## 2. 概念

### 2.1 沙普利值信用归因

三个智能体合作。输出得分 0.8。谁贡献了什么？沙普利值是唯一满足四个公理的信用分配。对智能体 i：在所有排序上平均 i 的边际贡献。

```
shapley(i) = (1/N!) * Σ over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

实践中采样排序而非穷举。

### 2.2 次价拍卖

N 个智能各出价。拍卖者选最高者，支付第二高价。这激励诚实出价。

### 2.3 声誉资本

经确认贡献累积的 DID 绑定分数：

```
rep(i, t+1) = alpha * rep(i, t) + (1-alpha) * quality(i, t)
```

声誉：便宜读取、难伪造、可削减。

---

## 3. 从零实现

### 第 1 步：沙普利值

```python
def shapley_exact(value_fn, agents):
    n = len(agents)
    contribs = {a: 0.0 for a in agents}
    for order in permutations(agents):
        visited = set()
        prev = value_fn(frozenset(visited))
        for a in order:
            visited.add(a)
            new = value_fn(frozenset(visited))
            contribs[a] += new - prev
            prev = new
    return {a: v / math.factorial(n) for a, v in contribs.items()}
```

### 第 2 步：次价拍卖

```python
def second_price(bids):
    sorted_bids = sorted(bids, key=lambda b: b.value, reverse=True)
    winner = sorted_bids[0].bidder
    payment = sorted_bids[1].value
    return winner, payment
```

### 第 3 步：声誉路由

```python
class Reputation:
    def __init__(self, alpha=0.95, floor=0.1):
        self.scores = {}
        self.alpha = alpha; self.floor = floor

    def update(self, agent, quality):
        current = self.scores.get(agent, 1.0)
        self.scores[agent] = max(self.floor, self.alpha * current + (1-self.alpha) * quality)
```

### 第 4 步：运行演示

```python
def main():
    demo_shapley()  # 三个智能体合作，沙普利值归因
    demo_auction()  # 五个智能体竞标
    demo_reputation_routing()  # 100 轮，50 预热
```

完整代码见 code/main.py。

---

## 4. 工业工具

| 网络 | 层 | 机制 |
|------|------|------|
| Bittensor TAO | DePIN + 身份 | 子网挖矿+验证者排名 |
| Fetch.ai / ASI | 认知 + 结算 | LLM 使用 + FET 代币 |
| Gonka | DePIN | 变压器 PoW |

---

## 5. 常见错误

### 错误 1：忽略评估成本

沙普利值对 N=10 需要 360 万次评估。采样 100-1000 次排序。

### 错误 2：未经验证就分配信用

自报告的质量累积 sybil 攻击。必须有独立验证步骤。

### 错误 3：无下限的声誉

无界衰减抹杀合法贡献者；太慢衰减奖励过时的参与者。

---

## 6. 面试考点

### Q1：沙普利值的四个公理是什么？（难度：⭐）

效率、对称性、线性、空值。沙普利值是唯一满足四个的分配。

### Q2：次价拍卖为什么诚实的？（难度：⭐⭐）

在单调聚合下，代理人的最优策略是出真实价格——不撒谎。

---

## 关键术语

| 术语 | 含义 |
|------|------|
| 沙普利值 | 公平信用归因，满足四个公理 |
| 次价拍卖 | 赢家付第二高价格，诚实 |
| 声誉资本 | DID 绑定分数，随时间衰减 |
| DePIN | 去中心化物理基础设施 |

---

## 小结

沙普利值公平归因联合产出；次价拍卖按真实出价选择赢家；声誉资本让路由决策更好。五层栈（DePIN→身份→认知→结算→治理）是参考地图。从声誉开始，不从代币开始。

---

## 练习

1. 运行 code/main.py。确认沙普利值总和等于总价值。
2. 实现沙普利值采样（蒙特卡洛）。K 如何影响近似准确率？
3. 实现拍卖前的联盟形成步骤：智能体合并为团队投标。

---

## 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 智能体经济演示 | code/main.py | 沙普利值 + 次价拍卖 + 声誉路由 |

---

## 参考资料

1. The Agent Economy. arXiv:2602.14219
2. Google Research. Mechanism Design for LLMs.
3. Bittensor. https://docs.bittensor.com/
4. W3C DID. https://www.w3.org/TR/did-core/
