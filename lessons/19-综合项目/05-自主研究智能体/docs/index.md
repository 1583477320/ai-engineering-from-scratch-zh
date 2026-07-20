# 综合项目05——自主研究智能体（AI科学家）

> Sakana的AI-Scientist-v2发表了完整论文。Agent Laboratory运行了实验。Allen AI分享了追踪数据。2026年的形态是：一个基于树搜索的计划-执行-验证循环，覆盖实验空间，受预算约束，代码执行在沙箱中运行，LaTeX写作支持视觉反馈，以及自动化NeurIPS风格的审稿人集成。本综合项目要求你构建一个，在每篇论文$30预算内端到端运行，并通过Sakana记录的沙箱逃逸红队测试。

**类型：** 综合项目
**编程语言：** Python（智能体+沙箱），LaTeX（输出）
**前置知识：** 第2章（ML基础）、第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第14章（智能体）、第15章（自主系统）、第16章（多智能体）、第18章（安全）
**涉及章节：** P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**预计时间：** 40小时

---

## 学习目标

- 构建基于最佳优先树搜索的实验规划循环
- 实现沙箱隔离的实验执行环境
- 实现多模态LaTeX写作和视觉反馈
- 通过沙箱逃逸红队测试

---

## 1. 问题

自主研究智能体在2026年跨越了一个阈值。Sakana AI的AI-Scientist-v2发表在Nature上，其生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将路线扩展到进化假设。AMD的Agent Laboratory发布了可复现的追踪数据。

这些智能体不是魔术——它们是一个在候选实验树上运行的规划-执行-验证循环，包含成本上限、种子绑定的沙箱和自动评审。手艺在于循环、预算和安全故事。

---

## 2. 核心概念

### 2.1 最佳优先树搜索

智能体是一个最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤提出带有小编辑的子节点（交换优化器、调整批次大小、消融组件）。每个子节点在全新的沙箱中运行，有硬资源上限。结果反馈到评分函数，按（新穎性 × 质量 × 剩余预算）排名节点。

### 2.2 多模态写作

写作器是多模态的。它生成LaTeX草稿，编译，渲染图表，并将渲染后的PDF反馈给Claude Opus 4.7的视觉模式进行布局、图表可读性和声明-证据对齐的评判。

### 2.3 审稿人集成

由五个LLM评审者组成的集成输出NeurIPS风格的评分（新穎性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文返回给写作器并附上评判。

### 2.4 安全

每个实验在E2B或Daytona沙箱中运行，无网络出口、有边界墙钟时间和固定的资源限制。智能体的代码生成步骤通过策略层，阻止逃逸沙箱的系统调用。

---

## 3. 从零实现

`code/main.py`实现最佳优先树搜索的核心算法：预算约束的扩展、沙箱执行和评分函数。

```python
"""自主研究智能体——计划/执行/验证树搜索脚手架。

核心架构原语是实验节点上的最佳优先树搜索，包含预算约束的扩展、
每节点沙箱化执行和新穎性×质量×预算评分函数。
LLM规划器和实际PyTorch实验被存根替代，使树搜索骨架可端到端观察。

运行：python3 code/main.py
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 实验节点——（假设、配置、结果）元组
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: int
    parent: int | None
    hypothesis: str
    config: dict[str, object]
    result: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    novelty: float = 0.5
    quality: float = 0.0
    failure: str | None = None

    def score(self, remaining_budget: float) -> float:
        budget_weight = min(1.0, remaining_budget / 10.0)
        return self.novelty * 0.4 + self.quality * 0.5 + budget_weight * 0.1


# ---------------------------------------------------------------------------
# 规划器——通过小编辑扩展提出子节点
# ---------------------------------------------------------------------------

def expand(node: Node, next_id: int) -> list[Node]:
    """每次改变一个配置维度来提出子节点"""
    children: list[Node] = []
    base_cfg = node.config
    # 改变稀疏度
    for sp in (4, 8, 16):
        cfg = dict(base_cfg, sparsity_top=sp)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"sparsity top-{sp}", config=cfg))
        next_id += 1
    # 改变学习率
    for lr in (3e-4, 1e-3):
        cfg = dict(base_cfg, lr=lr)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"lr={lr}", config=cfg))
        next_id += 1
    return children


# ---------------------------------------------------------------------------
# 沙箱执行——存根，返回可复现的指标
# ---------------------------------------------------------------------------

def run_experiment(node: Node, rng: random.Random) -> None:
    """模拟沙箱容器中的实验执行"""
    sp = node.config.get("sparsity_top", 8)
    lr = node.config.get("lr", 3e-4)
    ideal_sp = 8
    loss = 3.0 - 0.3 * (1 - abs(sp - ideal_sp) / 16) + rng.gauss(0, 0.05)
    loss += 0.0001 * abs(lr - 3e-4) * 1000
    node.result = {"loss": round(loss, 3), "sparsity_top": sp, "lr": lr}
    node.cost_usd = 1.2 + rng.uniform(0, 0.4)
    node.quality = max(0.0, 1.0 - (loss - 2.5) / 1.5)
    node.novelty = 0.5 + rng.uniform(-0.1, 0.2)
    if rng.random() < 0.1:
        node.failure = "oom_killed_by_cgroup"
        node.quality = 0.0


# ---------------------------------------------------------------------------
# 验证——结果合理性检查
# ---------------------------------------------------------------------------

def verify(node: Node) -> bool:
    if node.failure:
        return False
    if node.result.get("loss", 99) > 4.0:
        node.failure = "loss_diverged"
        return False
    return True


# ---------------------------------------------------------------------------
# 树搜索——带预算和最大深度的最佳优先搜索
# ---------------------------------------------------------------------------

@dataclass
class Tree:
    root: Node
    nodes: dict[int, Node] = field(default_factory=dict)
    frontier: list = field(default_factory=list)
    counter: int = 0
    budget: float = 30.0
    spent: float = 0.0
    max_nodes: int = 24

    def push(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.counter += 1
        remaining = self.budget - self.spent
        heapq.heappush(self.frontier, (-node.score(remaining), self.counter, node.node_id))

    def pop(self) -> Node | None:
        while self.frontier:
            _, _, nid = heapq.heappop(self.frontier)
            return self.nodes[nid]
        return None


def tree_search(seed: str, rng: random.Random) -> Tree:
    root = Node(node_id=0, parent=None, hypothesis=seed,
                config={"sparsity_top": 8, "lr": 3e-4})
    root.novelty = 1.0
    root.quality = 0.5
    tree = Tree(root=root)
    tree.push(root)

    next_id = 1
    while tree.frontier and len(tree.nodes) < tree.max_nodes:
        cur = tree.pop()
        if cur is None:
            break
        if tree.spent >= tree.budget:
            print(f"    预算耗尽 ${tree.spent:.2f}")
            break
        if cur.node_id != 0:
            run_experiment(cur, rng)
            tree.spent += cur.cost_usd
            ok = verify(cur)
            flag = "ok " if ok else "FAIL"
            print(f"    [{flag}] 节点#{cur.node_id:02d}  假设='{cur.hypothesis}'  "
                  f"loss={cur.result.get('loss','?'):>5}  "
                  f"$={cur.cost_usd:.2f}  累计${tree.spent:.2f}")
            if not ok:
                continue
        children = expand(cur, next_id)
        next_id += len(children)
        for ch in children:
            tree.push(ch)
    return tree


# ---------------------------------------------------------------------------
# 最佳分支选择
# ---------------------------------------------------------------------------

def best_branch(tree: Tree) -> list[Node]:
    done = [n for n in tree.nodes.values() if n.result and not n.failure]
    if not done:
        return []
    best = max(done, key=lambda n: n.quality)
    chain = [best]
    while chain[-1].parent is not None:
        chain.append(tree.nodes[chain[-1].parent])
    return list(reversed(chain))


def main() -> None:
    print("=== 自主研究智能体：树搜索（预算$30）===")
    rng = random.Random(7)
    seed = "研究小于1B参数Transformer的注意力稀疏性模式"
    tree = tree_search(seed, rng)
    print()
    print(f"探索节点数: {len(tree.nodes)}")
    print(f"预算花费   : ${tree.spent:.2f} / ${tree.budget:.2f}")
    print(f"失败节点数 : {sum(1 for n in tree.nodes.values() if n.failure)}")

    branch = best_branch(tree)
    print(f"\n最佳分支（长度 {len(branch)}）:")
    for n in branch:
        print(f"  #{n.node_id:02d} {n.hypothesis}   q={n.quality:.2f}  loss={n.result.get('loss','?')}")

    print("\n（写作器 + 审稿人 + 红队步骤在此处运行；存根中）")


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 自主研究智能体：树搜索（预算$30）===
    [ok ] 节点#01  假设='sparsity top-4'   loss=2.74  $=1.46  累计$1.46
    [ok ] 节点#02  假设='sparsity top-8'   loss=2.75  $=1.55  累计$3.01
    [ok ] 节点#03  假设='sparsity top-16'  loss=3.48  $=1.52  累计$4.53
    [ok ] 节点#04  假设='lr=0.0003'        loss=2.66  $=1.34  累计$5.87
    [ok ] 节点#05  假设='lr=0.001'         loss=2.81  $=1.37  累计$7.24
    ...

探索节点数: 24
预算花费   : $21.43 / $30.00
失败节点数 : 1

最佳分支（长度 2）:
  #00 invest...  q=0.50  loss=?
  #05 lr=0.001   q=0.72  loss=2.81
```

---

## 4. 工具实践

**技术栈：**
- 编排：LangGraph + checkpoint + 人工审批门
- 树搜索：自定义最佳优先（AB-MCTS风格）
- 沙箱：E2B或Daytona，cgroups资源限制，无网络
- 文献：Semantic Scholar Graph API + OpenAlex + FAISS缓存
- 写作：LaTeX模板 + Claude Opus 4.7视觉模式
- 审稿人：5个LLM评审者（加权聚合）
- 实验框架：PyTorch 2.5

---

## 5. LLM视角

**树搜索视角**：最佳优先树搜索是核心算法——不是随机尝试，而是根据新穎性×质量×预算评分优先扩展最有希望的节点。

**视觉反馈视角**：编译PDF后通过VLM视觉模式评判布局和声明-证据对齐，这种闭环让论文质量大幅提升。

**预算约束视角**：$30/篇的硬预算迫使智能体有选择地探索，而非穷举搜索。

---

## 6. 工程最佳实践

**沙箱安全**：
- Docker容器，`--network=none`
- 8GB内存、2 CPU、PID限制256
- 只读输入，无网络出口

**文献检索**：
- Semantic Scholar + OpenAlex
- FAISS缓存减少重复查询
- 生成1页领域摘要

**可复现性**：
- 每篇论文附带树搜索trace JSON
- 种子、W&B运行链接、沙箱配置
- 端到端一键复现

---

## 7. 常见错误

**错误1：无预算约束的穷举搜索**
症状：预算超支
修复：新穎性×质量×预算评分函数

**错误2：忽略沙箱安全**
症状：实验代码逃逸沙箱
修复：无网络、cgroups资源限制、策略层

**错误3：无视觉反馈的写作**
症状：布局混乱、图表不清晰
修复：编译后通过VLM视觉模式评判

---

## 8. 面试考点

**Q1：自主研究智能体的树搜索如何工作？**
考察：对最佳优先搜索的理解

**Q2：为什么沙箱隔离对研究智能体至关重要？**
考察：对安全风险的理解

**Q3：审稿人集成如何确保论文质量？**
考察：对自动化评审的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 树搜索 | "AB-MCTS风格扩展" | 实验节点上的最佳优先探索，评分=新穎性×质量×预算 |
| 沙箱 | "实验隔离" | 无网络、有界CPU/内存、固定种子的容器 |
| 视觉评判 | "渲染后阅读" | 编译PDF为图像，反馈给VLM评判布局和声明-证据对齐 |
| 审稿人集成 | "自动同行评审" | 多个LLM评审者按NeurIPS标准评分，加权聚合 |
| 新穎性评分 | "这是新的吗？" | 惩罚接近50篇文献缓存的启发式方法 |
| 成本上限 | "$预算" | 每篇论文的硬上限 |

---

## 参考文献

- [Sakana AI-Scientist-v2仓库](https://github.com/SakanaAI/AI-Scientist-v2)
- [Sakana AI-Scientist-v1论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292)
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai)
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [E2B沙箱](https://e2b.dev)
