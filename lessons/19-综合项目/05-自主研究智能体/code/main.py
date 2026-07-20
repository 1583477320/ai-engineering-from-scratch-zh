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


def expand(node: Node, next_id: int) -> list[Node]:
    children: list[Node] = []
    base_cfg = node.config
    for sp in (4, 8, 16):
        cfg = dict(base_cfg, sparsity_top=sp)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"sparsity top-{sp}", config=cfg))
        next_id += 1
    for lr in (3e-4, 1e-3):
        cfg = dict(base_cfg, lr=lr)
        children.append(Node(node_id=next_id, parent=node.node_id,
                             hypothesis=f"lr={lr}", config=cfg))
        next_id += 1
    return children


def run_experiment(node: Node, rng: random.Random) -> None:
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


def verify(node: Node) -> bool:
    if node.failure:
        return False
    if node.result.get("loss", 99) > 4.0:
        node.failure = "loss_diverged"
        return False
    return True


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
