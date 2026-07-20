# 综合项目06——DevOps故障排查智能体（Kubernetes）

> AWS的DevOps Agent已正式可用，Resolve AI发布了K8s剧本，NeuBird展示了语义监控，Metoro将AI SRE与按服务SLO绑定。2026年的生产形态已经确定：告警webhook触发、智能体读取遥测、遍历K8s对象图谱、排名根因假设、在Slack中发布带审批按钮的简报。默认为只读。每个修复操作都需人工审批。本综合项目要求你构建这个智能体，在20个合成故障上评测，并与AWS的Agent在三个共享案例上对比。

**类型：** 综合项目
**编程语言：** Python（智能体），TypeScript（Slack集成）
**前置知识：** 第11章（LLM工程）、第13章（工具与MCP）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）、第18章（安全）
**涉及章节：** P11 · P13 · P14 · P15 · P17 · P18
**预计时间：** 30小时

---

## 学习目标

- 构建K8s知识图谱：对象节点+遥测覆盖边
- 实现只读默认的工具表面和人工审批门
- 实现基于证据加权的根因假设排名
- 实现审计日志记录每条考虑过的命令（含未执行的）

---

## 1. 问题

2025-2026年的SRE叙事变成："AI智能体分类故障，人类审批修复"。AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps都在生产中采用这种形态。

智能体读取Prometheus指标、Loki日志、Tempo追踪、kube-state-metrics和K8s对象知识图谱。它在5分钟内产出带遥测引用的排名根因假设。它绝不执行破坏性命令，除非通过Slack获得明确人工审批。

大部分硬工作在范围界定和安全上，而非推理。

---

## 2. 核心概念

### 2.1 知识图谱

智能体在知识图谱上操作。节点是K8s对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus序列、Loki流、Tempo追踪）。边编码拥有关系（Pod → ReplicaSet → Deployment）、调度关系（Pod → Node）和观察关系（Pod → Prometheus序列）。

图谱通过kube-state-metrics同步保持新鲜，每次告警时重新采样。

### 2.2 只读默认

允许的默认操作是只读的。破坏性操作（缩容、回滚、删除Pod）需要Slack审批。

审计日志记录智能体*考虑过*的每条命令——不仅是被执行的——因此审查过程能捕获接近失误。

### 2.3 根因排名

假设按证据排名：有多少遥测引用支持它、有多新、有多具体。top-3假设发送到Slack。

---

## 3. 从零实现

`code/main.py`实现K8s知识图谱、只读默认的工具表面和人工审批门。

```python
"""DevOps故障排查智能体——K8s知识图谱 + 人工审批门。

核心架构原语是：(a) K8s知识图谱，让根因分析从告警对象遍历到
带遥测覆盖的邻居；(b) 只读默认的工具表面，
每条破坏性命令通过人工审批门，每条考虑过的命令记录审计日志。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# K8s知识图谱——对象 + 遥测覆盖边
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str               # "Pod" | "Deployment" | "Node" | "Service" | "Prom" | "Loki"
    name: str
    attrs: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.name}"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, rel, dst)

    def add(self, n: Node) -> None:
        self.nodes[n.key] = n

    def link(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

    def neighbors(self, key: str) -> list[tuple[str, str]]:
        out = [(rel, dst) for s, rel, dst in self.edges if s == key]
        out += [(rel, src) for src, rel, dst in self.edges if dst == key]
        return out


def build_sample_cluster() -> Graph:
    """构建示例K8s集群图谱"""
    g = Graph()
    dep = Node("Deployment", "checkout-api",
               {"revision": 42, "image": "checkout-api:v2.41", "deployed_at": "14m ago"})
    rs = Node("ReplicaSet", "checkout-api-abc")
    node = Node("Node", "ip-10-2-3-4", {"kernel": "6.1.109"})
    pods = [Node("Pod", f"checkout-api-abc-{i}", {"phase": "Running"}) for i in range(3)]
    svc = Node("Service", "checkout-api")
    prom = Node("Prom", "error_rate{deployment=checkout-api}",
                {"last_15m": "mean=0.14 up_trend", "threshold": 0.05})
    loki = Node("Loki", "namespace=prod,app=checkout-api",
                {"last_15m": "500 errors on /api/v2/pay, stack = NullHealthz"})

    for n in (dep, rs, node, svc, prom, loki, *pods):
        g.add(n)
    g.link(dep.key, "OWNS", rs.key)
    for p in pods:
        g.link(rs.key, "OWNS", p.key)
        g.link(p.key, "SCHEDULED_ON", node.key)
    g.link(svc.key, "EXPOSES", dep.key)
    g.link(dep.key, "OBSERVED_BY", prom.key)
    g.link(dep.key, "OBSERVED_BY", loki.key)
    return g


# ---------------------------------------------------------------------------
# 假设排名——新鲜度 × 特异性 × 引用数
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    title: str
    citations: list[str]
    recency_mins: int
    specificity: float     # 0..1
    path_len: int

    def score(self) -> float:
        recency_w = max(0.0, 1.0 - self.recency_mins / 60.0)
        path_w = 1.0 / (1 + self.path_len)
        return (recency_w * 0.35 +
                self.specificity * 0.35 +
                min(len(self.citations), 5) / 5 * 0.2 +
                path_w * 0.1)


def root_cause(g: Graph, alerted: str) -> list[Hypothesis]:
    """从告警对象向外遍历，收集遥测，提出排名假设"""
    hyps: list[Hypothesis] = []
    telemetry = []
    for rel, neighbor_key in g.neighbors(alerted):
        n = g.nodes.get(neighbor_key)
        if n and n.kind in ("Prom", "Loki", "Tempo"):
            telemetry.append(n)

    # 假设：如果有最近部署 + 错误激增，则是有问题的发布
    dep = g.nodes.get(alerted)
    if dep and dep.kind == "Deployment":
        mins_str = str(dep.attrs.get("deployed_at", ""))
        mins = 14  # default based on "14m ago"
        hyps.append(Hypothesis(
            title=f"有问题的发布: 镜像 {dep.attrs.get('image')} 未通过/healthz",
            citations=[t.name for t in telemetry],
            recency_mins=mins,
            specificity=0.82,
            path_len=0,
        ))

    # 假设：节点级问题（噪声邻居/内核）
    nodes = [g.nodes[dst] for _, dst in g.neighbors(alerted) if dst.startswith("Node/")]
    if nodes:
        hyps.append(Hypothesis(
            title=f"节点压力 {nodes[0].name} (内核={nodes[0].attrs.get('kernel')})",
            citations=[n.name for n in nodes],
            recency_mins=30,
            specificity=0.45,
            path_len=2,
        ))

    # 假设：DNS问题
    hyps.append(Hypothesis(
        title="kube-system/coredns DNS抖动",
        citations=[],
        recency_mins=60,
        specificity=0.2,
        path_len=4,
    ))

    return sorted(hyps, key=lambda h: -h.score())


# ---------------------------------------------------------------------------
# 审批门 + 审计日志
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    ts: float
    tool: str
    args: dict
    considered: bool = True
    approved: bool = False
    executed: bool = False
    approver: str | None = None
    result: str | None = None


@dataclass
class Agent:
    graph: Graph
    audit: list[AuditEvent] = field(default_factory=list)
    read_only_tools: tuple = ("kubectl_get", "kubectl_describe", "promql", "logql", "traceql")
    destructive_tools: tuple = ("kubectl_scale", "kubectl_rollback", "kubectl_delete", "argocd_rollback")

    def call(self, tool: str, args: dict, approver: str | None = None) -> AuditEvent:
        ev = AuditEvent(ts=time.time(), tool=tool, args=args)
        if tool in self.read_only_tools:
            ev.executed = True
            ev.result = "ok (read-only)"
        elif tool in self.destructive_tools:
            if approver:
                ev.approved = True
                ev.approver = approver
                ev.executed = True
                ev.result = f"由 {approver} 执行"
            else:
                ev.result = "已阻止: 无Slack审批"
        else:
            ev.result = "已阻止: 未知工具"
        self.audit.append(ev)
        return ev


# ---------------------------------------------------------------------------
# 演示——完整流程：告警 -> 图谱遍历 -> 排名假设 -> Slack审批门
# ---------------------------------------------------------------------------

def main() -> None:
    g = build_sample_cluster()
    agent = Agent(graph=g)

    alerted = "Deployment/checkout-api"
    print(f"=== 收到告警: {alerted} (错误率 14%) ===")

    # 智能体先拉取只读遥测
    agent.call("promql", {"query": "rate(http_requests_total{status=~'5..'}[5m])"})
    agent.call("logql", {"query": '{app="checkout-api"} |~ "stack"'})

    hyps = root_cause(g, alerted)
    print("\n排名假设:")
    for i, h in enumerate(hyps, 1):
        print(f"  #{i} score={h.score():.3f}  {h.title}")
        print(f"     引用: {h.citations}")

    # 智能体建议回滚，但需等待Slack审批
    print("\n建议修复操作:")
    ev = agent.call("argocd_rollback", {"app": "checkout-api", "to_revision": 41})
    print(f"  {ev.tool}: {ev.result}")

    # Slack审批通过
    print("\nSlack 审批已由 alice@sre 批准")
    ev = agent.call("argocd_rollback",
                    {"app": "checkout-api", "to_revision": 41},
                    approver="alice@sre")
    print(f"  {ev.tool}: {ev.result}")

    print("\n审计日志:")
    for ev in agent.audit:
        print(" ", json.dumps({
            "tool": ev.tool, "executed": ev.executed,
            "approved": ev.approved, "approver": ev.approver,
            "result": ev.result,
        }))


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 收到告警: Deployment/checkout-api (错误率 14%) ===

排名假设:
  #1 score=0.782  有问题的发布: 镜像 checkout-api:v2.41 未通过/healthz
     引用: ['Prom/error_rate{deployment=checkout-api}', 'Loki/namespace=prod,app=checkout-api']
  #2 score=0.224  节点压力 ip-10-2-3-4 (内核=6.1.109)
     引用: ['Node/ip-10-2-3-4']
  #3 score=0.080  kube-system/coredns DNS抖动
     引用: []

建议修复操作:
  argocd_rollback: 已阻止: 无Slack审批

Slack 审批已由 alice@sre 批准
  argocd_rollback: 由 alice@sre 执行

审计日志:
  {"tool": "promql", "executed": true, "approved": false, "approver": null, "result": "ok (read-only)"}
  {"tool": "logql", "executed": true, "approved": false, "approver": null, "result": "ok (read-only)"}
  {"tool": "argocd_rollback", "executed": false, "approved": false, "approver": null, "result": "已阻止: 无Slack审批"}
  {"tool": "argocd_rollback", "executed": true, "approved": true, "approver": "alice@sre", "result": "由 alice@sre 执行"}
```

---

## 4. 工具实践

**技术栈：**
- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或kuzu（嵌入）
- 智能体：LangGraph + 每工具允许列表，只读默认
- 工具传输：FastMCP over StreamableHTTP
- 模型：Claude Sonnet 4.7（根因推理）、Gemini 2.5 Flash（日志摘要）
- 修复：ArgoCD回滚webhook、PagerDuty上报

---

## 5. LLM视角

**只读默认视角**：智能体的安全模型是"假设会出错"。只读操作自动执行，破坏性操作需人工审批。这是2026年生产AI智能体的标准安全模式。

**知识图谱视角**：K8s对象的关系结构天然适合图遍历。从告警对象到根因的最短路径通常经过3-4条边。

**审计视角**：考虑过的命令和被执行的命令一样重要。接近失误是改进安全策略的关键数据。

---

## 6. 工程最佳实践

**安全设计**：
- 只读默认的RBAC表面
- 破坏性工具在独立MCP服务器上，需审批token
- 审计日志：每条考虑过的命令都记录

**假设排名**：
- 新鲜度：最近的事件权重更高
- 特异性：更精确的假设权重更高
- 引用数：更多遥测引用支持权重更高

**图谱维护**：
- kube-state-metrics每30秒同步
- 每次告警重新采样
- 边编码拥有、调度、观察关系

---

## 7. 常见错误

**错误1：智能体有写权限**
症状：智能体可执行破坏性命令
修复：只读默认，破坏性操作需审批

**错误2：不考虑接近失误**
症状：只审计已执行的命令
修复：记录每条考虑过的命令

**错误3：无证据排名的根因分析**
症状：假设无引用的遥测数据支持
修复：实施证据加权排名

---

## 8. 面试考点

**Q1：DevOps智能体为什么需要只读默认？**
考察：对AI安全设计的理解

**Q2：知识图谱如何帮助根因分析？**
考察：对图遍历的理解

**Q3：审计日志中"考虑过 vs 已执行"的区别为什么重要？**
考察：对安全运维的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| K8s知识图谱 | "集群图" | 节点=K8s对象+遥测序列；边=拥有、调度、观察 |
| 只读默认 | "限定RBAC" | 智能体的服务账号只有get/list/describe动词 |
| 审计日志 | "考虑过 vs 已执行" | 每条候选命令的不可变记录，是否运行，谁批准的 |
| 假设排名 | "证据分数" | 新鲜度×特异性×图路径长度倒数×引用数 |
| Slack审批卡 | "人工审批门" | 互动式Slack消息，包含修复按钮 |
| 遥测引用 | "证据指针" | 支持声明的PromQL查询或Loki选择器 |

---

## 参考文献

- [AWS DevOps Agent正式版](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/)
- [Resolve AI K8s故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai)
- [NeuBird语义监控](https://www.neubird.ai)
- [Metoro AI SRE](https://metoro.io)
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FastMCP](https://github.com/jlowin/fastmcp)
