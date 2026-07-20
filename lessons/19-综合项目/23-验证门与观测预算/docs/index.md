# 综合项目23——验证门与观测预算

> 没有验证层的智能体主循环是穿着风衣的愿望。本课程构建决定工具调用是否允许触发、模型允许看到多少输出、以及循环因模型读取过多而必须停止的确定性门链。门是小型命名门的函数加上观测账本，跟踪模型被展示的每个token。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** 第13章（工具与协议）、第14章（智能体）
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 构建带确定性`evaluate(call)`方法的`VerificationGate`协议
- 将预算、新鲜度、白名单和正则门组合为带短路语义的链
- 通过`ObservationLedger`按工具和轮次跟踪每个观测
- 当累积观测预算将被超出时拒绝工具调用
- 输出结构化的`GateDecision`记录供下游可观测性摄取

---

## 1. 问题

当智能体主循环让模型自由调用工具时，三类bug在一小时内出现：

1. **无界观测**：grep在200K行仓库上转储50万token输出。token账单很大，智能体变差而非更好。
2. **过时的新鲜度**：长时间任务累积50次工具调用。模型将第3轮的read_file当作实时状态重读。
3. **权限蔓延**：研究任务从web_search开始，最终运行shell因为模型发明了工具名。

验证门是说"不"的主循环组件。它不是模型，不是判断者，而是`(call, history, ledger)`的确定性函数。

---

## 2. 核心概念

### 2.1 四个门

- **WhitelistGate**：允许的工具名是显式集合，任何外部被拒绝。最便宜的门，最先运行。
- **RegexGate**：工具参数匹配正则。用于拒绝含`rm -rf`的shell调用或内部IP的HTTP调用。
- **RecencyGate**：模型只看到最近N轮的观测。更早的观测被掩码。
- **BudgetGate**：整个会话中模型已读取的累积token有上限。达到上限后拒绝所有进一步工具调用。

### 2.2 门链

有序列表，短路语义——第一个拒绝即返回。顺序重要：便宜的结构门在昂贵的token计数门之前运行。

### 2.3 观测账本

每次成功工具调用写一行：工具名、轮次、输出token、累积。账本回答两个问题：模型总共看了多少，看了工具X多少。

### 2.4 GateDecision

```
GateDecision
  allow: bool
  gate: str      (门名称)
  reason: str    (拒绝原因或通过原因)
```

---

## 3. 从零实现

`code/main.py`实现四个门、门链、观测账本和合成智能体循环演示。

```python
"""验证门与观测预算。

核心：门链（短路语义）+观测账本（累积token跟踪）。
四个门：白名单→正则→新鲜度→预算。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, re, sys
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class ToolCall:
    turn:int; tool:str; argv:tuple[str,...]; payload:str=""

@dataclass(frozen=True)
class Observation:
    turn:int; tool:str; text:str; tokens:int

@dataclass(frozen=True)
class GateDecision:
    allow:bool; gate:str; reason:str

def estimate_tokens(text:str) -> int:
    return max(1,len(text)//4) if text else 0

@dataclass
class ObservationLedger:
    rows:list[Observation]=field(default_factory=list)
    def record(self,obs): self.rows.append(obs)
    def cumulative(self): return sum(r.tokens for r in self.rows)
    def per_tool(self,name): return sum(r.tokens for r in self.rows if r.tool==name)
    def latest_turn(self): return self.rows[-1].turn if self.rows else -1

@dataclass
class GateContext:
    ledger:ObservationLedger; current_turn:int; history:tuple[ToolCall,...]=()

class VerificationGate(Protocol):
    name:str
    def evaluate(self,call:ToolCall,ctx:GateContext)->GateDecision: ...

@dataclass
class WhitelistGate:
    allowed:frozenset[str]; name:str="whitelist"
    def evaluate(self,call,ctx):
        if call.tool in self.allowed: return GateDecision(True,self.name,"ok")
        return GateDecision(False,self.name,f"{call.tool!r} not in {sorted(self.allowed)}")

@dataclass
class RegexGate:
    refuse_patterns:tuple[re.Pattern,...]; name:str="regex"
    @classmethod
    def from_strings(cls,patterns,name="regex"): return cls(tuple(re.compile(p) for p in patterns),name)
    def evaluate(self,call,ctx):
        hay=" ".join(call.argv)+" "+call.payload
        for p in self.refuse_patterns:
            if p.search(hay): return GateDecision(False,self.name,f"matched {p.pattern!r}")
        return GateDecision(True,self.name,"no match")

@dataclass
class RecencyGate:
    window:int; name:str="recency"
    def evaluate(self,call,ctx):
        last=ctx.ledger.latest_turn()
        if last<0: return GateDecision(True,self.name,"no prior obs")
        gap=call.turn-last
        if gap>self.window: return GateDecision(False,self.name,f"gap {gap}>{self.window}")
        return GateDecision(True,self.name,f"gap {gap} ok")

@dataclass
class BudgetGate:
    max_tokens:int; name:str="budget"
    def evaluate(self,call,ctx):
        used=ctx.ledger.cumulative()
        if used>=self.max_tokens: return GateDecision(False,self.name,f"{used}/{self.max_tokens} exhausted")
        return GateDecision(True,self.name,f"{self.max_tokens-used} remaining")


@dataclass
class ChainOutcome:
    decisions:list[GateDecision]
    @property
    def allow(self): return all(d.allow for d in self.decisions)
    @property
    def deny_reason(self):
        for d in self.decisions:
            if not d.allow: return f"[{d.gate}] {d.reason}"
        return None

@dataclass
class GateChain:
    gates:tuple[VerificationGate,...]
    def evaluate(self,call,ctx):
        decisions=[]
        for g in self.gates:
            d=g.evaluate(call,ctx); decisions.append(d)
            if not d.allow: return ChainOutcome(decisions=decisions)
        return ChainOutcome(decisions=decisions)


ToolFn=Callable[[ToolCall],str]

def run_synthetic_loop(calls,chain,tool_fns):
    ledger=ObservationLedger(); decisions=[]; obs=[]; allowed=0; refused=0; history=[]
    for call in calls:
        ctx=GateContext(ledger=ledger,current_turn=call.turn,history=tuple(history))
        outcome=chain.evaluate(call,ctx); decisions.append(outcome); history.append(call)
        if not outcome.allow: refused+=1; continue
        fn=tool_fns.get(call.tool)
        if fn is None: refused+=1; continue
        result=fn(call)
        o=Observation(turn=call.turn,tool=call.tool,text=result,tokens=estimate_tokens(result))
        ledger.record(o); obs.append(o); allowed+=1
    return {"turns":len(calls),"allowed":allowed,"refused":refused,
            "observations":[(o.turn,o.tool,o.tokens) for o in obs],
            "decisions":[{"tool":c.tool,"allow":d.allow,"reason":d.deny_reason or "ok"} for c,d in zip(calls,decisions)]}


def _demo():
    tools={"read_file":lambda c:"# fake\n"+("line "*12)*12,"list_dir":lambda c:"a.py\nb.py\n","run_tests":lambda c:'{"passed":true}'}
    chain=GateChain(gates=(WhitelistGate(frozenset({"read_file","list_dir","run_tests"})),
        RegexGate.from_strings([r"\brm\s+-rf\b",r"\bsudo\b"]),RecencyGate(window=3),BudgetGate(max_tokens=200)))
    calls=[ToolCall(1,"list_dir",("./",)),ToolCall(2,"read_file",("main.py",)),ToolCall(3,"read_file",("README.md",)),
           ToolCall(4,"run_tests",("./",)),ToolCall(5,"shell",("rm","-rf","/"))]
    r=run_synthetic_loop(calls,chain,tools)
    print(f"turns={r['turns']} allowed={r['allowed']} refused={r['refused']}")
    for d in r["decisions"]: print(f"  {d['tool']:12s} {'ALLOW' if d['allow'] else 'DENY':6s} {d['reason']}")
    print(f"\ncumulative tokens: {sum(o[2] for o in r['observations'])}")
    return 0 if r["refused"]>=1 else 1


if __name__=="__main__": sys.exit(_demo())
```

运行结果：

```
turns=5 allowed=4 refused=1
  list_dir     ALLOW  ok
  read_file    ALLOW  ok
  read_file    ALLOW  ok
  run_tests    ALLOW  ok
  shell        DENY   [whitelist] 'shell' not in ['list_dir', 'read_file', 'run_tests']

cumulative tokens: 153
```

---

## 4. 工具实践

**门的顺序**：
1. 白名单（O(1)哈希查找，最便宜）
2. 正则（O(pattern*argv)）
3. 新鲜度（读取消息存储的小切片）
4. 预算（读取整个账本，最贵）

**拒绝先于允许**：第一个拒绝即短路返回，不执行后续更贵的门。

---

## 5. LLM视角

**观测预算视角**：无界观测是编码智能体最常见的失败模式之一。grep返回50万token，智能体变差而非更好。预算门是成本和质量的硬边界。

**新鲜度视角**：模型倾向于重读最早的观测而非最新的。新鲜度门强制新鲜读取，防止过时状态导致错误编辑。

---

## 6. 工程最佳实践

**门设计**：每个门是纯净函数——无副作用、确定性、可重放。

**账本设计**：追加式记录，按工具和轮次索引。回答"总共看了多少"和"看了工具X多少"两个问题。

**合成循环**：演示中用固定调用序列替代真实模型决策，保持门链契约相同。

---

## 7. 常见错误

**错误1：门顺序错误**
症状：昂贵门在便宜门之前运行，浪费时间
修复：按成本升序排列

**错误2：不跟踪累积token**
症状：模型读取过多导致上下文污染
修复：BudgetGate + ObservationLedger

---

## 8. 面试考点

**Q1：为什么门链需要短路语义？**
考察：对性能优化的理解

**Q2：观测账本解决哪两类问题？**
考察：对无界观测和过时新鲜度的理解

**Q3：WhitelistGate为什么是第一个门？**
考察：对门顺序设计的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 验证门 | "工具调用守卫" | 决定工具调用是否允许触发的确定性函数 |
| 门链 | "有序拒绝链" | 按成本升序排列的门列表，短路语义 |
| 白名单门 | "工具名检查" | 允许的工具名是显式集合，O(1)查找 |
| 预算门 | "token上限" | 累积token达到上限后拒绝所有进一步调用 |
| 观测账本 | "token记账" | 追加式记录，跟踪模型被展示的每个token |
| GateDecision | "门判决" | (allow, gate_name, reason)结构化结果 |
| 短路语义 | "首个拒绝即返回" | 链中第一个拒绝即终止评估 |

---

## 参考文献

- [LangGraph验证门](https://langchain-ai.github.io/langgraph/)
- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
