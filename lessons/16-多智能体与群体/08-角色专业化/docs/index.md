# 角色专业化——规划者、批评者、执行者、验证者

> 2026 年最常见的多智能体分解：一个智能体规划，一个执行，一个批评或验证。MetaGPT 将其形式化为编码到角色提示中的 SOP——产品经理、架构师、项目经理、工程师、QA 工程师——遵循 `Code = SOP(Team)`。ChatDev 通过"通信去幻觉"（智能体明确请求缺失细节）链式连接设计者、程序员、审查者、测试者。验证者是负载承载的：Cemri 等人（MAST）显示每个多智能体失败都可以追溯到缺失或损坏的验证。PwC 报告 CrewAI 中结构化验证循环带来 7× 准确率提升（10% → 70%）。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 04（原语模型）、阶段 16 · 05（监督者）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分四个标准角色——规划者、执行者、批评者、验证者——以及各自的关键区别
- [ ] 理解 MetaGPT 的 SOP 模式（`Code = SOP(Team)`）和 ChatDev 的通信去幻觉
- [ ] 实现一个四角色流水线——并演示批评者可能漏过但验证者捕获的缺陷
- [ ] 识别"全 LLM 反模式"——没有确定性检查的系统是 MAST 失败模式
- [ ] 解释通信去幻觉如何防止执行者在信息缺失时编造细节

---

## 1. 问题

通用多智能体系统产生通用输出。三个群聊中的编码者写出三种平庸代码。你可以增加更多智能体、更多轮次，仍然达不到质量门槛。

修复不是更多智能体——而是**不同**的智能体。分配不同角色。给批评者规划者没有的工具。给验证者客观的测试套件。现在系统有了有根基修正的内部异议，而不只是并行猜测。

核心洞察：**不是智能体数量的问题，是智能体多样性的问题。** 验证者是负载承载的角色——Cemri 等人追踪的 1642 个失败中 21.3% 是验证缺口。PwC 报告在 CrewAI 中添加验证后准确率从 10% 提升到 70%。

---

## 2. 概念

### 2.1 四个标准角色

| 角色 | 做什么 | 工具 | 输出 | 类型 |
|------|--------|------|------|------|
| **规划者** | 读目标，产生步骤列表或规格 | 知识检索、文档 | 结构化计划 | LLM |
| **执行者** | 一次读一个计划步骤，产出工件 | 编译器、shell、API | 工件 | LLM |
| **批评者** | 将执行者输出与规划者意图对比 | 只读访问、静态分析 | 接受/拒绝+理由 | LLM |
| **验证者** | 读工件并运行确定性检查 | 测试运行器、类型检查器、Schema 验证器 | 通过/失败+证据 | **确定性代码** |

**关键区别：** 批评者是主观的、有观点的、通常基于 LLM。验证者是客观的、确定性的、通常基于代码。它们不是同一个角色。

### 2.2 MetaGPT 的 SOP 模式

MetaGPT（arXiv:2308.00352）将软件工程 SOP 编码为角色提示：

- **产品经理** 写 PRD
- **架构师** 产出系统设计
- **项目经理** 拆分任务
- **工程师** 实现
- **QA 工程师** 运行测试

每个角色有严格的输入/输出 Schema。角色提示说角色*是什么*以及*必须产出什么*。`Code = SOP(Team)` 公式——确定性 SOP 将 LLM 团队变成可预测的流水线。

### 2.3 ChatDev 的通信去幻觉

ChatDev 添加了一个关键动作：当执行者需要计划中没有的特定细节时，它在继续之前明确询问设计者。这防止了经典的 LLM 失败——似是而非地编造细节。

实现：角色提示包含"当你需要计划中没有的特定信息时，在产出之前按名称询问相关角色。"

### 2.4 为什么验证者最重要

Cemri 等人（MAST）追踪了 1642 个多智能体执行失败：

| 失败类别 | 占比 | 说明 |
|---------|------|------|
| **验证缺口** | 21.3% | 系统交付了没有人检查过的答案 |
| 检查静默失败 | ~30% | 检查存在但静默失败或从未运行 |
| 其他 | ~49% | 分解、通信、协调问题 |

PwC 报告（CrewAI 部署，2025）在添加结构化验证循环后准确率从 10% 提升到 70%。**7× 增益来自一个角色。**

### 2.5 批评者 vs 验证者

| 维度 | 批评者 | 验证者 |
|------|--------|--------|
| 类型 | LLM 审查 | 确定性程序 |
| 检查 | 主观质量 | 客观通过/失败 |
| 可以被愚弄 | 是（似是而非的散文） | 否（运行时测试） |
| 捕获什么 | 品味问题、风格问题 | 运行时才出现的 bug |
| 速度 | 快 | 可能慢 |
| 成本 | 中 | 低（确定性执行） |

两者都需要。批评者捕获验证者无法表达的品味问题；验证者捕获批评者看不到的运行时 bug。

### 2.6 通信去幻觉的深层逻辑

没有通信去幻觉时，执行者面对信息缺失的典型行为：

```
执行者需要：数据库连接字符串
实际行为：编造一个看起来合理的字符串（"mongodb://localhost:27017"）
后果：连接到错误的数据库，静默产生错误结果
```

有通信去幻觉时：

```
执行者需要：数据库连接字符串
实际行为：按名称询问规划者"数据库连接字符串是什么？"
规划者响应：从需求文档中检索，返回正确值
后果：正确连接，正确结果
```

这防止了经典的 LLM 失败——编造看起来合理但实际错误的信息。通过强制显式通信，而不是让智能体在信息缺失时静默填充。

---

## 3. 从零实现

### 第 1 步：定义规格和工件

```python
@dataclass
class Spec:
    task_name: str
    signature: str
    description: str
    tests: list[tuple[tuple, int]]

@dataclass
class Artifact:
    code: str

@dataclass
class CriticReport:
    approved: bool
    notes: list[str] = field(default_factory=list)

@dataclass
class VerifierReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
```

### 第 2 步：实现四个角色

```python
def planner(user_wish):
    """规划者：将愿望转为结构化规格。"""
    return Spec(
        task_name="add_two",
        signature="add_two(a: int, b: int) -> int",
        description=user_wish,
        tests=[((1, 2), 3), ((10, 20), 30), ((-5, 5), 0)],
    )

def executor_correct(spec):
    return Artifact(code="def add_two(a, b):\n    return a + b\n")

def executor_buggy(spec):
    return Artifact(code="def add_two(a, b):\n    return a * b\n")

def critic(spec, art):
    """LLM 批评者——可以被似是而非的代码愚弄。"""
    notes = []
    if "def" not in art.code: notes.append("缺少 def 语句")
    if "return" not in art.code: notes.append("缺少 return")
    if spec.task_name not in art.code: notes.append(f"函数名不匹配 '{spec.task_name}'")
    return CriticReport(approved=not notes, notes=notes)

def verifier(spec, art):
    """确定性验证者——在沙箱中执行代码并运行测试。"""
    ns = {}
    try:
        exec(art.code, ns, ns)
    except Exception as e:
        return VerifierReport(passed=False, failures=[f"执行错误: {e}"])
    fn = ns.get(spec.task_name)
    if not callable(fn):
        return VerifierReport(passed=False, failures=[f"未产生可调用的 '{spec.task_name}'"])
    failures = []
    for args, expected in spec.tests:
        try:
            got = fn(*args)
        except Exception as e:
            failures.append(f"调用 {args} 抛出 {e}")
            continue
        if got != expected:
            failures.append(f"调用 {args}: 期望 {expected}, 得到 {got}")
    return VerifierReport(passed=not failures, failures=failures)
```

### 第 3 步：实现流水线

```python
def run_pipeline(user_wish, executor, label):
    print(f"\n=== {label} ===")
    spec = planner(user_wish)
    print(f"  [规划者] 规格: {spec.signature}, {len(spec.tests)} 个测试")
    art = executor(spec)
    print(f"  [执行者] 产出:\n    {art.code.replace(chr(10), chr(10)+'    ')}")
    crep = critic(spec, art)
    print(f"  [批评者] approved={crep.approved}, notes={crep.notes}")
    vrep = verifier(spec, art)
    print(f"  [验证者] passed={vrep.passed}, failures={vrep.failures}")
    if crep.approved and vrep.passed:
        print("  结论: 交付。")
    elif not vrep.passed:
        print("  结论: 验证者阻止交付（确定性捕获）。")
    elif not crep.approved:
        print("  结论: 批评者阻止交付（主观捕获）。")
```

### 第 4 步：运行演示

```python
def main():
    run_pipeline("返回两个整数之和的函数。", executor_correct, "正确执行输出")
    run_pipeline("返回两个整数之和的函数。", executor_buggy, "有缺陷执行输出（乘法而非加法）")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 框架角色映射

| 框架 | 角色支持 | 角色数量 |
|------|---------|---------|
| CrewAI | `Agent(role, goal, backstory)` | 无限（任意角色） |
| LangGraph | 节点专业化提示词 + 边强制流水线 | 按图节点 |
| AutoGen | GroupChat 中的单字命名角色 | 按会话 |
| OpenAI Agents SDK | 角色专业化 Agent 之间的移交 | 按工具 |

---

## 5. 工程最佳实践

### 5.1 角色专业化原则

| 原则 | 说明 |
|------|------|
| 至少一个确定性验证者 | 永远不要全 LLM |
| 每个角色显式 I/O Schema | 规划者返回规格而非散文 |
| 通信去幻觉 | 执行者缺少信息时必须问规划者，不编造 |
| 批评者在验证者之前 | 批评者廉价捕获设计问题，验证者捕获 bug |
| 修订循环预算 | 批评者-执行者修订最多 2 轮后升级到人类 |

### 5.2 中文场景特别建议

- **角色提示用中文写**——方便中文团队理解和维护
- **验证者检查可以用中文输出错误信息**——方便中文开发者理解
- **通信去幻觉在中文中更必要**——中文 LLM 更容易编造似是而非的信息
- **MetaGPT 的 SOP 模式在中文项目中同样适用**——代码 = SOP(团队) 的公式通用

---

## 6. 常见错误

### 错误 1：全 LLM 反模式

**现象：** 每个角色都是 LLM，每个角色的输出是"看起来不错"。经典 MAST 失败模式。21.3% 的失败是验证缺口。

**修复：** 至少一个确定性验证者。验证者的通过/失败由代码决定，不由 LLM 决定。

### 错误 2：没有通信去幻觉

**现象：** 执行者在信息缺失时编造细节，而不问规划者。编造的字符串连接到错误的数据库。

**修复：** 角色提示包含"当你需要计划中没有的特定信息时，在产出之前按名称询问相关角色。"

### 错误 3：验证者和批评者顺序颠倒

**现象：** 验证者先运行（慢），然后批评者（快）。浪费验证者的计算在设计问题上。

**修复：** 批评者先运行（廉价，捕获设计问题），验证者后运行（昂贵，捕获运行时 bug）。

---

## 7. 面试考点

### Q1：四个标准角色是什么？关键区别是什么？（难度：⭐）

**参考答案：**
规划者（产生规格）、执行者（产出工件）、批评者（主观 LLM 审查）、验证者（确定性代码检查）。

**关键区别：** 批评者是主观的、可以被似是而非的散文愚弄；验证者是客观的、确定性的、不能被愚弄。两者都需要——批评者捕获品味问题，验证者捕获运行时 bug。

### Q2：MAST 的验证缺口数据说明什么？（难度：⭐⭐）

**参考答案：**
1642 个多智能体执行失败中，21.3% 是验证缺口——系统交付了没有人检查过的答案。另外约 30% 是检查存在但静默失败或从未运行。

PwC 报告在 CrewAI 中添加验证循环后准确率从 10% 提升到 70%。验证者是负载承载的角色。没有验证者的系统是经典 MAST 失败模式。

### Q3：通信去幻觉解决了什么问题？（难度：⭐⭐）

**参考答案：**
当执行者需要计划中没有的特定细节时，经典 LLM 行为是似是而非地编造细节——例如编造数据库连接字符串连接到错误的数据库。

ChatDev 的解决方案：执行者在继续之前明确按名称询问相关角色。这防止了经典 LLM 失败——通过强制显式通信，而不是让智能体在信息缺失时静默填充。

### Q4：MetaGPT 的 SOP 模式如何工作？（难度：⭐⭐⭐）

**参考答案：**
`Code = SOP(Team)` 公式：每个角色有严格的输入/输出 Schema。产品经理写 PRD → 架构师产系统设计 → 项目经理拆任务 → 工程师实现 → QA 工程师运行测试。

确定性 SOP 将 LLM 团队变成可预测的流水线。每个角色提示说角色*是什么*以及*必须产出什么*。这不是自由发挥——是编码的标准操作程序。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 角色专业化 | "不同智能体，不同工作" | 为规划者/执行者/批评者/验证者角色调优的不同系统提示词 |
| SOP 模式 | "编码的标准操作程序" | MetaGPT：每个角色的严格 I/O Schema 将团队变成流水线 |
| 通信去幻觉 | "先问再编造" | ChatDev 模式：执行者在信息缺失时询问规划者而非编造 |
| 批评者 | "LLM 审查者" | 主观的、有观点的审查者。捕获品味问题 |
| 验证者 | "确定性检查" | 基于代码的通过/失败。测试运行器、类型检查器、Schema 验证器 |
| 验证缺口 | "没人检查" | MAST 21.3% 的失败。答案未经检查就交付 |
| 全 LLM 反模式 | "看起来不错" | 每个角色都是 LLM，没有确定性检查。经典 MAST 失败 |

---

## 📚 小结

四个标准角色：规划者（规格）、执行者（工件）、批评者（主观审查）、验证者（确定性检查）。批评者和验证者是不同的角色——批评者是主观 LLM，验证者是确定性代码。MAST 追踪的 1642 个失败中 21.3% 是验证缺口。PwC 在 CrewAI 中添加验证后准确率从 10% 提升到 70%——一个角色的 7× 增益。通信去幻觉防止执行者在信息缺失时编造。全 LLM 管道是经典 MAST 失败模式。

下一课：并行、群体和网络化架构。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。观察验证者如何捕获批评者漏过的 bug。添加静态分析检查（计算 `return` 出现次数）作为额外验证者。

2. **【实现】** 添加第五个角色："需求分析师"——将用户愿望转化为规划者就绪的规格。应该向上游流动什么通信去幻觉请求？

3. **【阅读】** 阅读 MetaGPT 第 3 节（"Agents"）。列出 MetaGPT 5 个角色中每个的输入/输出 Schema。

4. **【阅读】** 阅读 ChatDev 的聊天链图（arXiv:2307.07924 图 3）。识别通信去幻觉在哪里打破了本来会无限的循环。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 角色流水线 | `code/main.py` | 规划者→执行者→批评者→验证者，含 bug 演示 |
| 技能提示词 | `outputs/skill-role-designer.md` | 为任务生成角色名单、I/O Schema、验证者检查 |

---

## 📖 参考资料

1. [论文] Hong 等人. "MetaGPT: Meta Programming for Multi-Agent Collaboration". https://arxiv.org/abs/2308.00352
2. [论文] Qian 等人. "Communicative Agents for Software Development (ChatDev)". https://arxiv.org/abs/2307.07924
3. [论文] Cemri 等人. "Why Do Multi-Agent LLM Systems Fail?". https://arxiv.org/abs/2503.13657 — MAST 分类法
4. [文档] CrewAI. https://docs.crewai.com/en/introduction — 角色规范表面

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
