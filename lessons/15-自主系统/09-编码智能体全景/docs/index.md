# 编码智能体全景（2026）

> SWE-bench Verified 在不到三年内从 4% 涨到 80.9%。同一 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——围绕模型的脚手架现在和模型本身一样重要。OpenHands（前身 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环在沙箱中直接执行 Python 动作而非 JSON 工具调用。头条数字隐藏了一个方法学问题：500 个 SWE-bench Verified 任务中 161 个只需 1-2 行改动，SWE-bench Pro（10+ 行改动）对同一前沿模型仍停留在 23-59%。

**类型：** 概念课
**语言：** Python（标准库，CodeAct vs JSON 工具调用对比）
**前置知识：** 阶段 14 · 07（工具使用）、阶段 15 · 01（长期智能体）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么"哪个编码智能体最好"是错误的问题——正确的问题是"在我工作分布上、在我生产脚手架中，端到端可靠性是多少"
- [ ] 区分 CodeAct（代码即动作）和 JSON 工具调用的架构权衡
- [ ] 识别 SWE-bench Verified 的方法学问题——简单任务尾巴膨胀了分数
- [ ] 理解为什么脚手架是负载承载的——同一模型在不同脚手架下表现差异可达 16 个百分点
- [ ] 对比 2026 年主要编码智能体平台的架构和特性

---

## 1. 问题

"哪个编码智能体最好"是错误的问题。正确的问题是：**在匹配我工作的任务分布上、在我生产中运行的脚手架中，端到端可靠性是多少？**

2022-2026 年间，领域学到了脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是负载承载的。Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%；同一模型在 Cline 的自主脚手架中得分 59.8%。16.6 个百分点的绝对差异，相同权重。基础模型是一个组件；循环是产品。

伴随问题是基准饱和隐藏了回归。SWE-bench Verified 接近饱和，简单任务尾巴（500 个任务中 161 个只需 ≤2 行改动）拉高了顶级分数。真实质量更好地用 SWE-bench Pro（10+ 行改动）等分布来测量，同一前沿系统仍停留在 23-59%。

---

## 2. 概念

### 2.1 SWE-bench 一段话总结

SWE-bench（Jimenez 等人）取真实 GitHub issue 和真实补丁，要求智能体产生一个让测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工策划的 500 个任务子集，移除了模糊和损坏的任务。SWE-bench Pro 是更难的后续——需要 10+ 行改动的任务，当前前沿智能体停留在 23-59%。

### 2.2 2022 → 2026 曲线实际展示了什么

| 时间 | 得分 | 关键因素 |
|------|------|---------|
| 2022 | ~4% | 研究模型在原始 SWE-bench 上 |
| 2024 | ~14% | GPT-4 + Devin 式脚手架；SWE-agent ~12% |
| 2025 | 40-55% | Claude 3.5/3.7 Sonnet + Aider/SWE-agent |
| 2026 | 70-80%+ | Claude Sonnet 4.5 + 前沿竞争者在 Verified 上 |

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、更好的基准（Verified 移除噪声）。

### 2.3 CodeAct vs JSON 工具调用

| 维度 | JSON 工具调用 | CodeAct |
|------|-------------|---------|
| 动作粒度 | 每个动作一个 JSON 载荷 | 一个动作可以是整个程序 |
| 组合性 | 有限——每个调用独立验证 | 高——一个代码片段可以循环、链接工具、捕获异常 |
| 安全性 | 默认安全——每个调用通过显式验证器 | 需要加固沙箱（Docker 隔离） |
| 审计 | 容易——每个动作是结构化 JSON | 较难——一个动作可以做更多事 |
| 典型用户 | 管理服务（Anthropic Managed Agents、OpenAI Assistants） | 开源平台（OpenHands、smolagents） |

两种架构都在生产中。CodeAct 在开源平台中占主导；JSON 工具调用在管理服务中保持主导。

### 2.4 2026 年脚手架景观

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|--------|--------|---------|---------|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃开源平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的 diff 编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上最高分开源脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细覆盖智能体循环 |

### 2.5 为什么脚手架主导

编码运行是长期轨迹（第 1 课）。可靠性跨步骤复合。脚手架在三个地方购买分数：

1. **检索**——找到正确的文件是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的仓库地图都攻击这个
2. **验证器循环**——运行测试、读取堆栈跟踪、重新尝试——在 SWE-bench 上是 10+ 分的差异
3. **失败遏制**——出错时回滚的沙箱防止复合损伤。同一模型有和没有验证器循环看起来像两个不同的产品

### 2.6 基准饱和和真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾巴：500 个任务中 161 个只需 1-2 行改动。高分部分由这个尾巴驱动。SWE-bench Pro 限制为 10+ 行改动，即使前沿系统也返回 23-59%。你的生产分布几乎肯定更接近 Pro 而非 Verified。

对选择智能体的含义：在你自己的 bug 积压中运行一个类似 Pro 的子集。重要的分数是代表你交付内容的任务上的分数。

---

## 3. 从零实现

### 第 1 步：定义迷你仓库和测试

```python
INITIAL_REPO = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "util.py": "def lower(s):\n    return s.upper()\n",
    "cli.py": "VERSION = 'v0.0'\n",
}

TESTS = [
    ("app.py", "add(2, 3) == 5"),
    ("util.py", "lower('AB') == 'ab'"),
    ("cli.py", "VERSION == 'v1.0'"),
]

FIXES = {
    "app.py": ("a - b", "a + b"),
    "util.py": ("s.upper()", "s.lower()"),
    "cli.py": ("v0.0", "v1.0"),
}
```

### 第 2 步：实现 JSON 工具调用脚手架

```python
@dataclass
class JsonScaffold:
    repo: dict = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0

    def step(self) -> str:
        """每次返回一个 JSON 动作。"""
        self.turns += 1
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results):
            if not ok:
                _apply_fix(self.repo, path)
                return json.dumps({"tool": "edit", "path": path})
        return json.dumps({"tool": "done"})
```

### 第 3 步：实现 CodeAct 脚手架

```python
@dataclass
class CodeActScaffold:
    repo: dict = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0
    worst_touched: int = 0

    def step(self) -> str:
        """返回一个可能编辑多个文件的 Python 片段。"""
        self.turns += 1
        snippet_lines = []
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results):
            if not ok:
                _apply_fix(self.repo, path)
                snippet_lines.append(f"fs.write('{path}', ...)")
        self.worst_touched = max(self.worst_touched, len(snippet_lines))
        if not snippet_lines:
            return "done()"
        return "; ".join(snippet_lines)
```

### 第 4 步：运行对比

```python
def main():
    print("同一存根模型，三 bug 仓库。纯脚手架对比。")

    js = JsonScaffold()
    passed, turns = js.run()
    report("JSON 工具调用", passed, turns, js.blast_radius())

    ca = CodeActScaffold()
    passed, turns = ca.run()
    report("CodeAct (存根)", passed, turns, ca.blast_radius())
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 SWE-bench 分数的正确解读

| 子集 | 任务数 | 难度 | 前沿得分 | 解读 |
|------|--------|------|---------|------|
| SWE-bench Verified | 500 | 混合 | 70-80%+ | 有简单尾巴（161 个 ≤2 行） |
| SWE-bench Pro | ~200 | 高 | 23-59% | 10+ 行改动，更接近生产 |

### 4.2 脚手架选择策略

| 需求 | 推荐 | 原因 |
|------|------|------|
| 快速原型 | Aider | 最小脚手架，强回归稳定性 |
| 最高分 | Cline | Sonnet 4.5 上最高分 |
| 开源可审计 | OpenHands | MIT 许可，事件流可重放 |
| 托管服务 | Devin / Claude Code | 供应商控制执行器 |

---

## 5. 工程最佳实践

### 5.1 编码智能体选择原则

| 原则 | 说明 |
|------|------|
| 脚手架是产品 | 同一模型在不同脚手架下差异可达 16 个百分点 |
| 基准分数要解读 | Verified 有简单尾巴；Pro 更接近生产 |
| 用你自己的分布测试 | 在你的 bug 积压上运行类似 Pro 的子集 |
| CodeAct vs JSON 是权衡 | CodeAct 组合性高但需要加固沙箱 |

---

## 6. 常见错误

### 错误 1：只看 SWE-bench Verified 分数

**现象：** "我们的智能体在 SWE-bench Verified 上得了 80%！" 但在生产中只有 50% 的任务被正确修复。

**原因：** Verified 有简单尾巴——161/500 个任务只需 1-2 行改动。高分部分由这个尾巴驱动。

**修复：** 在你的生产分布上测试，或者至少在 SWE-bench Pro（10+ 行）上测试。

### 错误 2：忽视脚手架

**现象：** "Claude 4.5 够好了"，然后直接用默认配置运行。结果远低于基准分数。

**原因：** 基准分数假设特定脚手架。Claude Sonnet 4.5 在 SWE-agent v1 上 43.2%，在 Cline 上 59.8%——同一模型，16.6 个百分点差异。

**修复：** 脚手架是产品。选择脚手架和选择模型同样重要。

### 错误 3：认为 CodeAct 总是更好

**现象：** CodeAct 组合性高，用 CodeAct 替换了所有 JSON 工具调用。

**原因：** CodeAct 需要加固沙箱。一个动作可以做更多事，但也意味着更大的爆炸半径。管理服务需要可审计性——JSON 工具调用更容易审计。

**修复：** CodeAct vs JSON 是权衡，不是替代。开源平台用 CodeAct，管理服务用 JSON 工具调用。

---

## 7. 面试考点

### Q1：为什么"哪个编码智能体最好"是错误的问题？（难度：⭐）

**参考答案：**
因为分数取决于三个变量的组合：模型、脚手架、任务分布。同一模型在不同脚手架下差异可达 16 个百分点（Claude Sonnet 4.5: 43.2% vs 59.8%）。

正确的问题是：在我的任务分布上、在我生产的脚手架中，端到端可靠性是多少？这需要在你自己的数据上测试。

### Q2：CodeAct 和 JSON 工具调用的核心权衡是什么？（难度：⭐⭐）

**参考答案：**

| 维度 | JSON 工具调用 | CodeAct |
|------|-------------|---------|
| 动作粒度 | 每次一个结构化 JSON | 一个代码片段可以做更多事 |
| 安全性 | 默认安全——每个调用独立验证 | 需要加固沙箱 |
| 审计 | 容易——结构化 JSON | 较难——一个动作可以做更多事 |

两种架构都在生产中。开源平台（OpenHands）用 CodeAct；管理服务（Anthropic、OpenAI）用 JSON 工具调用。

### Q3：SWE-bench Verified 的方法学问题是什么？（难度：⭐⭐）

**参考答案：**
500 个任务中 161 个只需 1-2 行改动。高分部分由这个简单尾巴驱动。SWE-bench Pro（10+ 行改动）对同一前沿模型返回 23-59%——更接近生产分布。

对选择智能体的含义：不要只看 Verified 分数。在你自己的 bug 积压中运行类似 Pro 的子集。

### Q4：脚手架在哪些地方购买分数？（难度：⭐⭐⭐）

**参考答案：**
三个地方：

（1）**检索**——找到正确的文件是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的仓库地图都攻击这个。
（2）**验证器循环**——运行测试、读取堆栈跟踪、重新尝试——在 SWE-bench 上是 10+ 分的差异。
（3）**失败遏制**——出错时回滚的沙箱防止复合损伤。同一模型有和没有验证器循环看起来像两个不同的产品。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SWE-bench | "编码基准" | 真实 GitHub issue + 真实补丁 + 测试套件 |
| SWE-bench Verified | "清理子集" | 500 个人工策划任务，有简单尾巴 |
| SWE-bench Pro | "更难子集" | 10+ 行改动；前沿系统停留在 23-59% |
| CodeAct | "代码即动作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个动作是一个结构化 JSON 载荷，执行前验证 |
| 脚手架 (Scaffold) | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI | "SWE-agent 的格式" | 为 LLM 人体工程学设计的命令集，而非人类 shell |
| 验证器循环 | "测试-重试" | 运行测试、读取输出、修订补丁——最大的非模型可靠性收益 |

---

## 📚 小结

"哪个编码智能体最好"是错误的问题。脚手架是负载承载的——同一模型在不同脚手架下差异可达 16 个百分点。CodeAct（代码即动作）提供高组合性但需要加固沙箱；JSON 工具调用提供安全性但组合性有限。SWE-bench Verified 有简单尾巴——161/500 个任务只需 1-2 行改动。SWE-bench Pro（10+ 行）更接近生产分布，前沿系统仍停留在 23-59%。选择脚手架和选择模型同样重要。

至此第 15 章第 1-9 节完成：视界增长（01）→ STaR（02）→ AlphaEvolve（03）→ DGM（04）→ AI Scientist v2（05）→ AAR（06）→ RSI 竞赛（07）→ 有界改进（08）→ 编码智能体全景（09）。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。每个脚手架在相同任务集上用了多少轮？每个脚手架的每次动作爆炸半径是多少？

2. **【阅读】** 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证 CodeAct 在复杂任务上优于 JSON 工具调用。识别论文承认的一个失败模式，写一句话说明该模式在生产中何时占主导。

3. **【思考】** 选择你 bug 积压中一个需要 10+ 行改动跨两个文件的任务。估算前沿模型在（a）JSON 工具调用和（b）CodeAct 下的端到端成功率。论证差距。

4. **【实验】** SWE-bench Verified 有 161 个单文件、1-2 行的任务。构建一个排除它们的分数。排行榜如何变化？

5. **【阅读】** 阅读 OpenAI "Introducing SWE-bench Verified"。解释用于移除模糊任务的具体方法学，命名一个策划会遗漏的类别。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 脚手架对比 | `code/main.py` | JSON 工具调用 vs CodeAct，相同模型 |
| 技能提示词 | `outputs/skill-scaffold-audit.md` | 审计提议的编码智能体脚手架 |

---

## 📖 参考资料

1. [排行榜] SWE-bench. https://www.swebench.com/ — 原始基准和方法论
2. [博客] OpenAI. "Introducing SWE-bench Verified". https://openai.com/index/introducing-swe-bench-verified/ — 策划子集的构建
3. [论文] Wang et al. "OpenHands: An Open Platform for AI Software Developers". https://arxiv.org/abs/2407.16741 — CodeAct 架构和事件流设计
4. [排行榜] Epoch AI. "SWE-bench Leaderboard". https://epoch.ai/benchmarks — 实时追踪分数
5. [博客] Anthropic. "Measuring Agent Autonomy". https://www.anthropic.com/research/measuring-agent-autonomy — 长期编码智能体可靠性框架

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
