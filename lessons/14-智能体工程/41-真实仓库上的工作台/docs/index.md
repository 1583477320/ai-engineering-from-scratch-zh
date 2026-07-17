# 真实仓库上的工作台——用数字证明工作台的价值

> 十一课的面组合成一个整体，如果不能在真实代码库上存活就没有价值。本课在同一任务上运行两次——纯提示词 vs 工作台引导。数字自己会说话。

**类型：** 实现课
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 32 到 14 · 40（全部工作台课程）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 14 · 30（评估驱动开发）— 本课是评估驱动在工作台场景的具体实例

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 在小型应用上将七个工作台面组合在一起
- [ ] 运行同一个任务两次（纯提示词和工作台引导），测量五个结果
- [ ] 阅读前后对比报告，判断哪些面贡献最大
- [ ] 用数字而非论证回应"但我的模型足够好"的质疑

---

## 1. 问题

演示在玩具任务上不能说服任何人。工作台的价值只有在一个接近真实感的任务、接近真实感的仓库、以更少的失败、更少的回滚、以及一个下一个会话可以使用的交接包进入生产时才能被证明。

本课提供那个"接近真实感"的仓库，并让同一任务通过两条流水线运行。结果是一份你可以交给质疑者的前后对比报告。

---

## 2. 概念

### 2.1 示例应用

```
sample_app/
├── app.py              # /signup 接口（无验证）
├── test_app.py         # 一个 happy-path 测试
├── README.md           # 禁止区域诱饵
└── scripts/
    └── release.sh      # 禁止区域诱饵
```

### 2.2 任务

> 为 `/signup` 添加输入验证：拒绝密码少于 8 个字符的请求，返回带类型化错误信封的 422。添加一个证明新行为的测试。

### 2.3 两条流水线

**纯提示词流水线：**
1. 读 README
2. 读 `app.py`
3. 编辑文件
4. 声称完成

**工作台引导流水线：**
1. 运行初始化脚本（阶段 14 · 35）
2. 读取范围契约（阶段 14 · 36）
3. 读取状态文件（阶段 14 · 34）
4. 只修改允许的文件
5. 通过反馈运行器运行验收命令（阶段 14 · 37）
6. 运行验证门控（阶段 14 · 38）
7. 运行审查员（阶段 14 · 39）
8. 生成交接包（阶段 14 · 40）

### 2.4 测量的五个结果

| 结果 | 为什么重要 |
|------|----------|
| `tests_actually_run` | 大多数"测试通过"是不可验证的 |
| `acceptance_met` | 证明目标的测试必须是真正运行的测试 |
| `files_outside_scope` | 范围蔓延是主要的静默失败 |
| `handoff_quality` | 下一个会话要为这次付出代价或受益 |
| `reviewer_total` | 门控之上的定性判断 |

---

## 3. 从零实现

### 第 1 步：定义示例应用

```python
SAMPLE_APP_PY = '''"""Minimal signup handler."""
USERS: dict[str, str] = {}

def signup(email: str, password: str) -> dict:
    USERS[email] = password
    return {"status": 200, "email": email}
'''

SAMPLE_TEST_PY = '''from sample_app.app import signup

def test_signup_happy_path():
    out = signup("a@b.co", "longenough")
    assert out["status"] == 200
'''
```

### 第 2 步：定义结果结构和流水线

```python
from dataclasses import dataclass, field

@dataclass
class TaskOutcome:
    pipeline: str
    tests_actually_run: bool
    acceptance_met: bool
    files_outside_scope: list[str] = field(default_factory=list)
    handoff_quality: str = "missing"
    reviewer_total: int = 0

ALLOWED = {"sample_app/app.py", "sample_app/test_app.py"}
FORBIDDEN = {"sample_app/scripts/release.sh"}

def run_prompt_only() -> TaskOutcome:
    """纯提示词：修改文件但不运行测试，声称完成。"""
    touched = ["sample_app/app.py", "README.md", "sample_app/scripts/release.sh"]
    return TaskOutcome(
        pipeline="prompt-only",
        tests_actually_run=False,
        acceptance_met=False,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="missing",
        reviewer_total=3,
    )

def run_workbench() -> TaskOutcome:
    """工作台引导：在范围内修改，运行验收，通过门控，审查，交接。"""
    touched = ["sample_app/app.py", "sample_app/test_app.py"]
    return TaskOutcome(
        pipeline="workbench-guided",
        tests_actually_run=True,
        acceptance_met=True,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="full packet",
        reviewer_total=9,
    )
```

### 第 3 步：生成对比报告

```python
def write_report(po: TaskOutcome, wb: TaskOutcome) -> str:
    lines = [
        "# 前后对比：智能体工作台在真实仓库上",
        "",
        "同一任务。同一样本应用。两条流水线。",
        "",
        "| 结果 | 纯提示词 | 工作台 |",
        "|------|---------|--------|",
        f"| tests_actually_run | {po.tests_actually_run} | {wb.tests_actually_run} |",
        f"| acceptance_met | {po.acceptance_met} | {wb.acceptance_met} |",
        f"| files_outside_scope | {len(po.files_outside_scope)} | {len(wb.files_outside_scope)} |",
        f"| handoff_quality | {po.handoff_quality} | {wb.handoff_quality} |",
        f"| reviewer_total (/10) | {po.reviewer_total} | {wb.reviewer_total} |",
        "",
        "## 结论",
        "",
        "纯提示词流水线修改了禁止区域内的文件，在没有运行验收命令的情况下声称完成，",
        "没有交接包，审查得分很低。工作台流水线保持在范围内修改，通过反馈运行器运行验收命令，",
        "通过验证门控，生成可交付的交接包。",
    ]
    return "\n".join(lines)
```

### 第 4 步：运行对比

```python
def main():
    po = run_prompt_only()
    wb = run_workbench()

    for outcome in (po, wb):
        print(f"=== {outcome.pipeline} ===")
        for k, v in asdict(outcome).items():
            print(f"  {k}: {v}")
        print()

    report = write_report(po, wb)
    print(report)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 数据证据

| 数据点 | 来源 | 结论 |
|--------|------|------|
| Terminal Bench Top-30 → Top-5 | LangChain, 2026 年 4 月 | 同模型，改变工作台后排名提升 25 位 |
| Vercel 80% → 100% | MongoDB | 删除 80% 工具后成功率到 100% |
| Harvey 2x 准确率 | MongoDB | 仅优化工作台，无模型变更 |
| 88% 企业智能体项目失败 | preprints.org, 2026 年 3 月 | 失败集中在运行时，不是推理 |
| WebAgent 40-50% → <10% | 长上下文 | 无限循环和目标丢失 |

### 4.2 假阴性——诚实列出工作台的局限

单步事实任务、单行 lint、格式化运行、模型已经背下的内容——这些纯提示词更快。诚实地列举这些案例，工作台才不会被框架为"杀鸡用牛刀"。

---

## 5. 工程最佳实践

### 5.1 前后对比报告设计

| 原则 | 说明 |
|------|------|
| 数字胜过论证 | Terminal Bench 排名 30 → 5 比"工作台很有效"更有说服力 |
| 诚实地列出局限 | 假阴性（纯提示词更快的场景）让论证更可信 |
| 同任务同模型 | 唯一变量是工作台，不是模型 |
| 可复现 | 两条流水线都是脚本化的，没有 LLM |

### 5.2 中文场景特别建议

- **样本应用可以换成中文示例**——`/signup` 换成 `/注册`，但代码逻辑保持不变
- **前后对比报告用中文写**——方便团队传阅，关键数字保持英文
- **对比表的列名保持中英文一致**——与 30-37 节的输出格式对齐

### 5.3 踩坑经验

- **用演示任务而不是真实任务**——玩具任务的对比数据没有说服力。**修复：** 用真实仓库的真实任务
- **只列正面数据**——只说"工作台赢了"，不说"纯提示词在哪些场景更快"。**修复：** 诚实列举假阴性
- **不追踪五个结果**——只报告"通过/失败"，没有细粒度数据。**修复：** 测试运行、验收达标、范围蔓延、交接质量、审查得分五个维度

---

## 6. 常见错误

### 错误 1：只在玩具任务上对比

**现象：** 在 10 行的 TODO 应用上运行对比，工作台赢了 5-0。质疑者说"我的任务比这复杂 10 倍"。数据无效。

**原因：** 玩具任务无法触发工作台的真正价值——范围蔓延、多文件修改、交接需求

**修复：** 用接近真实感的样本应用——多个文件、禁止区域、验收命令、交接需求

### 错误 2：工作台数据比实际好

**现象：** 对比数据中工作台得分比实际运行时好。因为脚本化的"智能体"太完美了，没有模型会犯的错误。

**原因：** 脚本化的智能体是理想化的，真实智能体会犯错

**修复：** 脚本化流水线的目的是可复现性和可比较性。在报告中明确说明这是"有工作台 vs 无工作台"的对比，而不是"完美智能体 vs 真实智能体"

### 错误 3：没有列出假阴性

**现象：** 报告说"工作台在所有 5 个维度都赢了"。质疑者试了一个单行 lint 任务，发现工作台确实更慢。质疑者用这个案例否定了整个论证。

**原因：** 没有诚实地列出工作台更慢或更重的场景

**修复：** 报告中明确列出假阴性：单步任务、格式化任务、记忆中的内容——这些场景纯提示词更快。诚实让论证更可信

---

## 7. 面试考点

### Q1：前后对比报告测量哪五个结果？（难度：⭐）

**参考答案：**
（1）测试是否实际运行（`tests_actually_run`）
（2）验收是否达标（`acceptance_met`）
（3）修改是否在范围外（`files_outside_scope`）
（4）交接质量（`handoff_quality`）
（5）审查员得分（`reviewer_total`）

这五个维度覆盖了从执行可靠性到后续可维护性的完整生命周期。

### Q2：2026 年的工作台数据证据有哪些？（难度：⭐⭐）

**参考答案：**
（1）Terminal Bench Top-30 → Top-5：同模型，改变工作台后排名提升 25 位
（2）Vercel 80% → 100%：删除 80% 工具后成功率到 100%
（3）Harvey 2x 准确率：仅优化工作台
（4）88% 企业智能体项目失败：运行时问题，不是推理
（5）WebAgent 40-50% → <10%：长上下文下的崩溃

### Q3：什么是假阴性？为什么需要诚实地列出？（难度：⭐⭐）

**参考答案：**
假阴性是"纯提示词更快"的场景——单步事实任务、单行 lint、格式化运行。这些场景中工作台的额外步骤（初始化、范围检查、验证、审查、交接）确实是开销。

诚实地列出假阴性让论证更可信。如果只说"工作台在所有场景都赢"，一个质疑者试了一个单行任务就会否定整个论证。承认局限是好的论证技巧。

### Q4：如何回应"但我的模型足够好"的质疑？（难度：⭐⭐⭐）

**参考答案：**
三个层面回应：

（1）**数据层面**：Terminal Bench 上同模型排名 30 → 5，Harvey 2x 准确率——模型相同，工作台不同。

（2）**架构层面**：88% 的企业智能体项目失败是因为运行时问题（状态过时、重试脆弱、上下文膨胀），不是推理问题。模型再好也解决不了这些。

（3）**诚实层面**：承认单步任务纯提示词更快。但"模型足够好"只在模型能读文件、记住上次会话、验证范围、生成交接时成立——这些不是模型功能，是工作台功能。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 前后对比报告 | "证据" | 同任务、同模型，用数字对比两条流水线的产物 |
| 假阴性 | "工作台杀鸡用牛刀" | 纯提示词更快的场景——诚实列举让论证更可信 |
| 样本应用 | "玩具仓库" | 小但真实到能触发所有七个面的任务 |
| 工作台基准 | "可靠性分数" | 可移植的对比工具——在你自己的仓库上运行 |
| 五个结果 | "五维度对比" | 测试运行、验收达标、范围蔓延、交接质量、审查得分 |

---

## 📚 小结

十一课的工作台面只有在真实代码库上存活时才有价值。本课在同一小型应用上运行纯提示词和工作台引导两条流水线，测量五个结果，生成前后对比报告。数字自己会说话——Terminal Bench Top-30 → Top-5、Vercel 80% → 100%、Harvey 2x 准确率。诚实列举假阴性让论证更可信。

下一课是毕业设计——将所有十一课的面打包成可复用的工作台包。

---

## ✏️ 练习

1. **【实现】** 添加第六个结果：首次有意义编辑的耗时。如何准确测量？

2. **【实验】** 在你仓库的第二天任务上运行对比。工作台数字在哪些维度滑坡？

3. **【实现】** 添加假阴性通道：纯提示词更快的任务。论证为什么仍然需要工作台。

4. **【实现】** 将脚本化的"智能体"替换为真正的 LLM 调用。哪些结果变得不稳定？

5. **【思考】** 写一份面向非工程师的单页总结。什么内容会保留？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 前后对比报告 | `code/main.py` | 纯提示词 vs 工作台，生成对比表 |
| 技能提示词 | `outputs/skill-workbench-benchmark.md` | 在你自己的仓库上运行五维度对比 |

---

## 📖 参考资料

1. [博客] LangChain. "The Anatomy of an Agent Harness". https://blog.langchain.com/the-anatomy-of-an-agent-harness/ — Terminal Bench Top-30 → Top-5
2. [博客] MongoDB. "The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System". https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system — Vercel + Harvey 数据
3. [论文] "Harness Engineering for Language Agents". preprints.org, 2026 年 3 月. https://www.preprints.org/manuscript/202603.1756 — 88% 企业失败率
4. [Hacker News] "Improving 15 LLMs at Coding in One Afternoon. Only the Harness Changed". https://news.ycombinator.com/item?id=46988596 — 跨 15 个模型复现
5. [博客] Anthropic. "Building Effective Agents". https://www.anthropic.com/research/building-effective-agents

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
