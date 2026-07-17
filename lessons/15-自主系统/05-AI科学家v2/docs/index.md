# AI Scientist v2——工作坊级别的自主研究

> Sakana 的 AI Scientist v2（Yamada 等人，arXiv:2504.08066）运行完整的研究循环：假设、代码、实验、图表、论文、投稿。它是第一个生成论文通过 ICLR 2025 工作坊同行评审的系统。独立评估（Beel 等人）发现 42% 的实验因编码错误失败，文献综述经常将已建立的概念错误标记为新颖。Sakana 自己的文档警告代码库执行 LLM 编写的代码并建议 Docker 隔离。这幅图的两半都是重点。

**类型：** 概念课
**语言：** Python（标准库，研究循环状态机玩具）
**前置知识：** 阶段 15 · 03（AlphaEvolve）、阶段 15 · 04（DGM）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 AI Scientist v2 的七步研究循环架构
- [ ] 理解"呈现质量差距"——VLM 图表润色可以掩盖实验缺陷
- [ ] 对比三种系统的评估器严格性：AlphaEvolve（强）、DGM（中）、AI Scientist v2（弱）
- [ ] 识别沙箱逃逸风险——LLM 编写的代码可以做任何进程允许的事
- [ ] 理解为什么"看起来完美的论文"比"明显失败的系统"更危险

---

## 1. 问题

研究是一个开放式任务。不同于 AlphaEvolve 的算法搜索或 DGM 的基准约束自我修改，研究结果没有机器可检查的正确性标准。论文由评审人评判，不是单元测试。这使得循环更难闭合——但如果闭合，价值更大，因为研究是复合进步所在。

AI Scientist v1（Sakana，2024）通过从人类编写的模板开始来闭合循环。LLM 在固定脚手架内填充实验。AI Scientist v2（Yamada 等人，2025）通过使用带视觉语言模型评判循环的智能体树搜索来移除模板要求。

同行评审结果：一篇 v2 生成的论文在 ICLR 2025 工作坊被接受（已披露来源）。独立评估结果：系统远不可靠。两者都是真的。

---

## 2. 概念

### 2.1 架构

```
假设生成 → 文献新颖性检查 → 实验计划 → 代码执行（沙箱）
                                    ↓
                              图表生成（VLM 润色）
                                    ↓
                              论文撰写 → 内部评审 → 接受或迭代
                                    ↓
                              投稿（可选）
```

| 步骤 | 失败概率（Beel 等人数据） | 备注 |
|------|------------------------|------|
| 假设生成 | 低 | 但新颖性检查有误标风险 |
| 文献新颖性检查 | ~25% 误标 | 已建立概念被标记为新颖 |
| 实验执行 | ~42% 编码错误 | 导入错误、形状不匹配、未定义变量 |
| 重试恢复 | ~55% | 重试捕获部分错误，不是全部 |
| VLM 图表润色 | ~70% 掩盖缺陷 | 呈现质量超过实验质量 |
| 论文撰写 | ~85% 成功率 | 有缺陷的数据上生成连贯论文 |
| 内部评审 | ~50% 接受率 | 弱评审者 |

### 2.2 "呈现质量差距"的含义

VLM 图表润色产出了出版级视觉效果，掩盖了底层实验弱点。

**产生有说服力的输出但不做有说服力的研究的系统，比明显失败的系统更危险，而不是更安全。** 评估必须触及底层主张，而不是停在图表上。

### 2.3 沙箱逃逸担忧

Sakana 自己的仓库 README 警告：

> 由于本软件执行 LLM 生成的代码，我们无法保证安全。存在危险包、不受控的网络访问和意外进程生成的风险。自行承担风险并考虑 Docker 隔离。

这是未验证领域中自主性的操作形态。LLM 写代码；代码运行；代码可以做进程允许的任何事情。没有硬限制文件系统、网络和进程操作的沙箱，任何自主研究智能体都可以泄露数据、烧掉算力或重写自己。

### 2.4 在前沿栈中的位置

| 系统 | 目标 | 输出类型 | 评估器 | 已知失败 |
|------|------|---------|-------|---------|
| AlphaEvolve | 算法 | 代码 | 单元测试 + 基准 | 受评估器严格性约束 |
| DGM | 智能体脚手架 | 代码 | SWE-bench | 奖励黑客 |
| AI Scientist v2 | 研究论文 | 文本 + 代码 + 图表 | 同行评审（弱） | 实验失败、误标、润色掩盖弱点 |

v2 的自动评估器最弱，输出面最宽，到公共产物的路径最短。操作控制（沙箱、审查、披露）承担了大部分安全工作。

---

## 3. 从零实现

### 第 1 步：定义循环配置和结果

```python
import random
from dataclasses import dataclass

@dataclass
class LoopConfig:
    novelty_mislabel: float = 0.25       # 文献新颖性误标率
    experiment_failure: float = 0.42     # 实验失败率
    retry_recovery: float = 0.55         # 重试恢复率
    polish_masks_weakness: float = 0.70  # VLM 润色掩盖缺陷的概率
    writeup_success: float = 0.85        # 论文撰写成功率
    internal_review_accept: float = 0.50 # 内部评审接受率

@dataclass
class Outcome:
    submitted: bool
    has_novelty_flaw: bool
    has_experiment_flaw: bool
    polished_but_flawed: bool
    polished_ok: bool
    abandoned_stage: str
```

### 第 2 步：实现单次循环

```python
def run_one(cfg: LoopConfig) -> Outcome:
    """运行一次研究循环。"""
    has_novelty_flaw = random.random() < cfg.novelty_mislabel

    # 实验执行 + 重试
    failed = random.random() < cfg.experiment_failure
    if failed:
        recovered = random.random() < cfg.retry_recovery
        if not recovered:
            return Outcome(submitted=False, has_novelty_flaw=has_novelty_flaw,
                          has_experiment_flaw=True, polished_but_flawed=False,
                          polished_ok=False, abandoned_stage="experiment")
        has_experiment_flaw = True  # 重试恢复后仍有残余缺陷
    else:
        has_experiment_flaw = False

    # VLM 图表润色——可能掩盖缺陷
    polished_hides = has_experiment_flaw and random.random() < cfg.polish_masks_weakness

    # 论文撰写
    if random.random() > cfg.writeup_success:
        return Outcome(submitted=False, has_novelty_flaw=has_novelty_flaw,
                      has_experiment_flaw=has_experiment_flaw,
                      polished_but_flawed=False, polished_ok=False,
                      abandoned_stage="writeup")

    # 内部评审
    if random.random() > cfg.internal_review_accept:
        return Outcome(submitted=False, has_novelty_flaw=has_novelty_flaw,
                      has_experiment_flaw=has_experiment_flaw,
                      polished_but_flawed=False, polished_ok=False,
                      abandoned_stage="internal_review")

    polished_ok = not has_experiment_flaw and not has_novelty_flaw
    polished_but_flawed = has_experiment_flaw or has_novelty_flaw
    return Outcome(submitted=True, has_novelty_flaw=has_novelty_flaw,
                   has_experiment_flaw=has_experiment_flaw,
                   polished_but_flawed=polished_but_flawed,
                   polished_ok=polished_ok, abandoned_stage="")
```

### 第 3 步：运行多次试验并报告

```python
def report(n: int, cfg: LoopConfig) -> None:
    outs = [run_one(cfg) for _ in range(n)]
    submitted = [o for o in outs if o.submitted]
    polished_ok = [o for o in submitted if o.polished_ok]
    polished_but_flawed = [o for o in submitted if o.polished_but_flawed]

    print(f"  试验数     : {n}")
    print(f"  投稿数     : {len(submitted)} ({len(submitted)/n:.1%})")
    print(f"  干净（新颖+有效）: {len(polished_ok)} ({len(polished_ok)/n:.1%})")
    print(f"  润色但有缺陷    : {len(polished_but_flawed)} ({len(polished_but_flawed)/n:.1%})")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 沙箱隔离要求

| 隔离级别 | 说明 | 适用场景 |
|---------|------|---------|
| Docker 容器 | 文件系统 + 进程隔离 | 最低要求 |
| seccomp | 系统调用过滤 | 中等安全 |
| gVisor | 用户空间内核 | 高安全 |
| 网络隔离 | 无出站连接 | 防止数据泄露 |

### 4.2 人工审查门控

AI Scientist v2 的操作控制是主要安全层：
- **沙箱隔离**——LLM 代码不能逃逸
- **人工审查**——所有投稿前必须人工审查
- **披露**——向程序委员会披露论文来源

---

## 5. 工程最佳实践

### 5.1 研究循环设计原则

| 原则 | 说明 |
|------|------|
| 呈现质量差距是真实风险 | VLM 润色可以掩盖实验缺陷 |
| 评估必须触及底层主张 | 不能停在图表上 |
| 沙箱是最低要求 | LLM 代码可以做任何进程允许的事 |
| 人工审查是最后一道防线 | 自动评估最弱时，人工审查最重要 |

### 5.2 中文场景特别建议

- **中文研究论文的文献新颖性检查更难**——中文文献数据库覆盖不全，误标率可能更高
- **VLM 图表润色在中文场景中同样危险**——中文学术出版物对图表质量有高要求，润色差距更隐蔽
- **沙箱隔离在中文云环境中同样重要**——Docker、Kubernetes 网络策略是通用的

---

## 6. 常见错误

### 错误 1：信任呈现质量

**现象：** VLM 润色后的图表看起来完美。团队认为论文质量很高。但检查发现实验有编码错误，图表掩盖了缺陷。

**原因：** 呈现质量 ≠ 研究质量。VLM 润色可以让有缺陷的研究看起来完美。

**修复：** 评估必须触及底层主张——检查实验代码、检查数据、检查结论是否由数据支持。不能停在图表上。

### 错误 2：没有沙箱隔离

**现象：** LLM 生成的代码在宿主机上运行。代码执行了意外的网络请求或文件操作。

**原因：** LLM 代码可以做任何进程允许的事。

**修复：** Docker 是最低要求。seccomp/gVisor 提供更强隔离。网络出站连接应该被禁止。

### 错误 3：将工作坊接受等同于系统可靠

**现象：** "AI Scientist v2 的论文通过了 ICLR 评审"→ "系统可以做研究了"。

**原因：** 工作坊论文是较低的门槛。同行评审有噪声。一篇成功是概念验证，不是可靠性声明。

**修复：** 区分概念验证和可靠性声明。Beel 等人的独立评估发现 42% 的实验失败——这才是系统可靠性的更准确度量。

---

## 7. 面试考点

### Q1：AI Scientist v2 的七步研究循环是什么？（难度：⭐）

**参考答案：**
（1）假设生成 → （2）文献新颖性检查 → （3）实验计划 → （4）代码执行（沙箱） → （5）VLM 图表润色 → （6）论文撰写 → （7）内部评审 → 投稿（可选）

每一步都有可配置的失败概率。Beel 等人的数据：42% 实验失败、25% 新颖性误标、70% VLM 润色掩盖缺陷。

### Q2：什么是"呈现质量差距"？为什么它危险？（难度：⭐⭐）

**参考答案：**
VLM 图表润色产出了出版级视觉效果，但底层实验可能有缺陷。呈现质量超过研究质量——论文看起来完美但实验不可靠。

危险的原因：**产生有说服力的输出但不做有说服力的研究的系统，比明显失败的系统更危险。** 明显失败的系统会被拒绝；润色后的论文可能被接受并传播错误信息。评估必须触及底层主张，不能停在图表上。

### Q3：AlphaEvolve、DGM、AI Scientist v2 的评估器严格性如何对比？（难度：⭐⭐⭐）

**参考答案：**

| 系统 | 评估器 | 严格性 | 失败模式 |
|------|--------|-------|---------|
| AlphaEvolve | 单元测试 + 硬件基准 | 强 | 受评估器约束 |
| DGM | SWE-bench | 中 | 奖励黑客 |
| AI Scientist v2 | 同行评审（弱） | 弱 | 实验失败、误标、润色掩盖 |

AI Scientist v2 的评估器最弱（同行评审是有噪声的人工过程），输出面最宽（文本+代码+图表），到公共产物的路径最短。操作控制（沙箱、审查、披露）承担了大部分安全工作。

### Q4：为什么 Sakana 建议 Docker 隔离？沙箱逃逸的风险是什么？（难度：⭐⭐）

**参考答案：**
AI Scientist v2 执行 LLM 生成的代码。代码可以做任何进程允许的事——网络请求、文件操作、进程生成。

Docker 提供文件系统和进程隔离。但 Docker 不是完美的——容器逃逸是已知的攻击向量。更强的隔离（seccomp 系统调用过滤、gVisor 用户空间内核）提供额外安全层。

对于多天自主运行，还需要：网络出站禁止、资源限制（CPU/内存/磁盘）、进程数限制。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| AI Scientist v1 | "Sakana 的模板化研究智能体" | 在固定脚手架内填充实验 |
| AI Scientist v2 | "无模板研究智能体" | 智能体树搜索 + VLM 图表润色 |
| 智能体树搜索 | "分支研究智能体" | 并行扩展多个实验计划；内部评判器剪枝 |
| VLM 润色 | "图表美化" | 多模态模型读取图表并重写以提高清晰度 |
| 文献检索 | "新颖性检查" | 搜索先前工作确认想法新颖——已记录会误标 |
| 润色掩盖 | "漂亮的论文，破碎的研究" | 呈现质量超过实验质量；掩盖弱点 |
| 沙箱逃逸 | "LLM 代码逃出" | 智能体执行的代码做了循环设计者未预料的事 |

---

## 📚 小结

AI Scientist v2 运行完整的研究循环：假设→实验→图表→论文→评审。它是第一个生成论文通过 ICLR 工作坊评审的系统。但独立评估发现 42% 实验失败、25% 新颖性误标、VLM 润色掩盖缺陷。"呈现质量差距"意味着漂亮的论文可能隐藏破碎的研究——比明显失败更危险。沙箱隔离是最低要求，人工审查是最后一道防线。

下一课：自动化对齐研究——当智能体开始研究如何对齐自己。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。多少比例的循环运行产生"干净"论文？多少比例产生实验缺陷被图表润色掩盖的论文？

2. **【实验】** 用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 和 `--experiment-failure 0.60 --novelty-mislabel 0.40` 重跑。润色但有缺陷的比例如何变化？

3. **【阅读】** 阅读 Sakana AI Scientist v2 仓库 README 中的沙箱要求。命名两个你在多天自主运行中会额外应用的限制（Docker 之外）。

4. **【设计】** 设计一个能捕获"润色后但实验有缺陷"论文的额外评估器。

5. **【思考】** 提出一个比"博士读每篇论文"更好的研究智能体输出的人工审查协议。识别瓶颈并围绕它设计。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 研究循环模拟器 | `code/main.py` | 七步研究循环 + 呈现质量差距演示 |
| 技能提示词 | `outputs/skill-ai-scientist-sandbox-review.md` | 研究智能体输出的双门审查清单 |

---

## 📖 参考资料

1. [论文] Yamada et al. (2025). "The AI Scientist-v2". https://arxiv.org/abs/2504.08066 — 论文
2. [博客] Sakana. "AI Scientist v2 Nature Publication". https://sakana.ai/ai-scientist-nature/ — 供应商摘要
3. [论文] Beel et al. (2025). "Independent Evaluation of The AI Scientist". https://arxiv.org/abs/2502.14297 — 独立评估数据
4. [论文] Sakana AI Scientist v1. https://arxiv.org/abs/2408.06292 — 模板化前身
5. [博客] Anthropic. "Measuring AI Agent Autonomy". https://www.anthropic.com/research/measuring-agent-autonomy — 开放式研究智能体的更广泛框架

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
