# Skills 与智能体 SDK——Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说"有哪些工具可用"。Skills 说"如何完成任务"。2026 年的技术栈将两者分层。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 和渐进式披露的形式发布。OpenAI 的 Apps SDK 是 MCP 加小部件元数据。AGENTS.md（现在出现在 6 万多个仓库中）位于仓库根目录，作为项目级智能体上下文。本课命名每个层级覆盖的内容，并构建一个可以跨智能体传递的最小 SKILL.md + AGENTS.md 套件。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分三个层级：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）
- [ ] 设计 AGENTS.md——为智能体提供项目级上下文
- [ ] 实现 SKILL.md 的渐进式披露——从简要到详细的信息层级
- [ ] 理解 MCP + Skills 如何互补——工具 + 方法论

---

## 1. 问题

MCP 告诉智能体"有哪些工具可用"——但不告诉它"什么时候用哪个工具"、"用什么策略"。Skills 补充了这个缺失——它们是"如何完成任务"的知识。

三个层级的关系：
```
AGENTS.md → "这个项目是做什么的"（项目上下文）
SKILL.md → "如何完成特定任务"（方法论）
MCP → "可以调用哪些工具"（工具接口）
```

---

## 2. 概念

### 2.1 三层架构

| 层级 | 位置 | 内容 | 更新频率 |
|------|------|------|---------|
| **AGENTS.md** | 仓库根目录 | 项目目标、架构、约定 | 低 |
| **SKILL.md** | 任务文件或目录 | 具体任务的步骤和注意事项 | 中 |
| **MCP Server** | 独立进程 | 工具定义和执行 | 低 |

### 2.2 AGENTS.md 示例

```markdown
# 项目说明
这是一个 Python 数据管道项目。

## 架构
- 数据源：S3 JSONL 文件
- 处理：Spark + PySpark
- 输出：Parquet 格式，写入数据仓库

## 约定
- 使用 Black 格式化代码
- 中文注释
- 提交前运行测试

## 智能体使用指南
- 构建新管道时参考 `pipelines/` 目录
- 测试前查看 `tests/` 目录
- 部署参考 `deploy/` 目录
```

### 2.3 SKILL.md 示例

```markdown
# 创建数据管道

## 前提条件
- 数据源已配置
- PySpark 环境已安装

## 步骤
1. 在 `pipelines/` 目录创建新的 Python 文件
2. 定义数据模式（输入/输出）
3. 实现 `transform()` 函数
4. 添加单元测试
5. 运行 `make test` 验证

## 注意事项
- 遵循现有代码风格
- 处理边界情况（空输入、格式错误）
- 添加日志记录
```

### 2.4 Anthropic Skills

Anthropic 的 Agent Skills 是开放标准——以 SKILL.md 文件形式发布，支持渐进式披露：

```
Level 1: 标题和一句话描述（智能体决定是否需要）
Level 2: 前提条件和步骤概览（智能体判断是否相关）
Level 3: 详细步骤和注意事项（智能体深入执行）
```

---

## 3. 从零实现

### Step 1：AGENTS.md 生成器

```python
def generate_agents_md(project_name, description, architecture, conventions):
    """生成 AGENTS.md 文件。"""
    return f"""# {project_name}

{description}

## 架构
{architecture}

## 约定
{conventions}

## 智能体使用指南
- 参考各子目录中的 SKILL.md 文件
- 测试前运行 `make test`
- 部署参考 `deploy/` 目录
"""
```

### Step 2：SKILL.md 生成器

```python
def generate_skill_md(title, prerequisites, steps, notes=None):
    """生成 SKILL.md 文件。"""
    content = f"""# {title}

## 前提条件
{chr(10).join('- ' + p for p in prerequisites)}

## 步骤
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps))}

"""
    if notes:
        content += f"## 注意事项\n{chr(10).join('- ' + n for n in notes)}"
    return content
```

---

## 4. 工具

### 4.1 Anthropic Agent Skills

```python
# SKILL.md 文件位于智能体可访问的目录中
# 渐进式披露：Level 1（标题）→ Level 2（概览）→ Level 3（详细）
```

### 4.2 OpenAI Apps SDK

OpenAI 的 Apps SDK 是 MCP 加上小部件元数据——允许工具返回交互式 UI。

### 4.3 AGENTS.md

```bash
# 在仓库根目录创建 AGENTS.md
echo "# 项目说明
架构: Python + FastAPI
约定: Black 格式化, 中文注释" > AGENTS.md
```

---

## 5. 工程最佳实践

### 5.1 三层架构选择

| 场景 | 使用 | 原因 |
|------|------|------|
| 工具调用 | MCP | 标准化工具接口 |
| 任务方法论 | SKILL.md | 可复用的步骤指南 |
| 项目上下文 | AGENTS.md | 智能体理解项目 |
| 三者结合 | AGENTS.md + SKILL.md + MCP | 最佳实践 |

### 5.2 踩坑经验

- **AGENTS.md 太长**：包含过多无关信息——智能体注意力分散
- **SKILL.md 过于详细**：每步都写——维护成本高
- **MCP 工具和 SKILL 重复**：工具执行操作，Skill 说明何时/如何使用——不冲突

---

## 6. 常见错误

### 错误 1：AGENTS.md 放错位置

**现象：** 智能体找不到项目上下文。

**修复：** AGENTS.md 必须放在仓库根目录——智能体只在根目录查找。

### 错误 2：SKILL.md 缺少前提条件

**现象：** 智能体执行任务时缺少必要的环境——失败。

**修复：** SKILL.md 必须列出所有前提条件——环境、依赖、权限。

---

## 7. 面试考点

### Q1：AGENTS.md、SKILL.md 和 MCP 分别解决什么问题？（难度：⭐⭐）

**参考答案：**
AGENTS.md 解决"项目是什么"——智能体理解项目目标、架构、约定。SKILL.md 解决"怎么做"——具体任务的步骤和注意事项。MCP 解决"用什么"——工具定义和调用接口。三者分层互补：AGENTS.md 提供全局上下文，SKILL.md 提供任务方法，MCP 提供工具能力。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| AGENTS.md | "项目说明书" | 位于仓库根目录的智能体上下文文件——项目目标、架构、约定 |
| SKILL.md | "任务操作手册" | 特定任务的步骤和注意事项——支持渐进式披露 |
| MCP | "工具接口" | 标准化工具调用协议——智能体与工具交互 |
| 渐进式披露 | "分层加载" | 只加载当前需要的信息层级——Level 1→2→3 |

---

## 📚 小结

三层架构：AGENTS.md（项目上下文）→ SKILL.md（任务方法论）→ MCP（工具接口）。MCP 说"有哪些工具"，Skills 说"如何用"，AGENTS.md 说"项目是什么"。三者分层互补。

---

## ✏️ 练习

1. **【实现】** 为一个 Python 项目生成 AGENTS.md——包含架构、约定、使用指南
2. **【设计】** 为一个数据管道任务编写 SKILL.md——包含前提条件和步骤

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| AGENTS.md + SKILL.md | `code/main.py` | 三层架构生成器 |

---

## 📖 参考资料

1. [文档] Anthropic Agent Skills: https://docs.anthropic.com/en/docs/agents/skills
2. [文档] AGENTS.md: https://github.com/agents/agents-md-spec
3. [文档] OpenAI Apps SDK: https://platform.openai.com/docs/apps

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
