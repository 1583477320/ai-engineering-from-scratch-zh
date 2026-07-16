# 技能库与终身学习（Voyager）

> Voyager（Wang 等人，TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的，并由环境反馈改进。这是 Claude Agent SDK 技能、skillkit 和 2026 年技能库模式的参考架构。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）、08（Letta Blocks）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 Voyager 的三个组件——自动课程、技能库、迭代提示——以及各自的角色
- [ ] 实现技能存储和检索——从库中找到最相关的技能
- [ ] 理解终身学习如何让智能体通过环境反馈改进技能
- [ ] 设计一个技能库系统——技能的存储、检索、组合、改进

---

## 1. 问题

传统的智能体每次从零开始推理——即使它已经成功解决过类似问题。Voyager 的答案：**将成功的解决方案存储为可检索的技能**——下次遇到类似问题时直接复用。

---

## 2. 概念

### 2.1 Voyager 三组件

| 组件 | 功能 | 类比 |
|------|------|------|
| **自动课程** | 选择当前学习任务 | 教科书目录 |
| **技能库** | 存储成功的解决方案 | 代码库 |
| **迭代提示** | 用环境反馈改进技能 | 代码审查 |

### 2.2 技能库结构

```python
class Skill:
    """技能——命名、描述、代码、成功次数。"""
    def __init__(self, name, description, code):
        self.name = name
        self.description = description
        self.code = code
        self.success_count = 0
        self.last_used = None

    def to_dict(self):
        return {"name": self.name, "description": self.description,
                "success_count": self.success_count}
```

### 2.3 技能生命周期

```
发现：环境反馈发现新任务
  ↓
生成：LLM 生成解决代码
  ↓
验证：在环境中执行验证
  ↓
存储：验证通过 → 存入技能库
  ↓
检索：类似任务 → 从库中检索技能
  ↓
复用：直接执行或微调后执行
  ↓
改进：失败时迭代改进
```

### 2.4 技能组合

```python
def compose_skills(skill_a, skill_b):
    """组合两个技能。"""
    return f"{skill_a.code}\n{skill_b.code}"
```

---

## 3. 从零实现

### Step 1：技能库

```python
class SkillLibrary:
    """技能库——存储、检索、管理技能。"""
    def __init__(self):
        self.skills = []

    def add_skill(self, name, description, code):
        skill = {"name": name, "description": description, "code": code, "success": 0, "uses": 0}
        self.skills.append(skill)
        return skill

    def search(self, task_description):
        """检索最相关的技能。"""
        results = []
        for skill in self.skills:
            # 简化：关键词匹配
            score = sum(1 for w in task_description.split() if w in skill["description"])
            if score > 0:
                results.append((skill, score))
        return sorted(results, key=lambda x: -x[1])

    def get_top_skill(self, task_description):
        """获取最相关的技能。"""
        results = self.search(task_description)
        return results[0][0] if results else None

    def update_success(self, skill):
        """更新技能成功计数。"""
        skill["success"] = skill.get("success", 0) + 1

    def status(self):
        return f"技能库: {len(self.skills)} 个技能"
```

### Step 2：自动课程

```python
class AutoCurriculum:
    """自动课程——选择当前学习任务。"""
    def __init__(self, skill_library):
        self.skill_library = skill_library
        self.tasks = []

    def generate_tasks(self):
        """生成任务列表。"""
        self.tasks = [
            {"difficulty": 1, "task": "在迷宫中找到出口"},
            {"difficulty": 2, "task": "收集5个金币"},
            {"difficulty": 3, "task": "击败BOSS"},
        ]
        return self.tasks

    def select_next(self):
        """选择当前学习任务。"""
        # 简化：选择最简单且未掌握的任务
        for task in self.tasks:
            existing = self.skill_library.search(task["task"])
            if not existing:
                return task
        return self.tasks[-1]  # 都掌握了，选最难的
```

### Step 3：技能检索和复用

```python
def use_skill(skill_library, task):
    """检索并复用技能。"""
    skill = skill_library.get_top_skill(task)
    if skill:
        print(f"复用技能: {skill['name']} (成功 {skill['success']} 次)")
        skill["uses"] = skill.get("uses", 0) + 1
        return skill["code"]
    else:
        print(f"未找到相关技能，需要从零学习")
        return None
```

---

## 4. 工具

### 4.1 Voyager

```python
# Voyager 通过 Minecraft 环境演示技能学习
# https://github.com/MineDojo/Voyager
```

### 4.2 技能库实现

| 库 | 功能 |
|------|------|
| LangChain Skills | 技能定义和检索 |
| Claude Agent SDK | 内置技能支持 |
| SkillKit | 可组合技能 |

---

## 5. 工程最佳实践

### 5.1 技能设计原则

- **原子性**：每个技能解决一个问题
- **可检索性**：技能描述要精确，便于检索
- **可组合性**：技能可以组合成更复杂的解决方案
- **可改进性**：技能可以通过环境反馈迭代改进

### 5.2 踩坑经验

- **技能描述不精确**：检索时找不到相关技能
- **技能库无限增长**：需要定期清理低成功率技能
- **技能过时**：环境变化后旧技能可能不适用

---

## 6. 常见错误

### 错误 1：技能描述太泛

**现象：** 检索时匹配到不相关的技能。

**修复：** 技能描述包含具体的场景、步骤和限制条件。

---

## 7. 面试考点

### Q1：Voyager 的三个组件是什么？（难度：⭐⭐）

**参考答案：**
(1) **自动课程**：选择当前学习任务——类似教材目录，按难度递增；(2) **技能库**：存储成功的解决方案——命名、描述、代码、成功次数；(3) **迭代提示**：用环境反馈改进技能——类似代码审查。技能库是核心——它让智能体可以复用之前学到的解决方案。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Voyager | "技能学习智能体" | 用可执行代码作为技能的终身学习架构 |
| 技能库 | "代码库" | 存储已验证的解决方案——命名、可检索、可组合 |
| 自动课程 | "难度递增" | 自动选择当前应学习的任务——从简单到复杂 |
| 终身学习 | "不断改进" | 智能体通过环境反馈持续改进已有技能 |

---

## 📚 小结

Voyager 三个组件：自动课程（选任务）、技能库（存方案）、迭代提示（改进技能）。技能库是核心——命名、可检索、可组合、可改进。这是 2026 年智能体技能库的参考架构。

---

## ✏️ 练习

1. **【实现】** 构建技能库系统——支持添加、搜索、更新技能成功计数
2. **【设计】** 为一个数据处理智能体设计技能库——存储常用的数据清洗/转换技能

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 技能库 | `code/main.py` | 技能存储+检索+自动课程 |

---

## 📖 参考资料

1. [论文] Wang et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models". TMLR, 2024.
2. [GitHub] Voyager: https://github.com/MineDojo/Voyager
3. [文档] Claude Agent Skills: https://docs.anthropic.com/en/docs/agents/skills

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
