# 仓库记忆与持久状态

> 聊天历史是可变的。仓库是持久的。工作台将智能体状态存储在版本化文件中，以便下一个会话、下一个智能体和下一个审查员都从同一个真相源读取。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 32（最小工作台）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义什么属于仓库记忆（持久）和什么属于聊天历史（临时）
- [ ] 实现版本化的状态文件——支持回滚和审查
- [ ] 理解为什么仓库比聊天历史更可靠
- [ ] 设计状态文件的 Schema 和迁移策略

---

## 1. 问题

聊天历史随着对话结束而消失。但智能体的状态——学到的知识、完成的任务、配置的偏好——应该跨会话持久化。**仓库记忆**将智能体状态存储在版本化的仓库文件中。

---

## 2. 概念

### 2.1 仓库记忆 vs 聊天历史

| 方面 | 聊天历史 | 仓库记忆 |
|------|---------|---------|
| 存储 | LLM 上下文 | 文件系统 |
| 持久性 | 会话级 | 永久 |
| 共享 | 当前对话 | 所有智能体 |
| 版本控制 | 无 | Git 可追踪 |

### 2.2 什么存仓库记忆

```python
REPO_MEMORY = {
    "user_preferences": {"language": "Python", "framework": "PyTorch"},
    "completed_tasks": ["数据管道搭建", "模型训练"],
    "learned_patterns": {"编码风格": "PEP 8"},
}

CHAT_HISTORY = {
    "current_dialogue": "用户: 帮我建一个 Transformer 模型 ...",
    "temp_state": {"current_file": "main.py"},
}
```

### 2.3 版本化状态

```python
import json
import hashlib

class VersionedState:
    """版本化的状态文件。"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self.version = 0

    def save(self):
        self.version += 1
        self.data["_version"] = self.version
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    def load(self):
        try:
            with open(self.filepath) as f:
                self.data = json.load(f)
                self.version = self.data.get("_version", 0)
        except FileNotFoundError:
            pass
```

---

## 3. 从零实现

### Step 1：仓库记忆管理器

```python
class RepoMemory:
    """仓库记忆管理器。"""
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.memory = {}

    def store(self, key, value):
        """存储到仓库记忆。"""
        self.memory[key] = value
        self._persist(key, value)

    def retrieve(self, key):
        """从仓库记忆检索。"""
        return self.memory.get(key)

    def _persist(self, key, value):
        """持久化到文件。"""
        with open(f"{self.repo_path}/{key}.json", "w") as f:
            json.dump(value, f, indent=2)

    def load_all(self):
        """加载所有记忆文件。"""
        import glob, os
        for f in glob.glob(f"{self.repo_path}/*.json"):
            key = os.path.basename(f).replace(".json", "")
            with open(f) as fp:
                self.memory[key] = json.load(fp)
```

---

## 4. 工具

### 4.1 存储对比

| 存储 | 持久性 | 版本控制 | 适用 |
|------|--------|---------|------|
| 聊天历史 | 会话 | 无 | 当前对话 |
| 仓库记忆 | 永久 | Git | 跨会话 |
| 向量数据库 | 永久 | 无 | 语义搜索 |

---

## 5. 工程最佳实践

- **聊天历史**存当前对话——用完就丢
- **仓库记忆**存学到的知识——跨会话
- **版本化**支持回滚

---

## 6. 常见错误

### 错误 1：把聊天历史当仓库记忆

**现象：** 会话结束后所有状态丢失。

**修复：** 关键信息（偏好、已完成任务）存仓库记忆。当前对话存聊天历史。

---

## 7. 面试考点

### Q1：仓库记忆和聊天历史的区别是什么？（难度：⭐⭐）

**参考答案：**
聊天历史是临时的——当前对话的上下文。仓库记忆是持久的——学到的知识、偏好、已完成任务。一个存工具运行结果，一个存经验教训。一个会话级，一个永久。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 仓库记忆 | "永久存储" | 仓库中版本化的状态文件——跨会话持久 |
| 聊天历史 | "会话上下文" | 当前对话的 LLM 上下文——会话结束消失 |
| 版本化状态 | "可回滚" | 状态文件带有版本号和 Git 历史 |

---

## 📚 小结

仓库记忆持久、版本化、跨会话共享。聊天历史临时、会话级、易失去。关键信息存仓库，当前对话存聊天历史。仓库记忆支持 Git 版本控制和回滚。

---

## ✏️ 练习

1. **【实现】** 实现版本化的状态文件——支持保存和加载
2. **【设计】** 为一个智能体设计仓库记忆 Schema——哪些信息该存储？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 仓库记忆 | `code/main.py` | 持久化 + 版本化 |

---

## 📖 参考资料

1. [文档] Claude Code Memory
2. [文档] Letta: https://github.com/letta-ai/letta
3. [论文] MemGPT. 2023.
