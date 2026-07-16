# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具跟踪不是。MemGPT（Packer 等人，2023）将此框架化为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，智能体在它们之间分页。这是每个 2026 年记忆系统继承的模式。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、06（工具使用）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 MemGPT 的操作系统类比：主上下文=RAM，外部存储=磁盘，记忆工具=分页
- [ ] 实现记忆管理——自动压缩、摘要、检索
- [ ] 设计记忆优先级策略——什么信息保留、什么信息压缩
- [ ] 理解长期记忆对智能体的重要性

---

## 1. 问题

LLM 的上下文窗口是有限的（如 128K token）。但对话、文档和工具调用记录可能远超这个限制。智能体需要**长期记忆**——记住之前说过什么、做过什么、学到了什么。

**核心挑战：** 如何在有限的上下文窗口中管理无限的信息？

---

## 2. 概念

### 2.1 MemGPT 的操作系统类比

```
主上下文（RAM）  ←→  外部存储（磁盘）
     ↑                      ↑
   读/写                  读/写
     ↑                      ↑
智能体（CPU）     ←→   记忆工具（I/O 总线）
```

| 操作系统概念 | MemGPT 对应 |
|------------|------------|
| RAM | 主上下文窗口（有限） |
| 磁盘 | 向量数据库/文件（无限） |
| 分页 | 将信息在主上下文和外部存储间移动 |
| 进程调度 | 智能体决定何时读/写 |

### 2.2 记忆层次

| 层次 | 存储位置 | 内容 | 容量 |
|------|---------|------|------|
| **工作记忆** | 主上下文 | 当前对话 | 有限（128K token） |
| **短期记忆** | 可搜索存储 | 近期交互摘要 | 中等 |
| **长期记忆** | 向量数据库 | 历史知识 | 无限 |

### 2.3 记忆工具

```python
memory_tools = {
    "core_memory_append": "在核心记忆中添加新信息",
    "core_memory_replace": "替换核心记忆中的旧信息",
    "archival_memory_insert": "插入长期记忆",
    "archival_memory_search": "搜索长期记忆",
    "conversation_search": "搜索历史对话",
}
```

### 2.4 自动记忆管理

```
新信息进入 → 检查是否需要存储
  ├── 高优先级（用户偏好、事实）→ 存入核心记忆
  ├── 中优先级（对话摘要）→ 存入短期记忆
  └── 低优先级（细节）→ 丢弃或存入长期记忆
```

---

## 3. 从零实现

### Step 1：记忆管理器

```python
class MemoryManager:
    """简化版记忆管理器。"""
    def __init__(self, max_context=2048):
        self.max_context = max_context
        self.core_memory = []  # 核心记忆
        self.archival = []     # 长期记忆
        self.current_tokens = 0

    def add_core(self, info):
        """添加核心记忆。"""
        self.core_memory.append(info)
        self.current_tokens += len(info) // 4  # 粗略 token 估算

    def add_archival(self, info):
        """添加长期记忆。"""
        self.archival.append(info)

    def search_archival(self, query):
        """搜索长期记忆。"""
        return [a for a in self.archival if any(k in a for k in query.split()[:3])]

    def get_context(self):
        """获取当前上下文——自动截断。"""
        context = "\n".join(self.core_memory)
        # 如果超过限制，压缩旧记忆
        while len(context) > self.max_context * 4:
            self.archival.append(self.core_memory.pop(0))
            context = "\n".join(self.core_memory)
        return context

    def status(self):
        return f"核心记忆: {len(self.core_memory)} 条, 长期记忆: {len(self.archival)} 条, 当前token: {self.current_tokens}"


if __name__ == "__main__":
    print("MemGPT 记忆管理器演示\n")
    mm = MemoryManager(max_context=500)
    mm.add_core("用户喜欢简洁回答")
    mm.add_core("用户是 Python 开发者")
    mm.add_core("上次讨论了 Transformer 架构")
    mm.add_archival("2023-10-01: 讨论了 RAG 管道")
    mm.add_archival("2023-11-01: 讨论了微调策略")
    print(mm.status())
    print(f"搜索 'RAG': {mm.search_archival('RAG')}")
    print(f"上下文: {mm.get_context()[:80]}...")
