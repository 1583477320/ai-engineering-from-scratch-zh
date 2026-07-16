# 记忆块与睡眠时计算（Letta）

> MemGPT 在 2024 年更名为 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散功能记忆块，以及主智能体空闲时异步整合记忆的睡眠时智能体。这是如何将记忆扩展到单次对话之外的方式。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 Letta 使用的三个记忆层（核心、召回、归档）及其各自角色
- [ ] 实现功能记忆块——可编辑的结构化记忆
- [ ] 设计睡眠时计算——异步记忆整合
- [ ] 理解记忆压缩策略——什么信息保留、什么信息压缩

---

## 1. 问题

MemGPT 解决了"上下文窗口有限"的问题，但还不够。现实应用需要：

- **功能记忆**：不只是文本记录，而是结构化信息（用户偏好、待办事项）
- **异步整理**：在不活跃时整合记忆——就像人类睡觉时整理记忆
- **记忆块编辑**：模型可以直接更新记忆块，而不是重写整个上下文

Letta 2026 的两个新概念：**功能记忆块**和**睡眠时计算**。

---

## 2. 概念

### 2.1 三层记忆架构

| 层次 | 存储 | 内容 | 操作 |
|------|------|------|------|
| **核心记忆** | 主上下文 | 关键事实（用户名、偏好） | append/replace |
| **召回记忆** | 可搜索存储 | 交互历史摘要 | search |
| **归档记忆** | 向量数据库 | 原始对话、文档 | search/insert |

### 2.2 功能记忆块

```
记忆块: {
  "type": "user_preference",
  "key": "编程语言",
  "value": "Python",
  "confidence": 0.95,
  "last_updated": "2024-01-15"
}
```

与纯文本记忆不同——功能记忆块有类型、键值对、置信度。

### 2.3 睡眠时计算

```
主智能体活跃时：
  - 处理用户交互
  - 将新信息存入临时缓冲区

主智能体空闲时（"睡眠"）：
  - 整合临时缓冲区到核心记忆
  - 压缩冗余信息
  - 更新记忆权重
  - 清理过期记忆
```

### 2.4 记忆优先级

```
优先级 1: 安全/权限信息 → 核心记忆（永不压缩）
优先级 2: 用户偏好 → 核心记忆（低概率压缩）
优先级 3: 最近交互 → 短期记忆（自动压缩）
优先级 4: 历史记录 → 长期记忆（可搜索）
```

---

## 3. 从零实现

### Step 1：功能记忆块

```python
class MemoryBlock:
    """可编辑的功能记忆块。"""
    def __init__(self, block_type, key, value, confidence=1.0):
        self.type = block_type
        self.key = key
        self.value = value
        self.confidence = confidence
        self.created_at = None
        self.updated_at = None

    def update(self, new_value, confidence=1.0):
        self.value = new_value
        self.confidence = max(self.confidence, confidence)

    def __repr__(self):
        return f"Block({self.type}: {self.key}={self.value}, conf={self.confidence:.2f})"
```

### Step 2：睡眠时计算

```python
class SleepTimeCompute:
    """睡眠时记忆整合。"""
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.temp_buffer = []

    def on_interaction(self, event):
        """主智能体活跃时——记录到临时缓冲区。"""
        self.temp_buffer.append(event)

    def consolidate(self):
        """主智能体空闲时——整合记忆。"""
        # 1. 分析临时缓冲区
        if not self.temp_buffer:
            return

        # 2. 提取关键信息
        key_facts = [e for e in self.temp_buffer if e.get("priority", 0) > 0.5]

        # 3. 更新核心记忆
        for fact in key_facts:
            self.memory_manager.add_core(fact["content"])

        # 4. 压缩旧记忆
        if len(self.memory_manager.core_memory) > 20:
            oldest = self.memory_manager.core_memory.pop(0)
            self.memory_manager.add_archival(f"历史: {oldest}")

        # 5. 清空临时缓冲区
        self.temp_buffer = []
        print(f"  睡眠整合: {len(key_facts)} 条关键信息，核心记忆 {len(self.memory_manager.core_memory)} 条")

    def status(self):
        return f"临时缓冲区: {len(self.temp_buffer)} 条"
```

### Step 3：记忆压缩策略

```python
def compress_memory(core_memory, max_items=15):
    """压缩核心记忆——保留最重要的信息。"""
    if len(core_memory) <= max_items:
        return core_memory
    # 按置信度排序，保留最高的
    sorted_memories = sorted(core_memory, key=lambda m: m.get("confidence", 0.5), reverse=True)
    return sorted_memories[:max_items]
```

---

## 4. 工具

### 4.1 Letta 框架

Letta（原 MemGPT）提供了完整的记忆系统：
- 核心记忆（core_memory_append/replace）
- 归档记忆（archival_memory_insert/search）
- 对话历史（conversation_search）

### 4.2 向量数据库

| 数据库 | 特点 |
|--------|------|
| Chroma | 轻量级、嵌入式 |
| Pinecone | 托管、可扩展 |
| Weaviate | 混合搜索 |
| Qdrant | 高性能 |

---

## 5. 工程最佳实践

### 5.1 记忆管理原则

- **最小化核心记忆**：只保留最关键的信息
- **主动压缩**：空闲时整合冗余记忆
- **定期清理**：过期信息移入归档或删除
- **优先级标记**：关键信息永不压缩

### 5.2 踩坑经验

- **核心记忆膨胀**：没有压缩限制→上下文溢出
- **睡眠时整合延迟**：整合周期太长→记忆过时
- **记忆冲突**：新旧信息矛盾→需要冲突解决策略

---

## 6. 常见错误

### 错误 1：所有信息都存入核心记忆

**现象：** 核心记忆膨胀→上下文窗口溢出。

**修复：** 只有关键信息（用户偏好、权限、关键事实）才入核心记忆。

### 错误 2：睡眠时整合被跳过

**现象：** 临时缓冲区无限增长→内存泄漏。

**修复：** 定期触发整合——即使主智能体活跃也要异步处理。

---

## 7. 面试考点

### Q1：Letta 的三层记忆架构是什么？（难度：⭐⭐）

**参考答案：**
核心记忆：存放在主上下文中，包含关键事实（用户偏好、权限），容量有限但访问快速。召回记忆：存放在可搜索存储中，包含近期交互摘要——需要时检索。归档记忆：存放在向量数据库中，包含历史记录——长期保留但不常访问。三层对应操作系统的 RAM → 缓存 → 磁盘。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 记忆块 | "结构化记忆" | 可编辑的功能记忆单元——有类型、键值、置信度 |
| 睡眠时计算 | "后台记忆整理" | 主智能体空闲时异步整合和压缩记忆 |
| 核心记忆 | "关键信息" | 放在主上下文中的最重要的记忆——永不压缩 |
| 召回记忆 | "可搜索记忆" | 存储在可搜索存储中的记忆——需要时检索 |

---

## 📚 小结

Letta 的三层记忆：核心（关键事实）、召回（近期摘要）、归档（历史记录）。功能记忆块让模型直接编辑结构化记忆。睡眠时计算异步整合记忆。记忆压缩策略确保核心记忆不膨胀。

---

## ✏️ 练习

1. **【实现】** 构建功能记忆块管理器——支持插入、搜索、编辑、压缩
2. **【设计】** 为一个客服智能体设计记忆架构——什么信息放核心记忆？什么放归档？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 记忆管理器 | `code/main.py` | 功能记忆块 + 睡眠时整合 |

---

## 📖 参考资料

1. [论文] Packer et al. "MemGPT: Towards LLMs as Operating Systems". 2023.
2. [项目] Letta: https://github.com/letta-ai/letta
3. [论文] Fang et al. "MemoryBank: Enhancing Large Language Models with Long-Term Memory". AAAI, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
