# 上下文工程

> 提示词是上下文的一部分。上下文窗口是模型的全部工作记忆。把正确的信息放进窗口——太多信息稀释重点，太少信息丢失关键上下文。这是工程，不是运气。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-03（提示词工程）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释上下文窗口的工作原理——为什么信息顺序和位置影响生成质量
- [ ] 实现上下文压缩技术——减少 token 消耗同时保留关键信息
- [ ] 设计对话历史管理策略——滑动窗口、摘要、重要信息保留
- [ ] 优化长上下文任务的性能和成本

---

## 1. 问题

你让模型分析一篇 10,000 词的文档。你把整个文档塞进提示词。模型给出了正确的摘要——但遗漏了文档中段提到的一个关键细节。这不是模型不聪明，是信息太多——注意力机制会在长上下文中稀释关键信息。

上下文工程解决这个问题：**正确选择、排序和组织进入模型窗口的信息。**

---

## 2. 概念

### 2.1 上下文窗口的特性

| 特性 | 影响 |
|------|------|
| **有限长度** | GPT-4: 128K, Claude: 200K |
| **位置偏差** | 开头和结尾的信息权重更高 |
| **注意力稀释** | 信息过多时注意力分散 |
| **成本** | 输入 token 通常按 1/3 价格计费 |

### 2.2 信息位置策略

```
[系统消息（角色、规则、约束）] ← 持久，最重要的放在前面
[长文档/上下文]                ← 可以放中间
[对话历史]                     ← 按时间排序
[当前查询]                     ← 放最后，最近的最重要
```

### 2.3 上下文压缩技术

| 技术 | 描述 | 效果 |
|------|------|------|
| **摘要压缩** | 用小模型先摘要长文档 | 减少 80% token |
| **选择性保留** | 只保留与查询相关的段落 | 减少 70% token |
| **递归摘要** | 超长文档分段摘要再合并 | 支持任意长度 |

### 2.4 对话历史管理

```
方法 1：滑动窗口（保留最近 N 条）
  [消息1, 消息2, ..., 消息N] → 固定窗口

方法 2：摘要 + 最近
  [LLM摘要: 前10轮的核心内容] + [最近5轮]

方法 3：重要信息提取
  [LLM提取: 关键决策、待办事项] + [最近消息]
```

---

## 3. 从零实现

### Step 1：上下文窗口管理

```python
class ContextWindow:
    """上下文窗口管理器。"""
    
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.system = ""
        self.history = []
    
    def set_system(self, message):
        self.system = message
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
    
    def get_messages(self, current_query):
        """构建最终消息列表——按重要性排序。"""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        
        # 历史消息——最近的优先
        for msg in self.history[-10:]:  # 保留最近10条
            messages.append(msg)
        
        # 当前查询
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def estimate_tokens(self, messages):
        """粗略估计 token 数（1中文字≈1.5 token）。"""
        return sum(len(m["content"]) * 1.5 for m in messages)
```

### Step 2：摘要压缩

```python
def summarize_document(document, chunk_size=2000, overlap=200):
    """将长文档分块摘要。"""
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        chunks.append(chunk)
    
    # 对每块独立摘要
    summaries = []
    for chunk in chunks:
        summary = f"[摘要] {chunk[:200]}..."
        summaries.append(summary)
    
    return "\n".join(summaries)
```

---

## 4. 工具

### 4.1 LangChain 的文档加载器

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("large_document.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(loader.load_documents())
```

### 4.2 对话管理库

```python
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(max_token_limit=4000)
```

---

## 6. 工程最佳实践

### 6.1 上下文窗口分配

| 区域 | Token 预算 | 说明 |
|------|-----------|------|
| 系统消息 | 200-500 | 角色、规则、格式 |
| 文档/检索内容 | 50-60% | RAG 结果或相关文档 |
| 对话历史 | 20-30% | 最近的 5-10 轮 |
| 当前查询 | 剩余 | 确保不被截断 |

### 6.2 中文场景建议

- 中文字符约 1.5 token/字，比英文更紧凑
- 长文档摘要时用中文 LLM 效果更好

### 6.3 踩坑经验

- **信息放错位置**：关键信息应放最前或最后（位置偏差效应）
- **上下文太长**：模型注意力稀释，关键信息被忽略
- **对话历史无上限**：超长历史导致 token 浪费

---

## 7. 面试考点

### Q1：为什么 LLM 在长上下文中会"遗忘"关键信息？（难度：⭐⭐）

**参考答案：**
Transformer 的自注意力机制计算每个位置与其他所有位置的相关性。随着序列变长，每个位置的注意力权重被更多位置"稀释"。更重要的是，LLM 的训练目标是 next-token prediction——开头和结尾的信息对预测中间 token 的贡献较小，导致这些位置的信息编码更弱。这就是"Lost in the Middle"现象——长文档中间的信息最容易被忽略。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 上下文窗口 | "模型的短期记忆" | 模型一次能处理的最大 token 数 |
| Lost in the Middle | "中间的信息被遗忘" | 长上下文中中间部分的信息容易被忽略 |
| 摘要压缩 | "长文变短文" | 用 LLM 先对长文档做摘要，再用摘要代替原文 |

---

## 📚 小结

上下文工程是 LLM 应用的基础设施——正确选择和组织信息让模型产生更好输出。关键是：重要信息放首尾，长文档先摘要，对话历史用滑动窗口+摘要。上下文成本与质量需要权衡。

---

## ✏️ 练习

1. **【实验】** 对比在开头和末尾放置关键信息的生成质量差异
2. **【实现】** 实现滑动窗口对话管理器，维护最近 N 轮对话

---

## 📖 参考资料

1. [论文] Liu et al. "Lost in the Middle: How Language Models Use Long Contexts". NeurIPS, 2023. https://arxiv.org/abs/2307.03172
2. [博客] Anthropic Context Engineering: https://docs.anthropic.com/en/docs/build-with-claude/context-engineering
