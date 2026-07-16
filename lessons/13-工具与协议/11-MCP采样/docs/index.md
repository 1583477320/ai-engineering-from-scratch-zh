# MCP 采样——Server 请求的 LLM 补全与智能体循环

> 大多数 MCP Server 是呆板的执行者：接收参数、运行代码、返回内容。采样让 Server 反转方向：请求 Client 的 LLM 做出决策。这使得 Server 主持的智能体循环无需 Server 拥有任何模型凭证。SEP-1577（2025-11-25 合并）在采样请求中添加了工具，使循环可以包含更深层的推理。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、10（Resources 和 Prompts）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 `sampling/createMessage` 解决什么问题（无 Server 端 API 密钥的 Server 主持循环）
- [ ] 实现 Server 端采样请求——向 Client 的 LLM 发起生成请求
- [ ] 理解采样循环中的工具使用（SEP-1577）
- [ ] 设计 Server 主持的智能体循环架构

---

## 1. 问题

传统 MCP Server 是"呆板执行者"——接收参数、运行代码、返回内容。但有些任务需要 Server "思考"——比如：分析一段代码并给出改进建议。这需要 LLM 推理。

问题：Server 没有自己的 LLM——它只有代码。如何在不拥有 API 密钥的情况下让 Server 调用 LLM？

**采样（Sampling）** 解决了这个问题：Server 请求 Client 的 LLM 生成内容。

---

## 2. 概念

### 2.1 采样流程

```
Client (拥有 LLM) → Server
                        ↓
              Server 决定需要 LLM 生成
                        ↓
              Server 发送 sampling/createMessage
                        ↓
Client 使用自己的 LLM 生成内容 → 返回给 Server
                        ↓
Server 继续处理
```

### 2.2 采样 vs 直接 LLM 调用

| 方面 | 直接调用 | MCP 采样 |
|------|---------|---------|
| API 密钥 | Server 需要 | Client 拥有 |
| 模型选择 | Server 控制 | Client 控制 |
| 安全性 | Server 访问外部 API | Server 不接触凭证 |
| 灵活性 | 固定模型 | Client 可切换模型 |

### 2.3 SEP-1577：采样中的工具

2025-11-25 规范添加了在采样请求中使用工具的能力——使 Server 主持的智能体循环可以调用工具，实现更复杂的推理。

### 2.4 Server 主持的智能体循环

```
用户请求 → Server
    ↓
Server 分析请求 → 需要 LLM 推理
    ↓
Server → Client: sampling/createMessage
    ↓
Client 使用 LLM 生成 → 返回
    ↓
Server 根据生成结果执行动作
    ↓
Server → Client: 完成响应
```

---

## 3. 从零实现

### Step 1：采样请求处理器

```python
class SamplingHandler:
    """MCP 采样请求处理器。"""
    def __init__(self):
        self.llm_fn = None  # Client 的 LLM 函数

    def set_llm(self, llm_fn):
        """注册 Client 的 LLM 函数。"""
        self.llm_fn = llm_fn

    def create_message(self, messages, model=None, max_tokens=1024):
        """处理 sampling/createMessage 请求。"""
        if not self.llm_fn:
            return {"error": "未配置 LLM"}

        # 调用 Client 的 LLM
        response = self.llm_fn(messages, model=model, max_tokens=max_tokens)
        return {"role": "assistant", "content": response, "model": model}
```

### Step 2：Server 主持循环

```python
class ServerHostedLoop:
    """Server 主持的智能体循环。"""
    def __init__(self, sampler):
        self.sampler = sampler

    def analyze_code(self, code):
        """分析代码并给出建议。"""
        # Server 调用 Client 的 LLM
        analysis = self.sampler.create_message([
            {"role": "user", "content": f"分析以下代码并给出改进建议：\n{code}"}
        ])
        return analysis
```

---

## 4. 工具

### 4.1 MCP 采样消息格式

```json
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {"role": "user", "content": "分析这段代码"}
    ],
    "model": "gpt-4o",
    "maxTokens": 1024
  }
}
```

---

## 5. 工程最佳实践

### 5.1 采样设计原则

- **最小权限**：Server 只请求必要的生成——不要过度调用
- **超时控制**：设置合理的超时——防止 LLM 生成过长
- **结果验证**：Server 验证 LLM 生成的结果——不盲目执行

### 5.2 踩坑经验

- **Server 没有 API 密钥**：采样请求必须通过 Client 发送——Server 不直接调用 LLM
- **采样循环死锁**：两个 Server 互相请求采样——需要超时和重试机制
- **结果不一致**：同一采样请求在不同 Client 上可能产生不同结果——LLM 非确定性

---

## 6. 常见错误

### 错误 1：Server 直接调用 LLM API

**现象：** Server 需要管理 API 密钥——安全风险。

**修复：** 使用 MCP 采样——Server 请求 Client 的 LLM，Client 拥有凭证。

### 错误 2：采样请求中缺少超时

**现象：** LLM 生成过长——Server 等待时间过长。

**修复：** 设置 `maxTokens` 和请求级超时。

---

## 7. 面试考点

### Q1：MCP 采样解决了什么问题？（难度：⭐⭐）

**参考答案：**
传统 MCP Server 是"呆板执行者"——只能接收参数、运行代码、返回内容。但有些任务需要 Server "思考"——比如分析代码并给出建议。采样让 Server 可以请求 Client 的 LLM 生成内容——而无需 Server 拥有 API 密钥。这实现了"Server 主持的智能体循环"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 采样 (Sampling) | "Server 请求 LLM 生成" | Server 通过 MCP 向 Client 的 LLM 发起生成请求 |
| sampling/createMessage | "采样请求" | MCP 的采样方法——Server 请求 Client 的 LLM 生成内容 |
| Server 主持循环 | "Server 有自己的智能体" | Server 在不拥有 API 密钥的情况下主持智能体循环 |
| SEP-1577 | "采样中的工具" | 2025-11-25 规范添加的在采样请求中使用工具的能力 |

---

## 📚 小结

MCP 采样让 Server 可以请求 Client 的 LLM 生成内容——无需 Server 拥有 API 密钥。这实现了 Server 主持的智能体循环。SEP-1577 在采样中添加了工具支持。关键是：Server 调用 Client 的 LLM，Client 控制模型选择和凭证。

---

## ✏️ 练习

1. **【实现】** 构建一个 Server 主持的代码分析循环——Server 调用 LLM 分析代码并返回建议
2. **【对比】** 对比 Server 直接调用 LLM 和使用 MCP 采样的安全差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 采样处理器 | `code/main.py` | Server 端采样请求 + 智能体循环 |

---

## 📖 参考资料

1. [文档] MCP 采样规范: https://spec.modelcontextprotocol.io
2. [文档] SEP-1577: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577
3. [文档] MCP 规范 2025-11-25

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
