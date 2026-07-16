# MCP 资源与提示词——超越工具的上下文暴露

> Tools 获得了 90% 的 MCP 关注。其他两个服务器原语解决了不同的问题。Resources 暴露数据供读取；Prompts 暴露可复用的模板作为斜杠命令。许多服务器应该使用 Resources 而不是将读操作包装为 Tools，使用 Prompts 而不是在 Client 提示词中硬编码工作流。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 根据领域决定将能力暴露为 Tool、Resource 还是 Prompt
- [ ] 实现 `resources/list`、`resources/read`、`resources/subscribe`、`resources/unsubscribe`
- [ ] 实现 `prompts/list`、`prompts/get`
- [ ] 理解资源变更通知 (`notifications/resources/list_changed`)

---

## 1. 问题

Tool 适合"执行动作"（写入/修改），Resource 适合"读取数据"（查询/浏览），Prompt 适合"模板化提示词"（斜杠命令）。很多开发者将一切都包装为 Tool——但这导致了一些问题：

- 读取操作暴露为 Tool 导致多轮冗余调用
- 提示词模板硬编码在 Client 端——无法动态更新
- 数据变更时 Client 无法感知

正确的选择让架构更清晰。

---

## 2. 概念

### 2.1 选择指南

| 场景 | 选择 | 原因 |
|------|------|------|
| 执行动作（写入） | Tool | 需要副作用 |
| 读取数据（查询） | Resource | 可缓存，支持订阅 |
| 模板化提示词 | Prompt | Client 可动态加载 |
| 既读又写 | Tool | Tool 可以做任何事 |

### 2.2 Resources 原语

- `resources/list` — 列出可用资源
- `resources/read` — 读取资源内容
- `resources/subscribe` — 订阅资源变更
- `resources/unsubscribe` — 取消订阅
- `notifications/resources/list_changed` — 通知 Client 资源变更

### 2.3 Prompts 原语

- `prompts/list` — 列出可用提示词模板
- `prompts/get` — 获取提示词模板（含参数）
- Prompt 有参数——Client 在 `prompts/get` 时提供参数值

---

## 3. 从零实现

### Step 1：Resources 处理器

```python
class ResourceManager:
    """MCP Resources 管理器。"""
    def __init__(self):
        self.resources = {}

    def add(self, uri, name, mime_type, reader, description=""):
        self.resources[uri] = {"uri": uri, "name": name, "mimeType": mime_type, "reader": reader, "description": description}

    def list_resources(self):
        return {"resources": [{"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]} for r in self.resources.values()]}

    def read_resource(self, uri):
        if uri in self.resources:
            content = self.resources[uri]["reader"]()
            return {"contents": [{"uri": uri, "mimeType": self.resources[uri]["mimeType"], "text": content}]}
        return {"error": {"code": -32601, "message": f"资源 {uri} 不存在"}}


class PromptManager:
    """MCP Prompts 管理器。"""
    def __init__(self):
        self.prompts = {}

    def add(self, name, description, arguments, template_fn):
        self.prompts[name] = {"name": name, "description": description, "arguments": arguments, "template": template_fn}

    def list_prompts(self):
        return {"prompts": [{"name": p["name"], "description": p["description"]} for p in self.prompts.values()]}

    def get_prompt(self, name, arguments={}):
        if name in self.prompts:
            messages = self.prompts[name]["template"](**arguments)
            return {"description": self.prompts[name]["description"], "messages": messages}
        return {"error": {"code": -32601, "message": f"提示词 {name} 不存在"}}
```

### Step 2：使用场景示例

```python
def demo():
    """Resources vs Tools vs Prompts 示例。"""
    # 应该用 Resource: 读取项目文档
    resource = {"uri": "docs://api-reference", "name": "API 参考文档", "mimeType": "text/markdown"}
    # 应该用 Tool: 发送邮件
    tool = {"name": "send_email", "description": "发送邮件"}
    # 应该用 Prompt: 代码审查模板
    prompt = {"name": "code-review", "description": "审查代码的提示词模板"}
```

---

## 4. 工具

### 4.1 MCP 规范

| 方法 | 说明 | 必需 |
|------|------|------|
| `resources/list` | 列出资源 | 如果支持 Resource |
| `resources/read` | 读取资源 | 如果支持 Resource |
| `prompts/list` | 列出提示词 | 如果支持 Prompt |
| `prompts/get` | 获取提示词 | 如果支持 Prompt |

### 4.2 最佳实践

- **Resource 适合只读数据**：文档、配置、状态
- **Tool 适合可写操作**：创建、更新、删除
- **Prompt 适合工作流模板**：审查、分析、总结

---

## 5. 工程最佳实践

### 5.1 选择决策树

```
这个操作可以缓存吗？
├── 可以 → 需要状态变更通知吗？
│   ├── 需要 → Resource（支持 subscribe）
│   └── 不需要 → Resource
└── 不可以 → 有副作用吗？
    ├── 有 → Tool
    └── 无 → 用户复用的模板吗？
        ├── 是 → Prompt
        └── 否 → Tool
```

### 5.2 踩坑经验

- **Resource 作为 Tool 包装**：读操作包装为 Tool 浪费了 Resource 的订阅和缓存能力
- **Prompt 硬编码**：提示词模板在 Client 端硬编码——每次更新需要部署

---

## 6. 常见错误

### 错误 1：将读操作包装为 Tool

**现象：** Client 每次需要数据时都调用 Tool——无法缓存。

**修复：** 读操作用 Resource——支持缓存和订阅。

### 错误 2：Client 端硬编码提示词模板

**现象：** 更新提示词模板需要部署 Client。

**修复：** 使用 Prompt 原语——Server 端管理模板，Client 动态加载。

---

## 7. 面试考点

### Q1：Tool、Resource、Prompt 三者的核心区别是什么？（难度：⭐⭐）

**参考答案：**
Tool 执行动作（写入/修改）——有副作用；Resource 提供数据（读取/浏览）——可缓存、支持订阅；Prompt 提供模板化提示词——Server 端管理模板，Client 动态加载。选择：读操作用 Resource，写操作用 Tool，模板用 Prompt。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Resource | "可读的数据" | 暴露只读数据的 MCP 原语——支持缓存和订阅变更通知 |
| Prompt | "提示词模板" | 暴露可复用提示词模板的 MCP 原语——Client 动态加载 |
| Tool | "可执行的函数" | 暴露可执行操作的 MCP 原语——有副作用，不可缓存 |
| 订阅 (Subscribe) | "监听变更" | Resource 支持 Client 订阅变更通知——数据变化时自动推送 |

---

## 📚 小结

MCP 有三种原语：Tool（执行）、Resource（读取）、Prompt（模板）。选择合适原语让架构更清晰。Resource 可缓存和订阅，适合读操作；Tool 有副作用，适合写操作；Prompt 动态可更新，适合模板。

---

## ✏️ 练习

1. **【实现】** 为"MCP 笔记服务器"添加 Resources 和 Prompts 支持
2. **【设计】** 将"读取笔记"从 Tool 改为 Resource——并添加变更通知

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Resources+Prompts 管理器 | `code/main.py` | Resource 和 Prompt 的完整实现 |

---

## 📖 参考资料

1. [文档] MCP Resources: https://spec.modelcontextprotocol.io
2. [文档] MCP Prompts: https://spec.modelcontextprotocol.io
3. [文档] MCP 规范 2025-11-25

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
