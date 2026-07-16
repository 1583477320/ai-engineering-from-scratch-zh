# Roots 与 Elicitation——作用域与运行时用户输入

> 硬编码路径在用户打开不同项目时就崩溃了。预填充工具参数在用户欠指定时就失败了。Roots 将 Server 限定在用户控制的一组 URI 内；Elicitation 在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，两种常见 MCP 失败模式的修复。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 声明 `roots` 并响应 `notifications/roots/list_changed`
- [ ] 在工具调用中使用 elicitation 暂停并向用户请求输入
- [ ] 设计一个带作用域限制的 MCP Server——只允许访问指定目录
- [ ] 理解 Root 和 Elicitation 对 MCP 安全性的意义

---

## 1. 问题

**Roots 问题**：Server 默认可以访问任何文件路径——用户打开不同项目时路径就错了。

**Elicitation 问题**：工具参数预填充在用户欠指定时失败——Server 需要在运行时向用户请求输入。

两种客户端原语解决两种常见 MCP 失败模式。

---

## 2. 概念

### 2.1 Roots——作用域限制

```
Client 声明 roots: ["file:///project-a/", "file:///project-b/"]
    ↓
Server 只能访问这些 URI 下的文件
    ↓
安全：Server 不能读取 /etc/passwd 等系统文件
```

### 2.2 Elicitation——运行时用户输入

```
工具调用开始 → Server 发现需要额外信息
    ↓
Server 发送 elicitation 请求（表单或 URL）
    ↓
用户提供输入
    ↓
工具调用继续
```

### 2.3 SEP-1036：URL 模式的 Elicitation

2025-11-25 规范添加了 URL 模式的 elicitation——Server 可以返回一个 URL 让用户在浏览器中填写表单。

---

## 3. 从零实现

### Step 1：Roots 管理

```python
class RootsManager:
    """MCP Roots 管理器。"""
    def __init__(self):
        self.roots = []

    def set_roots(self, uris):
        """设置允许的 URI 范围。"""
        self.roots = uris
        # 通知所有 Client
        return {"method": "notifications/roots/list_changed"}

    def check_access(self, uri):
        """检查 URI 是否在允许范围内。"""
        return any(uri.startswith(root) for root in self.roots)
```

### Step 2：Elicitation 处理

```python
class ElicitationHandler:
    """MCP Elicitation 处理器。"""
    def __init__(self):
        self.pending_requests = {}

    def request_input(self, request_id, schema, message=""):
        """向用户请求输入。"""
        self.pending_requests[request_id] = {
            "schema": schema,
            "message": message,
            "status": "pending",
        }
        return {
            "method": "elicitation/create",
            "params": {
                "requestId": request_id,
                "schema": schema,
                "message": message,
            }
        }

    def handle_response(self, request_id, values):
        """处理用户响应。"""
        if request_id in self.pending_requests:
            self.pending_requests[request_id]["values"] = values
            self.pending_requests[request_id]["status"] = "complete"
            return values
        return None
```

---

## 4. 工具

### 4.1 MCP 规范

| 客户端原语 | 说明 |
|-----------|------|
| Roots | 文件系统作用域——Server 只能访问指定 URI |
| Elicitation | 运行时用户输入——Server 在工具调用中暂停请求输入 |
| Sampling | Server 请求 Client 的 LLM 生成 |

---

## 5. 工程最佳实践

### 5.1 Roots 设计

- 默认为最小权限——不声明就拒绝
- 支持 `notifications/roots/list_changed`——用户切换项目时更新
- 多项目场景：为每个项目声明独立的 Root

### 5.2 Elicitation 使用

- 避免过多 elicitation——用户体验差
- 使用表单而非自由文本——结构化输入更容易处理
- 缓存 elicitation 结果——相同请求不重复

### 5.3 踩坑经验

- **Root URI 不匹配**：用户打开新项目但 Root 没更新——Server 访问失败
- **Elicitation 超时**：用户长时间不响应——设置超时并优雅降级

---

## 6. 常见错误

### 错误 1：Roots 范围过大

**现象：** Server 可以访问任意文件——安全风险。

**修复：** 最小权限原则——只声明必要的 URI。

### 错误 2：忽略 `notifications/roots/list_changed`

**现象：** 用户切换项目后 Server 仍访问旧路径。

**修复：** Client 监听 roots 变更通知——更新 Server 的作用域。

---

## 7. 面试考点

### Q1：Roots 和 Elicitation 分别解决什么问题？（难度：⭐⭐）

**参考答案：**
Roots 解决安全问题——限制 Server 的文件访问范围，防止 Server 读取用户未授权的文件。Elicitation 解决信息不足的问题——Server 在工具调用中发现缺少参数时，暂停并向用户请求输入。两者都是客户端原语——由 Client 实现和执行。

### Q2：SEP-1036 的 URL 模式 elicitation 是什么？（难度：⭐⭐⭐）

**参考答案：**
传统 elicitation 返回一个表单让用户填写。URL 模式的 elicitation（2025-11-25 规范，2026 年 H1 仍为实验性）返回一个 URL 让用户在浏览器中打开表单。优势：支持更复杂的输入（如文件上传、多步骤流程）；劣势：需要额外的浏览器交互，可能不适合所有部署环境。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Roots | "作用域限制" | 客户端原语——限制 Server 可访问的 URI 范围 |
| Elicitation | "运行时请求输入" | Server 在工具调用中暂停，向用户请求结构化输入 |
| Root URI | "允许的路径" | Server 可以访问的文件系统 URI 列表 |
| SEP-1036 | "URL elicitation" | URL 模式的 elicitation——2026 年 H1 仍为实验性 |

---

## 📚 小结

Roots 和 Elicitation 是 MCP 的两个客户端原语——分别解决安全和交互问题。Roots 限制 Server 的文件访问范围；Elicitation 在工具调用中暂停请求用户输入。两者都是客户端原语——由 Client 实现和执行。

---

## ✏️ 练习

1. **【实现】** 为笔记 MCP Server 添加 Roots 支持——限制只能访问指定目录
2. **【设计】** 设计一个 elicitation 表单——Server 在删除笔记时向用户确认

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Roots + Elicitation | `code/main.py` | 作用域限制 + 运行时输入请求 |

---

## 📖 参考资料

1. [文档] MCP Roots: https://spec.modelcontextprotocol.io
2. [文档] MCP Elicitation: https://spec.modelcontextprotocol.io
3. [文档] SEP-1036: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1036

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
