# MCP Apps——通过 `ui://` 的交互式 UI 资源

> 纯文本工具输出限制了智能体的展示能力。MCP Apps（SEP-1724，2026 年 1 月 26 日）让工具返回沙箱化的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染。仪表板、表单、地图、3D 场景——通过一个扩展实现。

**类型：** 概念课 | **语言：** Python + HTML | **前置知识：** 阶段 13 · 07（MCP Server）、10（Resources）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从工具调用中返回 `ui://` 资源并设置正确的 MIME 和元数据
- [ ] 理解 `text/html;profile=mcp-app` MIME 类型
- [ ] 设计一个沙箱化的 HTML UI 应用——通过 iframe 和 postMessage 安全通信
- [ ] 理解 MCP App 的安全面——允许 Server 渲染 HTML 带来的风险

---

## 1. 问题

LLM 的工具输出通常是纯文本或 JSON。但很多任务需要更丰富的展示——数据可视化仪表板、交互式表单、3D 模型预览。纯文本无法表达这些。

MCP Apps（SEP-1724）解决了这个问题：**让工具返回可渲染的 HTML**——在 Claude Desktop、Cursor 等客户端中内联显示。

---

## 2. 概念

### 2.1 MCP App 架构

```
Tool 返回: ui:// 资源
    ↓
Client 检测到 text/html;profile=mcp-app MIME 类型
    ↓
Client 在沙箱 iframe 中渲染 HTML
    ↓
用户交互 → postMessage → Client 接收结果
```

### 2.2 `ui://` 资源格式

```json
{
  "contents": [{
    "uri": "ui://dashboard-123",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<html>...交互式仪表板...</html>",
    "metadata": {
      "title": "销售仪表板",
      "width": 800,
      "height": 600
    }
  }]
}
```

### 2.3 沙箱化通信

HTML 应用运行在 iframe 沙箱中——通过 `postMessage` 与 Client 通信：

```html
<!-- 沙箱中的 HTML 应用 -->
<script>
  window.parent.postMessage({
    type: "mcp-app-action",
    action: "export",
    data: { format: "pdf" }
  }, "*");
</script>
```

### 2.4 安全面

| 风险 | 描述 | 缓解 |
|------|------|------|
| XSS | Server 注入恶意脚本 | iframe sandbox 属性 |
| 数据泄露 | HTML 访问敏感数据 | 后端 API 控制访问 |
| 恶意交互 | 用户在 App 中执行危险操作 | 输入验证 + 确认对话框 |

---

## 3. 从零实现

### Step 1：MCP App 资源生成器

```python
class MCPAppEmitter:
    """MCP App 资源生成器。"""
    @staticmethod
    def create_dashboard(data, title="仪表板"):
        """生成仪表板 HTML。"""
        html = f"""<html>
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<div id="data">{"<p>" + json.dumps(data) + "</p>"}</div>
<script>
  window.parent.postMessage({{type: 'mcp-app-ready'}}, '*');
</script>
</body></html>"""
        return {
            "uri": f"ui://dashboard-{hash(title)}",
            "mimeType": "text/html;profile=mcp-app",
            "text": html,
            "metadata": {"title": title, "width": 800, "height": 600},
        }
```

### Step 2：安全沙箱配置

```html
<!-- MCP App 的 iframe 应使用 sandbox 属性 -->
<iframe
  sandbox="allow-scripts allow-forms allow-popups"
  src="ui://dashboard-123"
  width="800" height="600"
></iframe>
```

---

## 4. 工具

### 4.1 MCP 规范

| 规范 | 日期 | 状态 |
|------|------|------|
| SEP-1724 (MCP Apps) | 2026-01-26 | 正式发布 |
| `text/html;profile=mcp-app` | 2026-01-26 | MIME 类型 |
| `ui://` | 2026-01-26 | 资源 URI scheme |

### 4.2 支持 MCP Apps 的客户端

| 客户端 | 支持情况 |
|--------|---------|
| Claude Desktop | ✅ |
| ChatGPT | ✅ |
| Cursor | ✅ |
| VS Code | ✅ |
| Goose | ✅ |

---

## 5. 工程最佳实践

### 5.1 UI App 设计原则

- **最小化 HTML**：App 应该小而专注——一个仪表板、一个表单
- **无外部依赖**：沙箱中不能加载外部脚本——所有代码必须内联
- **响应式**：支持不同尺寸的 iframe

### 5.2 踩坑经验

- **MIME 类型错误**：必须是 `text/html;profile=mcp-app` 而非 `text/html`
- **postMessage 方向**：App → Client 用 `window.parent.postMessage`
- **沙箱限制**：不能使用 `alert`、`confirm` 等弹窗——用 DOM 替代

---

## 6. 常见错误

### 错误 1：MIME 类型不正确

**现象：** Client 不渲染 HTML——当作纯文本处理。

**修复：** 必须使用 `text/html;profile=mcp-app`。

### 错误 2：沙箱中引用外部脚本

**现象：** HTML App 不加载——iframe sandbox 阻止了外部请求。

**修复：** 所有 HTML/JS/CSS 必须内联——沙箱不允许外部资源加载。

---

## 7. 面试考点

### Q1：MCP Apps 和传统 Web 应用有什么区别？（难度：⭐⭐）

**参考答案：**
传统 Web 应用在浏览器中运行——完全访问网络和本地资源。MCP App 在 iframe 沙箱中运行——只能访问 Client 允许的资源，通过 postMessage 与 Client 通信。MCP App 更安全（沙箱隔离）但功能受限（不能访问任意 API）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| MCP App | "交互式 UI 工具" | 通过 `ui://` 资源在客户端中渲染沙箱化 HTML 的 MCP 工具 |
| `ui://` | "UI 资源 URI" | MCP Apps 使用的 URI scheme——表示一个可渲染的 HTML 应用 |
| postMessage | "沙箱通信" | MCP App 通过 iframe postMessage 与 Client 安全通信 |
| SEP-1724 | "MCP Apps 规范" | 2026-01-26 发布——定义 MCP App 的 MIME、沙箱、交互协议 |

---

## 📚 小结

MCP Apps 让工具返回可渲染的 HTML——在 Claude Desktop 等客户端中内联显示。通过 `ui://` 资源、`text/html;profile=mcp-app` MIME、沙箱 iframe 和 postMessage 通信实现。安全面：iframe sandbox 隔离 + 后端 API 控制访问。

---

## ✏️ 练习

1. **【实现】** 构建一个 MCP App——返回一个简单的交互式仪表板 HTML
2. **【设计】** 设计一个带数据导出功能的 MCP App——通过 postMessage 返回用户选择的数据

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| MCP App 生成器 | `code/main.py` | ui:// 资源生成 + 沙箱 HTML |

---

## 📖 参考资料

1. [文档] SEP-1724 MCP Apps: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1724
2. [文档] MCP 规范 2025-11-25
3. [文档] iframe sandbox: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe/sandbox

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
