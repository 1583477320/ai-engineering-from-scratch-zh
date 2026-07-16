# 构建 MCP 服务器——Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio hello-world。真正的服务器暴露工具、资源和提示词，处理能力协商，发出结构化错误，并在 SDK 之间保持一致。本课从头到尾构建一个笔记服务器：标准 stdio 传输、JSON-RPC 调度、三种服务器原语，以及可以放入 Python SDK 的 FastMCP 或 TypeScript SDK 的纯函数风格。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 06（MCP 基础）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、`prompts/get` 方法
- [ ] 用 stdio 传输层运行 MCP 服务器
- [ ] 处理能力协商和版本兼容性
- [ ] 构建一个完整的笔记 MCP 服务器

---

## 1. 问题

大多数 MCP 教程只展示 stdio 上的 hello-world。但生产级 MCP 服务器需要：工具 + 资源 + 提示词、能力协商、结构化错误处理、跨 SDK 一致性。

本课从头到尾构建一个笔记 MCP 服务器。

---

## 2. 概念

### 2.1 MCP 服务器架构

```
Client (LLM 应用)
    ↓  JSON-RPC (stdio/HTTP)
Server (你的工具)
    ├── Tools: create_note, search_notes, delete_note
    ├── Resources: notes://list, notes://read/{id}
    └── Prompts: note-review
```

### 2.2 服务器生命周期

```
1. initialize: Client 发送能力声明
2. 响应: Server 返回自己的能力
3. notifications/initialized: Client 确认完成
4. 操作: tools/call, resources/read, prompts/get
```

### 2.3 笔记服务器示例

| 原语 | 操作 | 说明 |
|------|------|------|
| Tool: create_note | 创建新笔记 | 接收标题和内容 |
| Tool: search_notes | 搜索笔记 | 返回匹配的笔记列表 |
| Resource: notes://list | 读取所有笔记 | 返回笔记列表 |
| Prompt: note-review | 生成审查模板 | 返回审查提示词 |

---

## 3. 从零实现

### Step 1：MCP 服务器骨架

```python
import json
import sys

class MCPServer:
    """简化版 MCP 服务器——处理 JSON-RPC 2.0 消息。"""
    def __init__(self, name="note-server"):
        self.name = name
        self.tools = {}
        self.resources = {}
        self.prompts = {}

    def handle(self, message):
        """处理 JSON-RPC 2.0 消息。"""
        msg = json.loads(message)
        method = msg.get("method")
        params = msg.get("params", {})
        req_id = msg.get("id")

        if method == "initialize":
            return self._handle_initialize(params, req_id)
        elif method == "tools/list":
            return self._handle_tools_list(req_id)
        elif method == "tools/call":
            return self._handle_tools_call(params, req_id)
        elif method == "resources/list":
            return self._handle_resources_list(req_id)
        elif method == "resources/read":
            return self._handle_resources_read(params, req_id)
        elif method == "prompts/list":
            return self._handle_prompts_list(req_id)
        elif method == "prompts/get":
            return self._handle_prompts_get(params, req_id)

        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"未知方法: {method}"}, "id": req_id}

    def _handle_initialize(self, params, req_id):
        return {"jsonrpc": "2.0", "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": self.name, "version": "1.0.0"},
        }, "id": req_id}

    def _handle_tools_list(self, req_id):
        return {"jsonrpc": "2.0", "result": {"tools": list(self.tools.values())}, "id": req_id}

    def _handle_tools_call(self, params, req_id):
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name not in self.tools:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"工具 {name} 不存在"}, "id": req_id}
        try:
            result = self.tools[name]["executor"](**arguments)
            return {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "content": str(result)}]}, "id": req_id}
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": req_id}

    def _handle_resources_list(self, req_id):
        return {"jsonrpc": "2.0", "result": {"resources": list(self.resources.values())}, "id": req_id}

    def _handle_resources_read(self, params, req_id):
        uri = params.get("uri", "")
        if uri in self.resources:
            content = self.resources[uri]["reader"]()
            return {"jsonrpc": "2.0", "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": content}]}, "id": req_id}
        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"资源 {uri} 不存在"}, "id": req_id}

    def _handle_prompts_list(self, req_id):
        return {"jsonrpc": "2.0", "result": {"prompts": list(self.prompts.values())}, "id": req_id}

    def _handle_prompts_get(self, params, req_id):
        name = params.get("name")
        if name in self.prompts:
            prompt = self.prompts[name]["template"]
            return {"jsonrpc": "2.0", "result": {"description": f"{name} 模板", "messages": [{"role": "user", "content": prompt}]}, "id": req_id}
        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"提示词 {name} 不存在"}, "id": req_id}


# 注册工具
server = MCPServer("note-server")
server.tools["create_note"] = {
    "name": "create_note",
    "description": "创建新笔记",
    "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]},
    "executor": lambda title, content: f"笔记已创建: {title}",
}
server.tools["search_notes"] = {
    "name": "search_notes",
    "description": "搜索笔记",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "executor": lambda query: f"搜索结果: 找到与 '{query}' 相关的笔记",
}
server.tools["delete_note"] = {
    "name": "delete_note",
    "description": "删除笔记",
    "inputSchema": {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]},
    "executor": lambda note_id: f"笔记 {note_id} 已删除",
}

server.resources["notes://list"] = {
    "uri": "notes://list",
    "name": "所有笔记",
    "mimeType": "application/json",
    "reader": lambda: '{"notes": [{"id": "1", "title": "MCP笔记"}, {"id": "2", "title": "工具协议笔记"}]}',
}
server.prompts["note-review"] = {
    "name": "note-review",
    "description": "审查笔记的模板",
    "arguments": [{"name": "note_id", "description": "要审查的笔记ID", "required": True}],
}


if __name__ == "__main__":
    print("MCP 笔记服务器演示\n")

    # 模拟 initialize
    init_response = server.handle(json.dumps({
        "jsonrpc": "2.0", "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}}, "id": 1
    }))
    print(f"initialize: {json.loads(init_response)['result']['serverInfo']}")

    # 模拟 tools/list
    tools_resp = server.handle(json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 2}))
    tools = json.loads(tools_resp)["result"]["tools"]
    print(f"tools: {[t['name'] for t in tools]}")

    # 模拟 tools/call
    call_resp = server.handle(json.dumps({
        "jsonrpc": "2.0", "method": "tools/call",
        "params": {"name": "create_note", "arguments": {"title": "测试", "content": "内容"}}, "id": 3
    }))
    print(f"create_note: {json.loads(call_resp)['result']['content']}")

    # 模拟 resources/read
    res_resp = server.handle(json.dumps({
        "jsonrpc": "2.0", "method": "resources/read",
        "params": {"uri": "notes://list"}, "id": 4
    }))
    print(f"resources: {json.loads(res_resp)['result']['contents'][0]['text'][:50]}...")

    # 模拟 prompts/get
    prompt_resp = server.handle(json.dumps({
        "jsonrpc": "2.0", "method": "prompts/get",
        "params": {"name": "note-review", "arguments": {"note_id": "1"}}, "id": 5
    }))
    print(f"prompts: {json.loads(prompt_resp)['result']['description']}")
