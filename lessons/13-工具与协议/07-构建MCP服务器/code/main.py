# MCP 笔记服务器实现

import json


class MCPServer:
    """简化版 MCP 服务器。"""
    def __init__(self, name="note-server"):
        self.name = name
        self.tools = {}
        self.resources = {}
        self.prompts = {}

    def handle(self, message):
        msg = json.loads(message)
        method = msg.get("method")
        params = msg.get("params", {})
        req_id = msg.get("id")

        handlers = {
            "initialize": lambda: {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}, "resources": {}, "prompts": {}}, "serverInfo": {"name": self.name, "version": "1.0.0"}},
            "tools/list": lambda: {"tools": list(self.tools.values())},
            "tools/call": lambda: self._call_tool(params),
            "resources/list": lambda: {"resources": list(self.resources.values())},
            "prompts/list": lambda: {"prompts": list(self.prompts.values())},
        }

        if method in handlers:
            return json.dumps({"jsonrpc": "2.0", "result": handlers[method](), "id": req_id})
        return json.dumps({"jsonrpc": "2.0", "error": {"code": -32601, "message": f"未知: {method}"}, "id": req_id})

    def _call_tool(self, params):
        name = params.get("name")
        args = params.get("arguments", {})
        if name in self.tools:
            result = self.tools[name]["executor"](**args)
            return {"content": [{"type": "text", "content": str(result)}]}
        return {"error": {"code": -32601, "message": f"工具 {name} 不存在"}}


if __name__ == "__main__":
    print("MCP 笔记服务器演示\n")
    server = MCPServer("note-server")
    server.tools["create_note"] = {"name": "create_note", "description": "创建笔记", "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}, "executor": lambda title: f"笔记 '{title}' 已创建"}

    resp = server.handle(json.dumps({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "create_note", "arguments": {"title": "MCP笔记"}}, "id": 1}))
    result = json.loads(resp)["result"]["content"][0]["content"]
    print(f"结果: {result}")
