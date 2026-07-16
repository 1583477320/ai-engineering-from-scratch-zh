# MCP 基础：JSON-RPC 2.0 解析器

import json


def parse_jsonrpc(message):
    """解析 JSON-RPC 2.0 消息。"""
    msg = json.loads(message)
    return {"method": msg.get("method"), "params": msg.get("params", {}), "id": msg.get("id")}


def make_jsonrpc_response(result, request_id):
    """构建 JSON-RPC 2.0 响应。"""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": request_id})


def make_jsonrpc_error(error_code, message, request_id):
    """构建 JSON-RPC 2.0 错误响应。"""
    return json.dumps({"jsonrpc": "2.0", "error": {"code": error_code, "message": message}, "id": request_id})


if __name__ == "__main__":
    print("MCP 基础演示\n")

    # 模拟 initialize
    msg = json.dumps({"jsonrpc": "2.0", "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}}, "id": 1})
    parsed = parse_jsonrpc(msg)
    print(f"解析: method={parsed['method']}, id={parsed['id']}")

    resp = make_jsonrpc_response({
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "test-server", "version": "1.0.0"},
    }, parsed["id"])
    print(f"响应: {resp[:80]}...")
