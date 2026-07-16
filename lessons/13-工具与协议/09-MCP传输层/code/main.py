# MCP Streamable HTTP 传输层

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/mcp", methods=["POST"])
def handle_mcp():
    """MCP Streamable HTTP 端点。"""
    data = request.json
    method = data.get("method")
    params = data.get("params", {})
    req_id = data.get("id")

    session_id = request.headers.get("Mcp-Session-Id")
    if not session_id:
        return jsonify({"error": "缺少 Mcp-Session-Id"}), 401

    if method == "initialize":
        return jsonify({
            "jsonrpc": "2.0",
            "result": {"protocolVersion": "2025-11-25", "capabilities": {}},
            "id": req_id
        })

    return jsonify({"jsonrpc": "2.0", "result": {"status": "ok"}, "id": req_id})


if __name__ == "__main__":
    print("MCP Streamable HTTP 传输层演示\n")
    print("传输选择:")
    print("  stdio: 本地应用，最简单")
    print("  Streamable HTTP: 远程部署，会话连续性")
    print("  SSE: 已弃用（2026年中）")
