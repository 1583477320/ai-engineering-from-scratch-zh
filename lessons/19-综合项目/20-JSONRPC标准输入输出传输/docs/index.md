# 综合项目20——JSON-RPC 2.0换行分隔标准输入输出传输

> 模型客户端与工具服务器之间的传输是JSON-RPC over stdio。手写一次教会你每个帧层在为什么付费。JSON-RPC 2.0是两页规范，自2013年以来一直存活，因为它在流式、批处理和传输耦合之间不做取舍。本课程构建stdio变体：换行分隔JSON。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第13章（工具与协议）第01-07节、第14章（智能体）第01节
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 用换行分隔JSON实现JSON-RPC 2.0 over stdin/stdout
- 映射五个标准错误码（-32700到-32603）并用正确语义呈现
- 区分请求、响应、通知和批处理
- 处理每行一个解析错误而不污染流的其余部分
- 使用io.BytesIO构建自终止演示

---

## 1. 问题

2026年的编码智能体在单次会话中与约十二个工具服务器对话。每个服务器是独立进程或远程端点。传输格式自2013年以来相同。JSON-RPC 2.0是两页规范——它在stdio、套接字、WebSocket和HTTP上对称。

本课程构建stdio变体。换行分隔JSON。每个请求一行，每个响应一行。传输边界是`\n`。

---

## 2. 核心概念

### 2.1 线格式

四种信封形状：
- **请求**：`{jsonrpc:"2.0", id:7, method:"foo", params:{...}}`
- **成功响应**：`{jsonrpc:"2.0", id:7, result:{...}}`
- **通知**：`{jsonrpc:"2.0", method:"bar", params:{...}}`（无id，服务器不应响应）
- **错误响应**：`{jsonrpc:"2.0", id:7, error:{code, message, data?}}`
- **批处理**：JSON数组，服务器以任意顺序回复每个非通知条目的响应数组

### 2.2 五个错误码

```
-32700  解析错误      JSON无法解析
-32600  无效请求      信封形状错误
-32601  方法未找到
-32602  无效参数
-32603  内部错误
```

### 2.3 流行为

解析错误的特殊规则：响应中的id为`null`，因为请求从未解析出足够信息来提取id。坏JSON行不会停止循环——流不被污染，下一行重新解析。

---

## 3. 从零实现

`code/main.py`实现`StdioTransport`、解析辅助器、写入辅助器和`serve`分发循环。

```python
"""JSON-RPC 2.0换行分隔stdio传输。

核心：每行一个JSON-RPC消息，支持请求/响应/通知/批处理/错误码。

运行：python3 code/main.py
"""

from __future__ import annotations
import io, json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable


ERR_PARSE = -32700; ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601; ERR_INVALID_PARAMS = -32602; ERR_INTERNAL = -32603

class JsonRpcError(Exception):
    code = ERR_INTERNAL
    def __init__(self, message, data=None): super().__init__(message); self.message=message; self.data=data

class MethodNotFound(JsonRpcError): code = ERR_METHOD_NOT_FOUND
class InvalidParams(JsonRpcError): code = ERR_INVALID_PARAMS

@dataclass
class Request:
    method:str; params:Any; id:int|str|None; is_notification:bool


def _is_valid_envelope(msg):
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0": return False
    if not isinstance(msg.get("method"), str): return False
    if "params" in msg and not isinstance(msg["params"], (dict, list)): return False
    if "id" in msg and isinstance(msg["id"], bool): return False
    return True


def parse_request(raw):
    try: msg = json.loads(raw)
    except json.JSONDecodeError as exc: return None, _err(None, ERR_PARSE, str(exc))
    if not _is_valid_envelope(msg): return None, _err(msg.get("id") if isinstance(msg, dict) else None, ERR_INVALID_REQUEST, "invalid envelope")
    return Request(method=msg["method"], params=msg.get("params"), id=msg.get("id"), is_notification="id" not in msg), None


def _err(rid, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None: e["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": e}

def _ok(rid, result): return {"jsonrpc": "2.0", "id": rid, "result": result}


class StdioTransport:
    def __init__(self, stdin: BinaryIO, stdout: BinaryIO): self._in=stdin; self._out=stdout
    def read_line(self):
        line = self._in.readline()
        return line if line else None
    def write_response(self, rid, result): self._write(_ok(rid, result))
    def write_error(self, rid, code, message, data=None): self._write(_err(rid, code, message, data))
    def write_notification(self, method, params=None):
        env = {"jsonrpc": "2.0", "method": method}
        if params is not None: env["params"] = params
        self._write(env)
    def _write(self, obj):
        self._out.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        self._out.flush()


Handler = Callable[[str, Any], Any]


def _handle_one(handler, req):
    try: result = handler(req.method, req.params)
    except (MethodNotFound, InvalidParams, JsonRpcError) as exc:
        return None if req.is_notification else _err(req.id, exc.code, exc.message, exc.data)
    except Exception as exc:
        return None if req.is_notification else _err(req.id, ERR_INTERNAL, "internal", {"exception": type(exc).__name__})
    return None if req.is_notification else _err(req.id, result) if False else _ok(req.id, result)


def serve(handler, transport):
    while True:
        line = transport.read_line()
        if line is None: return
        text = line.decode("utf-8").rstrip("\n").strip()
        if not text: continue
        try: parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            transport.write_error(None, ERR_PARSE, str(exc)); continue
        if isinstance(parsed, list):
            if not parsed: transport.write_error(None, ERR_INVALID_REQUEST, "empty batch"); continue
            out = []
            for raw in parsed:
                if not _is_valid_envelope(raw): out.append(_err(raw.get("id") if isinstance(raw, dict) else None, ERR_INVALID_REQUEST, "invalid envelope")); continue
                req = Request(method=raw["method"], params=raw.get("params"), id=raw.get("id"), is_notification="id" not in raw)
                resp = _handle_one(handler, req)
                if resp is not None: out.append(resp)
            if out:
                transport._out.write((json.dumps(out, separators=(",", ":")) + "\n").encode("utf-8"))
                transport._out.flush()
        else:
            req, err = parse_request(text)
            if err is not None: transport._write(err); continue
            resp = _handle_one(handler, req)
            if resp is not None: transport._write(resp)


def _demo():
    def handler(method, params):
        if method == "math.add":
            if not isinstance(params, dict) or "a" not in params or "b" not in params: raise InvalidParams("a and b required")
            return params["a"] + params["b"]
        if method == "echo": return params
        raise MethodNotFound(f"method {method!r}")

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "math.add", "params": {"a": 2, "b": 3}},
        {"jsonrpc": "2.0", "id": 2, "method": "math.add", "params": {"a": 5}},
        {"jsonrpc": "2.0", "id": 3, "method": "missing"},
        {"jsonrpc": "2.0", "method": "log", "params": {"level": "info"}},  # notification
        [{"jsonrpc": "2.0", "id": 10, "method": "echo", "params": {"text": "hi"}},
         {"jsonrpc": "2.0", "method": "log", "params": {"msg": "skip"}},  # notification in batch
         {"jsonrpc": "2.0", "id": 11, "method": "math.add", "params": {"a": 1, "b": 1}}],
    ]
    stdin = io.BytesIO()
    for r in requests: stdin.write((json.dumps(r) + "\n").encode("utf-8"))
    stdin.write(b"{not json\n")  # parse error
    stdin.seek(0)
    stdout = io.BytesIO()
    serve(handler, StdioTransport(stdin, stdout))
    stdout.seek(0)
    lines = [json.loads(line) for line in stdout.read().decode("utf-8").splitlines() if line]
    print(json.dumps({"responses": lines}, indent=2))


if __name__ == "__main__":
    _demo()
```

运行结果：

```json
{
  "responses": [
    {"jsonrpc": "2.0", "id": 1, "result": 5},
    {"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "a and b required"}},
    {"jsonrpc": "2.0", "id": 3, "error": {"code": -32601, "message": "method 'missing'"}},
    {"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "..."}},
    {"jsonrpc": "2.0", "id": 10, "result": {"text": "hi"}},
    {"jsonrpc": "2.0", "id": 11, "result": 2}
  ]
}
```

---

## 4. 工具实践

**关键规则**：
- 通知无id，服务器不响应
- 解析错误id为null，流不被污染
- 批处理：JSON数组入，JSON数组出（任意顺序）
- 传输不知道哪些方法存在——委托给`handler(method, params)`

---

## 5. LLM视角

**传输层视角**：JSON-RPC是协议无关的——stdio、WebSocket、HTTP都适用。这使智能体可以与本地工具和远程服务使用相同的通信模式。

**通知视角**：通知是fire-and-forget——用于进度事件、取消信号和日志行。长时间运行的工具可以用通知流式传输状态更新。

---

## 6. 工程最佳实践

**错误处理**：每个解析错误单独处理，不中断流。坏行写错误响应后继续下一行。

**编码**：UTF-8编码，JSON紧凑格式（无多余空格）。

**测试**：io.BytesIO替代真实进程，行为完全相同（.readline()和.write()契约）。

---

## 7. 常见错误

**错误1：对通知返回响应**
症状：客户端无法将响应关联到调用点
修复：服务器永远不对通知返回响应

**错误2：解析错误中断流**
症状：一个坏JSON行导致整个会话停止
修复：写-32700响应后继续读取下一行

---

## 8. 面试考点

**Q1：JSON-RPC为什么比gRPC更适合编码智能体场景？**
考察：对协议选择的理解

**Q2：通知和请求的关键区别是什么？**
考察：对JSON-RPC语义的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| JSON-RPC 2.0 | "RPC协议" | 两页规范，支持stdio/WebSocket/HTTP |
| 通知 | "fire-and-forget" | 无id的请求，服务器不应响应 |
| 批处理 | "批量调用" | JSON数组，服务器回复响应数组 |
| 解析错误 | "格式错误" | 无法解析的JSON行，id为null |
| 换行分隔 | "帧边界" | 每个JSON-RPC消息占一行，\n分隔 |

---

## 参考文献

- [JSON-RPC 2.0规范](https://www.jsonrpc.org/specification)
- [RFC 8259 JSON](https://datatracker.ietf.org/doc/html/rfc8259)
