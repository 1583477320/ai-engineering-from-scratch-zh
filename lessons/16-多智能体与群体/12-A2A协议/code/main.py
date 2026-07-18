"""A2A 最小服务器和客户端——使用 http.server。

实现发现-提交-轮询-获取工件流程：
  - GET /.well-known/agent.json  -> Agent Card
  - POST /tasks                  -> 创建任务
  - GET /tasks/{id}              -> 状态 + 工件

服务器在后台线程运行；客户端与之通信并打印追踪。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from uuid import uuid4


AGENT_CARD = {
    "name": "code-review-agent",
    "version": "0.1.0",
    "skills": ["review-python"],
    "endpoints": {"tasks": "http://localhost:8765/tasks"},
    "auth": {"type": "none"},
    "modalities": ["text", "structured"],
    "protocol_version": "a2a-0.3",
}


# ── 任务存储 ──────────────────────────────────────────────

class TaskStore:
    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, skill: str, payload: dict) -> str:
        tid = str(uuid4())[:8]
        with self._lock:
            self.tasks[tid] = {
                "id": tid, "skill": skill, "payload": payload,
                "state": "submitted", "artifact": None, "created_at": time.time(),
            }
        threading.Thread(target=self._run, args=(tid,), daemon=True).start()
        return tid

    def _run(self, tid: str) -> None:
        with self._lock:
            self.tasks[tid]["state"] = "working"
        time.sleep(0.2)
        with self._lock:
            t = self.tasks[tid]
            if t["skill"] == "review-python":
                code = t["payload"].get("code", "")
                issues = []
                if "return" not in code: issues.append("no return statement")
                if "def " not in code: issues.append("no function definition")
                t["artifact"] = {"type": "structured", "data": {"issues": issues, "lines": code.count("\n") + 1}}
                t["state"] = "completed"
            else:
                t["state"] = "failed"
                t["artifact"] = {"type": "text", "data": f"unknown skill '{t['skill']}'"}

    def get(self, tid: str) -> dict | None:
        with self._lock:
            return self.tasks.get(tid)


STORE = TaskStore()


# ── HTTP 处理器 ──────────────────────────────────────────

class A2AHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/.well-known/agent.json":
            self._send_json(200, AGENT_CARD)
            return
        if self.path.startswith("/tasks/"):
            tid = self.path.split("/tasks/", 1)[1]
            task = STORE.get(tid)
            if task is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, task)
            return
        self._send_json(404, {"error": "route not found"})

    def do_POST(self):
        if self.path == "/tasks":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            tid = STORE.create(body.get("skill", ""), body.get("payload", {}))
            self._send_json(201, {"task_id": tid, "state": "submitted"})
            return
        self._send_json(404, {"error": "route not found"})


def run_server():
    server = HTTPServer(("localhost", 8765), A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def http_json(method, url, body=None):
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_client():
    print("\n[1] 发现: GET /.well-known/agent.json")
    card = http_json("GET", "http://localhost:8765/.well-known/agent.json")
    print(f"    name={card['name']}, skills={card['skills']}")

    print("\n[2] 提交任务: POST /tasks")
    submission = {"skill": "review-python", "payload": {"code": "x = 1\nprint(x)\n"}}
    resp = http_json("POST", card["endpoints"]["tasks"], submission)
    tid = resp["task_id"]
    print(f"    task_id={tid}, state={resp['state']}")

    print("\n[3] 轮询直到完成")
    for i in range(10):
        task = http_json("GET", f"http://localhost:8765/tasks/{tid}")
        print(f"    attempt {i + 1}: state={task['state']}")
        if task["state"] in ("completed", "failed"):
            print(f"    artifact: {task['artifact']}")
            break
        time.sleep(0.1)


def main():
    print("A2A 最小协议演示")
    print("-" * 30)
    server = run_server()
    time.sleep(0.1)
    try:
        run_client()
    finally:
        server.shutdown()
    print("\n要点: 发现 + 任务生命周期 + 类型化工件 + 认证 = A2A 表面。")
    print("MCP 是智能体 ↔ 工具（垂直）；A2A 是智能体 ↔ 智能体（水平）。生产两者都用。")


if __name__ == "__main__":
    main()
