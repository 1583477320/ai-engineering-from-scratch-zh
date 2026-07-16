# MCP Roots 和 Elicitation 实现


class RootsManager:
    """MCP Roots 管理器。"""
    def __init__(self):
        self.roots = []

    def set_roots(self, uris):
        self.roots = uris
        return {"method": "notifications/roots/list_changed"}

    def check_access(self, uri):
        return any(uri.startswith(root) for root in self.roots)


class ElicitationHandler:
    """MCP Elicitation 处理器。"""
    def __init__(self):
        self.pending = {}

    def request_input(self, request_id, schema, message=""):
        self.pending[request_id] = {"schema": schema, "message": message, "status": "pending"}
        return {"method": "elicitation/create", "params": {"requestId": request_id, "schema": schema, "message": message}}

    def handle_response(self, request_id, values):
        if request_id in self.pending:
            self.pending[request_id]["values"] = values
            self.pending[request_id]["status"] = "complete"
            return values
        return None


if __name__ == "__main__":
    print("MCP Roots + Elicitation 演示\n")

    roots = RootsManager()
    roots.set_roots(["file:///project-a/", "file:///project-b/"])
    print(f"允许访问: project-a, project-b")
    print(f"  访问 project-a/docs: {roots.check_access('file:///project-a/docs/readme.md')}")
    print(f"  访问 system/etc/passwd: {roots.check_access('file:///etc/passwd')}")

    elicitation = ElicitationHandler()
    elicitation.request_input("req-1", {"type": "object", "properties": {"name": {"type": "string"}}}, "请输入笔记标题")
    print(f"\nElicitation 请求已发送: req-1")
    result = elicitation.handle_response("req-1", {"name": "MCP笔记"})
    print(f"用户输入: {result}")
