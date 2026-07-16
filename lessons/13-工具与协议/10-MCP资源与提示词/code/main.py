# MCP Resources 和 Prompts 实现


class ResourceManager:
    """MCP Resources 管理器。"""
    def __init__(self):
        self.resources = {}
        self.subscribers = {}

    def add(self, uri, name, mime_type, reader, description=""):
        self.resources[uri] = {"uri": uri, "name": name, "mimeType": mime_type,
                                "reader": reader, "description": description}

    def list_resources(self):
        return {"resources": [{"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]}
                              for r in self.resources.values()]}

    def read(self, uri):
        if uri in self.resources:
            content = self.resources[uri]["reader"]()
            return {"contents": [{"uri": uri, "mimeType": self.resources[uri]["mimeType"], "text": content}]}
        return {"error": {"code": -32601, "message": f"资源 {uri} 不存在"}}


class PromptManager:
    """MCP Prompts 管理器。"""
    def __init__(self):
        self.prompts = {}

    def add(self, name, description, arguments, template_fn):
        self.prompts[name] = {"name": name, "description": description,
                               "arguments": arguments, "template": template_fn}

    def list_prompts(self):
        return {"prompts": [{"name": p["name"], "description": p["description"]}
                           for p in self.prompts.values()]}

    def get_prompt(self, name, arguments={}):
        if name in self.prompts:
            messages = self.prompts[name]["template"](**arguments)
            return {"description": self.prompts[name]["description"], "messages": messages}
        return {"error": {"code": -32601, "message": f"提示词 {name} 不存在"}}


if __name__ == "__main__":
    print("MCP Resources + Prompts 演示\n")

    rm = ResourceManager()
    rm.add("docs://api", "API文档", "text/markdown", lambda: "# API 文档\n\nGET /users - 获取用户列表")
    rm.add("config://settings", "配置", "application/json", lambda: '{"theme": "dark"}')

    print("Resources:")
    for r in rm.list_resources()["resources"]:
        print(f"  {r['uri']}: {r['name']}")

    print(f"\n读取 docs://api: {rm.read('docs://api')['contents'][0]['text'][:30]}...")

    pm = PromptManager()
    pm.add("code-review", "代码审查模板", [{"name": "code", "description": "要审查的代码", "required": True}],
           lambda code: [{"role": "user", "content": f"审查以下代码：{code}"}])

    print(f"\nPrompts:")
    for p in pm.list_prompts()["prompts"]:
        print(f"  {p['name']}: {p['description']}")

    result = pm.get_prompt("code-review", {"code": "def foo(): pass"})
    print(f"\n获取提示词: {result['messages'][0]['content'][:50]}...")
