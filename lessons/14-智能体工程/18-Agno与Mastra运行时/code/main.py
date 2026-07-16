# Agno 无状态运行时 + Mastra 统一存储


class StatelessAgent:
    """Agno 风格——无状态、微秒级实例化。"""
    def __init__(self, model_fn, tools):
        self.model_fn = model_fn
        self.tools = tools

    def run(self, query, context=None):
        context = context or []
        return self.model_fn(query, context=context, tools=self.tools)


class UnifiedStorage:
    """Mastra 风格——组合存储。"""
    def __init__(self):
        self.vector_store = {}
        self.document_store = {}

    def store(self, key, value, embedding=None):
        self.document_store[key] = value
        if embedding is not None:
            self.vector_store[key] = embedding

    def search(self, query):
        return [(k, v) for k, v in self.document_store.items()
                if any(w in k.lower() for w in query.lower().split()[:3])]


if __name__ == "__main__":
    print("Agno + Mastra 运行时演示\n")

    agent = StatelessAgent(
        lambda q, **kw: f"Agno 回复: {q[:30]}", {"weather": lambda c: "晴天"}
    )
    print(f"Agno: {agent.run('北京天气')[:40]}")

    store = UnifiedStorage()
    store.store("API文档", "REST API 参考指南")
    store.store("部署指南", "Docker 部署步骤")
    print(f"\nMastra 搜索 'API': {store.search('API 文档')}")
