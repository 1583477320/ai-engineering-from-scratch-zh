# MemGPT 记忆管理器


class MemoryManager:
    """简化版记忆管理器——核心记忆+归档记忆。"""
    def __init__(self, max_context=2048):
        self.max_context = max_context
        self.core_memory = []
        self.archival = []
        self.current_tokens = 0

    def add_core(self, info):
        self.core_memory.append(info)
        self.current_tokens += len(info) // 4

    def add_archival(self, info):
        self.archival.append(info)

    def search_archival(self, query):
        return [a for a in self.archival if any(k in a for k in query.split()[:3])]

    def get_context(self):
        context = "\n".join(self.core_memory)
        while len(context) > self.max_context * 4:
            self.archival.append(self.core_memory.pop(0))
            context = "\n".join(self.core_memory)
        return context

    def status(self):
        return f"核心: {len(self.core_memory)}, 归档: {len(self.archival)}, tokens: {self.current_tokens}"


if __name__ == "__main__":
    print("MemGPT 记忆管理器演示\n")
    mm = MemoryManager(max_context=500)
    mm.add_core("用户喜欢简洁回答")
    mm.add_core("用户是 Python 开发者")
    mm.add_core("上次讨论了 Transformer 架构")
    mm.add_archival("2023-10: 讨论了 RAG 管道")
    mm.add_archival("2023-11: 讨论了微调策略")
    print(mm.status())
    print(f"搜索 'RAG': {mm.search_archival('RAG')}")
    print(f"上下文: {mm.get_context()[:80]}...")
