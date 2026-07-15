# 上下文工程：对话历史管理、摘要压缩、滑动窗口


class ContextWindow:
    """上下文窗口管理器。"""
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.system = ""
        self.history = []

    def set_system(self, message):
        self.system = message

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def estimate_tokens(self, text):
        """粗略估计 token 数（中文字≈1.5 token/字）。"""
        return int(len(text) * 1.5)

    def build_messages(self, current_query, recent_n=10):
        """构建消息列表。"""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for msg in self.history[-recent_n:]:
            messages.append(msg)
        messages.append({"role": "user", "content": current_query})
        return messages


class ConversationSummary:
    """对话摘要管理——压缩旧历史。"""
    def __init__(self, keep_recent=5):
        self.keep_recent = keep_recent
        self.summary = ""
        self.recent = []

    def add(self, role, content):
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.keep_recent + 2:
            to_summarize = self.recent[:-self.keep_recent]
            self.recent = self.recent[-self.keep_recent:]
            new_summary = self._summarize(to_summarize)
            self.summary = (self.summary + " " + new_summary).strip()

    def _summarize(self, messages):
        """简化摘要——实际中用 LLM。"""
        key_points = []
        for msg in messages:
            if msg["role"] == "user":
                key_points.append(f"用户问: {msg['content'][:50]}...")
        return " ; ".join(key_points) if key_points else ""

    def get_context(self, query):
        """获取完整上下文。"""
        context = f"[历史摘要] {self.summary}\n" if self.summary else ""
        for msg in self.recent:
            context += f"{msg['role']}: {msg['content']}\n"
        context += f"\n{query}"
        return context


def recursive_summary(text, chunk_size=2000):
    """递归摘要——支持任意长度文档。"""
    if len(text) <= chunk_size:
        return text
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - 200)]
    summaries = [f"[块{i+1}] {chunk[:150]}..." for i, chunk in enumerate(chunks)]
    return "\n".join(summaries)


if __name__ == "__main__":
    # 1. 对话管理
    ctx = ContextWindow(max_tokens=4000)
    ctx.set_system("你是一个有帮助的助手")
    ctx.add_message("user", "第一轮问题")
    ctx.add_message("assistant", "第一轮回答")
    ctx.add_message("user", "第二轮问题")
    ctx.add_message("assistant", "第二轮回答")
    messages = ctx.build_messages("现在的问题")
    print(f"消息数: {len(messages)}, 估计 token: {ctx.estimate_tokens(str(messages))}")

    # 2. 摘要压缩
    conv = ConversationSummary(keep_recent=3)
    for i in range(10):
        conv.add("user", f"第{i+1}个问题 about topic_{i}")
        conv.add("assistant", f"第{i+1}个回答 about topic_{i}")
    print(f"\n历史摘要: {conv.summary[:80]}...")
    print(f"保留最近: {len(conv.recent)} 条")
