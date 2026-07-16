# AutoGen 演员模型实现


class ActorAgent:
    """简化版演员模型——异步消息传递。"""
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler
        self.inbox = []

    def send(self, message):
        self.inbox.append(message)

    def process(self):
        if not self.inbox:
            return None
        message = self.inbox.pop(0)
        return self.handler(message)


def demo():
    print("AutoGen 演员模型演示\n")

    def coder_handler(msg):
        return f"Coder: 生成了 {len(msg)} 行代码"

    def reviewer_handler(msg):
        return f"Reviewer: 评审了 {len(msg)} 行代码"

    coder = ActorAgent("Coder", coder_handler)
    reviewer = ActorAgent("Reviewer", reviewer_handler)

    # 模拟消息传递
    coder.send("实现排序函数")
    result1 = coder.process()
    print(f"  {result1}")

    reviewer.send(result1)
    result2 = reviewer.process()
    print(f"  {result2}")


if __name__ == "__main__":
    demo()
