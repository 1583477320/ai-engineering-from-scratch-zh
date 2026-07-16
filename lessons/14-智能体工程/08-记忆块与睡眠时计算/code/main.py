# 记忆块 + 睡眠时计算


class MemoryBlock:
    """可编辑的功能记忆块。"""
    def __init__(self, block_type, key, value, confidence=1.0):
        self.type = block_type
        self.key = key
        self.value = value
        self.confidence = confidence

    def update(self, new_value, confidence=1.0):
        self.value = new_value
        self.confidence = max(self.confidence, confidence)

    def __repr__(self):
        return f"Block({self.type}: {self.key}={self.value})"


class SleepTimeCompute:
    """睡眠时记忆整合。"""
    def __init__(self, memory):
        self.memory = memory
        self.temp_buffer = []

    def on_interaction(self, event):
        self.temp_buffer.append(event)

    def consolidate(self):
        if not self.temp_buffer:
            return
        key_facts = [e for e in self.temp_buffer if e.get("priority", 0) > 0.5]
        for fact in key_facts:
            self.memory.append(fact["content"])
        self.temp_buffer = []
        print(f"  整合: {len(key_facts)} 条关键信息")


if __name__ == "__main__":
    print("记忆块 + 睡眠时计算演示\n")
    memory = []
    sleep = SleepTimeCompute(memory)

    for i in range(5):
        sleep.on_interaction({"content": f"交互 {i}: 用户偏好 Python", "priority": 0.8 if i < 2 else 0.3})

    print(f"缓冲区: {len(sleep.temp_buffer)} 条")
    sleep.consolidate()
    print(f"核心记忆: {len(memory)} 条")
    print(f"内容: {memory}")
