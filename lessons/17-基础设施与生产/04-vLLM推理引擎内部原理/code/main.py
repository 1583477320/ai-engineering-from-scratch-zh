"""vLLM 连续批处理调度器模拟。"""
import random

class Request:
    def __init__(self, prompt_len, max_output):
        self.prompt_len = prompt_len
        self.max_output = max_output
        self.generated = 0

    def is_done(self):
        return self.generated >= self.max_output


class Simulator:
    def __init__(self):
        self.RUNNING = []
        self.WAITING = []
        self.total_tokens = 0
        self.steps = 0

    def add(self, prompt_len, max_output):
        self.WAITING.append(Request(prompt_len, max_output))

    def step(self):
        self.RUNNING = [r for r in self.RUNNING if not r.is_done()]
        while self.WAITING:
            self.RUNNING.append(self.WAITING.pop(0))
        if not self.RUNNING:
            return
        for r in self.RUNNING:
            r.generated += 1
            self.total_tokens += 1
        self.steps += 1

    def run(self, steps=200):
        for _ in range(steps):
            if random.random() < 0.3 and len(self.WAITING) < 5:
                self.add(random.randint(500, 4000), random.randint(100, 800))
            self.step()
        return {"tokens": self.total_tokens, "steps": self.steps,
                "avg_batch": self.total_tokens / max(self.steps, 1)}


if __name__ == "__main__":
    r = Simulator().run()
    print(f"总词元: {r['tokens']}  总步数: {r['steps']}  平均批次: {r['avg_batch']:.1f}")
