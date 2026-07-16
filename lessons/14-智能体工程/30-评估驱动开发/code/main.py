# 三层评估框架


class EvalPipeline:
    def __init__(self):
        self.static = []
        self.offline = []

    def add_static(self, name, input_data, expected):
        self.static.append({"name": name, "input": input_data, "expected": expected})

    def add_offline(self, name, input_data):
        self.offline.append({"name": name, "input": input_data})

    def run_static(self, agent_fn):
        pass_count = 0
        for t in self.static:
            if agent_fn(t["input"]) == t["expected"]:
                pass_count += 1
        return f"静态: {pass_count}/{len(self.static)}"

    def run_offline(self, agent_fn):
        return f"离线: {len(self.offline)} 个已测"


if __name__ == "__main__":
    print("评估驱动开发演示\n")
    pipe = EvalPipeline()
    pipe.add_static("加法", "2+3", "6")
    result = pipe.run_static(lambda x: str(eval(x)) if "+" in x else "0")
    print(f"  {result}")
