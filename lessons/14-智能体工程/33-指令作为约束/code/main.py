# 可执行约束引擎


class ConstraintEngine:
    def __init__(self):
        self.constraints = []

    def add(self, name, check_fn, message):
        self.constraints.append({"name": name, "check": check_fn, "message": message})

    def verify(self, response):
        failures = []
        for c in self.constraints:
            if not c["check"](response):
                failures.append(c["message"])
        return failures

    def verify_and_fix(self, response, model_fn, max_attempts=3):
        for _ in range(max_attempts):
            failures = self.verify(response)
            if not failures:
                return response
            response = model_fn(f"修复: {'; '.join(failures)}\n原始:{response}")
        return response


if __name__ == "__main__":
    print("可执行约束演示\n")
    engine = ConstraintEngine()
    engine.add("max_length", lambda r: len(r) <= 50, "超过50字")
    engine.add("no_url", lambda r: "http" not in r, "包含URL")

    for res in ["这是一个短回复", "这是一个较长的回复，包含了一些内容。http://example.com"]:
        failures = engine.verify(res)
        print(f"  '{res[:30]}...' -> {'通过' if not failures else '违反: ' + str(failures)}")
