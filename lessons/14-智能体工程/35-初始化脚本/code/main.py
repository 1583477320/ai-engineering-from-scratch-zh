# 初始化脚本

import time


class InitScript:
    def __init__(self):
        self.probes = []
        self.cache = {}

    def add_probe(self, name, probe_fn, ttl=3600):
        self.probes.append({"name": name, "fn": probe_fn, "ttl": ttl})

    def run(self):
        results = {}
        for probe in self.probes:
            cached = self.cache.get(probe["name"])
            if cached and cached.get("_time", 0) + probe["ttl"] > time.time():
                results[probe["name"]] = cached["value"]
                continue
            value = probe["fn"]()
            self.cache[probe["name"]] = {"value": value, "_time": time.time()}
            results[probe["name"]] = value
        return results


if __name__ == "__main__":
    print("初始化脚本演示\n")
    script = InitScript()
    script.add_probe("环境", lambda: {"python": "3.12", "cuda": False})
    script.add_probe("项目结构", lambda: {"src/": ["main.py"], "tests/": []})
    results = script.run()
    for k, v in results.items():
        print(f"  {k}: {v}")
