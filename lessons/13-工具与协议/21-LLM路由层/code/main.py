# LLM 路由网关

import random


class LLMRouter:
    """简化版 LLM 路由网关。"""
    def __init__(self):
        self.models = {}
        self.fallback = None
        self.costs = {}

    def register(self, name, model_fn, cost_per_call=0.01):
        self.models[name] = {"fn": model_fn, "cost": cost_per_call}

    def set_fallback(self, name):
        self.fallback = name

    def route(self, prompt, strategy="cheapest"):
        if strategy == "cheapest":
            best = min(self.models, key=lambda n: self.models[n]["cost"])
        elif strategy == "best":
            best = max(self.models, key=lambda n: self.models[n]["cost"])
        else:
            best = random.choice(list(self.models.keys()))

        try:
            result = self.models[best]["fn"](prompt)
            cost = self.models[best]["cost"]
            self.costs[best] = self.costs.get(best, 0) + cost
            return {"response": result, "model": best, "cost": cost}
        except Exception:
            if self.fallback:
                result = self.models[self.fallback]["fn"](prompt)
                return {"response": result, "model": self.fallback, "fallback": True}
            return {"error": "所有模型不可用"}


if __name__ == "__main__":
    print("LLM 路由网关演示\n")
    router = LLMRouter()
    router.register("gpt-4o-mini", lambda p: f"[小模型] 回复: {p[:20]}...", cost_per_call=0.01)
    router.register("gpt-4o", lambda p: f"[大模型] 回复: {p[:20]}...", cost_per_call=0.1)
    router.set_fallback("gpt-4o-mini")

    for strategy in ["cheapest", "best"]:
        result = router.route("你好，天气怎么样？", strategy=strategy)
        print(f"  策略 {strategy}: {result['model']} (成本: ${result['cost']})")

    print(f"\n累计成本: {router.costs}")
