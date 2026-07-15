# 生产应用部署：网关、监控、降级


class LLMMetrics:
    """监控指标收集。"""
    def __init__(self):
        self.latencies = []
        self.costs = []
        self.errors = []

    def record(self, latency, cost, error=None):
        self.latencies.append(latency)
        self.costs.append(cost)
        if error:
            self.errors.append(error)

    def summary(self):
        if not self.latencies:
            return {"status": "no data"}
        return {
            "avg_latency": f"{sum(self.latencies)/len(self.latencies)*1000:.1f}ms",
            "p95_latency": f"{sorted(self.latencies)[int(len(self.latencies)*0.95)]*1000:.1f}ms",
            "error_rate": f"{len(self.errors)/max(len(self.latencies),1):.1%}",
            "total_cost": f"${sum(self.costs):.4f}",
        }


class LLMGateway:
    """简单 API 网关。"""
    def __init__(self, models, cache=None, rate_limit=100):
        self.models = models
        self.cache = cache
        self.rate_limit = rate_limit
        self.call_count = 0
        self.metrics = LLMMetrics()

    def handle(self, prompt, system_prompt=""):
        if self.call_count >= self.rate_limit:
            return {"error": "速率限制", "retry_after": 60}

        # 缓存检查
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                self.metrics.record(0.001, 0.0)
                return {"response": cached, "source": "cache"}

        # 尝试模型列表（优雅降级）
        for model in self.models:
            try:
                start = time.time()
                response = f"[{model}] 对'{prompt[:20]}...'的回复"
                latency = time.time() - start
                cost = 0.001

                if self.cache:
                    self.cache.set(prompt, model, response)
                self.call_count += 1
                self.metrics.record(latency, cost)
                return {"response": response, "source": model}
            except Exception as e:
                self.metrics.record(0.0, 0.0, str(e))
                continue

        return {"response": "服务暂时不可用，请稍后再试。", "source": "fallback"}

    def get_metrics(self):
        return self.metrics.summary()


class ResponseCache:
    def __init__(self):
        self.cache = {}
    def get(self, prompt):
        return self.cache.get(hashlib.md5(prompt.encode()).hexdigest())
    def set(self, prompt, model, response):
        self.cache[hashlib.md5(prompt.encode()).hexdigest()] = response


if __name__ == "__main__":
    import hashlib
    print("生产应用部署演示\n")

    cache = ResponseCache()
    gw = LLMGateway(["gpt-4o", "claude-sonnet", "gpt-4o-mini"], cache=cache, rate_limit=10)

    for i in range(12):
        result = gw.handle(f"测试问题 {i % 3}")
        print(f"  请求 {i+1}: source={result['source']}")

    print(f"\n监控: {gw.get_metrics()}")
