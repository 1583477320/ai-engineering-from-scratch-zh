"""带故障切换的网关路由模拟。"""
import random


class MockGateway:
    def __init__(self, providers):
        self.providers = providers

    def route(self, prompt):
        for provider in self.providers:
            if random.random() < 0.05:  # 5% 错误率
                continue
            return {"provider": provider, "status": "success"}
        return {"provider": None, "status": "all_failed"}


if __name__ == "__main__":
    gw = MockGateway(["openai", "anthropic", "self-hosted"])
    results = [gw.route("测试") for _ in range(100)]
    success = sum(1 for r in results if r["status"] == "success")
    print(f"成功率: {success}/100")
