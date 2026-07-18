# resilient_runner.py — 带故障隔离的智能体运行器
# 依赖：标准库（无第三方依赖）
# 对应课程：阶段 16 · 20（多智能体系统生产部署）

"""
生产级智能体运行器，支持超时、重试、断路器和成本追踪。
"""

import time
import random
from dataclasses import dataclass, field
from collections import defaultdict


# === 配置 ===

@dataclass
class AgentConfig:
    """单个智能体的配置。"""
    name: str
    timeout: float = 30.0       # 单次调用超时（秒）
    max_retries: int = 2        # 最大重试次数
    max_tokens: int = 3000      # 词元预算上限


# === 断路器 ===

class CircuitBreaker:
    """断路器：连续失败 N 次后暂时跳过调用。"""

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        """
        Args:
            failure_threshold: 连续失败多少次后断路
            reset_timeout: 断路持续时间（秒）
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.open_since = 0.0

    def is_open(self) -> bool:
        """断路器是否打开（拒绝调用）。"""
        if self.failures < self.failure_threshold:
            return False
        # 检查是否到了半开时间
        if time.time() - self.open_since >= self.reset_timeout:
            return False  # 半开：允许尝试一次
        return True

    def record_success(self):
        """记录成功：重置失败计数。"""
        self.failures = 0

    def record_failure(self):
        """记录失败：增加失败计数，可能触发断路。"""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_since = time.time()


# === 成本追踪器 ===

class CostTracker:
    """追踪词元消耗和费用。"""

    # GPT-4o 的价格（美元/词元）
    PRICE_PER_TOKEN = {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000}

    def __init__(self, budget_limit: float = 1.0):
        self.budget_limit = budget_limit
        self.records = []
        self.total_cost = 0.0

    def record(self, agent_name: str, input_tokens: int, output_tokens: int):
        """记录一次 API 调用的词元消耗。"""
        cost = (
            input_tokens * self.PRICE_PER_TOKEN["input"]
            + output_tokens * self.PRICE_PER_TOKEN["output"]
        )
        self.total_cost += cost
        self.records.append({
            "agent": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative": self.total_cost,
        })

        if self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"预算超限: {self.total_cost:.4f} > {self.budget_limit:.4f} 美元"
            )

    def summary(self) -> dict:
        """返回成本摘要。"""
        by_agent = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        for r in self.records:
            by_agent[r["agent"]]["tokens"] += r["input_tokens"] + r["output_tokens"]
            by_agent[r["agent"]]["cost"] += r["cost"]
        return {
            "total_cost": self.total_cost,
            "budget_remaining": self.budget_limit - self.total_cost,
            "by_agent": dict(by_agent),
        }


class BudgetExceededError(Exception):
    """预算超限异常。"""
    pass


# === 容错运行器 ===

class ResilientAgentRunner:
    """带超时、重试和断路器的智能体运行器。"""

    def __init__(self, config: AgentConfig, circuit_breaker: CircuitBreaker = None):
        self.config = config
        self.cb = circuit_breaker or CircuitBreaker()

    def run(self, prompt: str, llm_call) -> dict:
        """执行智能体调用。

        Args:
            prompt: 输入提示词
            llm_call: LLM 调用函数，签名 fn(prompt, max_tokens, timeout) -> str

        Returns:
            包含 status, output, latency, attempt 的字典
        """
        # 断路器检查
        if self.cb.is_open():
            return {"status": "circuit_open", "output": None, "latency": 0, "attempt": 0}

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start = time.time()
                result = llm_call(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                elapsed = time.time() - start

                self.cb.record_success()
                return {
                    "status": "success",
                    "output": result,
                    "latency": elapsed,
                    "attempt": attempt + 1,
                }
            except TimeoutError:
                last_error = "超时"
            except Exception as e:
                last_error = str(e)

            # 指数退避 + 随机抖动
            if attempt < self.config.max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(min(backoff, 10))

        # 所有重试都失败
        self.cb.record_failure()
        return {"status": "failed", "output": None, "error": last_error, "latency": 0, "attempt": self.config.max_retries + 1}


# === 演示 ===

if __name__ == "__main__":
    # 模拟 LLM 调用（有 30% 概率超时）
    def mock_llm_call(prompt, max_tokens=1000, timeout=30):
        if random.random() < 0.3:
            raise TimeoutError("模拟超时")
        time.sleep(0.1)
        return f"模拟输出: {prompt[:20]}..."

    config = AgentConfig(name="测试智能体", timeout=5, max_retries=2)
    runner = ResilientAgentRunner(config)

    # 运行 10 次，观察断路器行为
    for i in range(10):
        result = runner.run(f"测试任务 {i}", mock_llm_call)
        print(f"  调用 {i+1}: status={result['status']}, "
              f"attempt={result['attempt']}, "
              f"latency={result['latency']:.3f}s")
