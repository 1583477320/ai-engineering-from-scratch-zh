# main.py — 故障模式 MAST 实现
# 对应课程：阶段 16 · 23（故障模式——MAST、群体思维、级联错误）

import random
import time
from enum import Enum


# === MAST 分类 ===

class MASTCategory(Enum):
    SPECIFICATION = "specification"
    COORDINATION = "coordination"
    VERIFICATION = "verification"


class FailureTaxonomy:
    CATEGORY_KEYWORDS = {
        MASTCategory.SPECIFICATION: ["角色歧义", "任务定义", "模糊"],
        MASTCategory.COORDINATION: ["状态漂移", "消息丢失", "超时"],
        MASTCategory.VERIFICATION: ["未验证", "未核验", "幻觉传播"],
    }

    @classmethod
    def classify(cls, incident):
        for cat, kws in cls.CATEGORY_KEYWORDS.items():
            if any(kw in incident for kw in kws):
                return cat
        return MASTCategory.COORDINATION


# === 断路器 ===

class CircuitBreaker:
    def __init__(self, threshold=0.1, reset_timeout=30):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.errors = 0
        self.total = 0
        self.open = False
        self.open_since = 0.0

    def call(self, fn, *args, **kwargs):
        if self.open:
            if time.time() - self.open_since > self.reset_timeout:
                self.open = False
            else:
                raise RuntimeError("断路器打开")

        try:
            result = fn(*args, **kwargs)
            self.total += 1
            return result
        except Exception:
            self.total += 1
            self.errors += 1
            if self.errors / max(self.total, 1) > self.threshold:
                self.open = True
                self.open_since = time.time()
            raise


# === 重试风暴模拟 ===

class RetryStormSimulator:
    def __init__(self, failure_rate=0.15, num=100):
        self.failure_rate = failure_rate
        self.num = num

    def without_cb(self):
        load = 0
        retries = 0
        for _ in range(self.num):
            success = False
            attempt = 0
            while not success and attempt < 5:
                attempt += 1
                load += 1
                if random.random() > self.failure_rate:
                    success = True
                else:
                    retries += 1
        return {"load": load, "amplification": load / self.num}

    def with_cb(self):
        cb = CircuitBreaker(threshold=0.1)
        load = 0
        for _ in range(self.num):
            try:
                load += 1
                if random.random() > self.failure_rate:
                    pass
                else:
                    raise RuntimeError("error")
            except RuntimeError:
                pass
        return {"load": load, "amplification": load / self.num}


if __name__ == "__main__":
    sim = RetryStormSimulator()

    r1 = sim.without_cb()
    print(f"无断路器: 负载={r1['load']}, 放大={r1['amplification']:.1f}x")

    r2 = sim.with_cb()
    print(f"有断路器: 负载={r2['load']}, 放大={r2['amplification']:.1f}x")
