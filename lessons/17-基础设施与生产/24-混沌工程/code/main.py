"""混沌实验模拟器——带安全平面门控。"""
import time


def simulate_experiment(name, failure_rate, recovery_time, expected_burn=0.5):
    burn_rate = failure_rate * 60
    breach = burn_rate > expected_burn * 2
    time.sleep(min(recovery_time, 0.2))
    return {"experiment": name, "burn_rate": burn_rate,
            "breach": breach, "action": "暂停" if breach else "继续"}


if __name__ == "__main__":
    for name, rate, rec in [("KV驱逐风暴", 0.3, 120), ("提供商宕机", 0.05, 30),
                             ("畸形提示词", 0.02, 5)]:
        r = simulate_experiment(name, rate, rec)
        print(f"{name:15s} 烧毁率={r['burn_rate']:.1f}/min {r['action']}")
