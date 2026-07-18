# main.py — 评估与基准测试实现
# 对应课程：阶段 16 · 24（评估与协作基准测试）

import random


# === 里程碑 KPI ===

class MilestoneTracker:
    def __init__(self, milestones):
        self.milestones = milestones
        self.completed = set()

    def mark(self, name):
        self.completed.add(name)

    def rate(self):
        return len(self.completed) / len(self.milestones) if self.milestones else 0


# === 基线对比 ===

def simulate_system(name, accuracy):
    """模拟一个系统的基准测试结果。"""
    return {"name": name, "accuracy": accuracy, "cost": random.uniform(0.5, 2.0)}


def compare_systems(systems):
    """比较多个系统，识别最佳性价比。"""
    for s in systems:
        s["cost_per_point"] = s["cost"] / max(s["accuracy"], 0.01)
        print(f"  {s['name']}: 准确率={s['accuracy']:.1%}, "
              f"成本={s['cost']:.2f}, 性价比={s['cost_per_point']:.3f}")

    best_acc = max(systems, key=lambda s: s["accuracy"])
    best_value = min(systems, key=lambda s: s["cost_per_point"])
    print(f"\n  最高准确率: {best_acc['name']} ({best_acc['accuracy']:.1%})")
    print(f"  最佳性价比: {best_value['name']} (每点 {best_value['cost_per_point']:.3f})")


# === 基准声明审查 ===

class BenchmarkClaimChecker:
    def __init__(self):
        self.checks = {
            "基准测试是否指定": False,
            "污染是否检查": False,
            "是否包含基线对比": False,
            "是否报告统计显著性": False,
            "任务是否多样化": False,
            "成本是否披露": False,
        }

    def grade(self):
        score = sum(self.checks.values()) / len(self.checks)
        if score >= 5 / 6:
            return "A — 可信"
        elif score >= 3 / 6:
            return "B — 部分可信"
        else:
            return "C — 证据不足"


if __name__ == "__main__":
    print("=== 系统对比 ===")
    systems = [
        simulate_system("单 LLM 基线", 0.35),
        simulate_system("多智能体 (星型)", 0.52),
        simulate_system("多智能体 (图型)", 0.61),
    ]
    compare_systems(systems)

    print("\n=== 基准声明审查 ===")
    checker = BenchmarkClaimChecker()
    print(f"评分: {checker.grade()}")
