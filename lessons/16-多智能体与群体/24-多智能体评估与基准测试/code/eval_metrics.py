# eval_metrics.py — 多智能体系统评估指标
# 依赖：标准库（无第三方依赖）
# 对应课程：阶段 16 · 21（多智能体评估与基准测试）

"""
多智能体系统的评估指标定义，含协作效率、一致性计算和端到端评估流水线。
"""

from dataclasses import dataclass, field
import time


# === 评估指标 ===

@dataclass
class EvaluationMetrics:
    """多智能体系统的评估指标集合。"""

    # 系统级指标
    task_success: bool = False
    total_latency: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    # 协作级指标
    agent_outputs: list = field(default_factory=list)
    communication_rounds: int = 0
    conflicts_detected: int = 0

    # 单智能体指标
    per_agent_tokens: dict = field(default_factory=dict)
    per_agent_latency: dict = field(default_factory=dict)

    def compute_efficiency(self) -> float:
        """协作效率 = 最终输出词元 / 所有智能体总词元。"""
        if self.total_tokens == 0:
            return 0.0
        final_output_tokens = len(self.agent_outputs[-1]) if self.agent_outputs else 0
        return final_output_tokens / self.total_tokens

    def compute_consistency(self) -> float:
        """一致性分数：多个智能体输出的语义一致性（简化版）。"""
        if len(self.agent_outputs) < 2:
            return 1.0

        all_keywords = []
        for output in self.agent_outputs:
            keywords = set(output.split())
            all_keywords.append(keywords)

        overlaps = 0
        pairs = 0
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                intersection = all_keywords[i] & all_keywords[j]
                union = all_keywords[i] | all_keywords[j]
                if union:
                    overlaps += len(intersection) / len(union)
                pairs += 1

        return overlaps / pairs if pairs > 0 else 1.0

    def summary(self) -> dict:
        """返回评估摘要。"""
        return {
            "task_success": self.task_success,
            "total_latency": f"{self.total_latency:.2f}s",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "efficiency": f"{self.compute_efficiency():.2%}",
            "consistency": f"{self.compute_consistency():.2%}",
            "communication_rounds": self.communication_rounds,
            "conflicts_detected": self.conflicts_detected,
        }


# === 评估流水线 ===

class MultiAgentEvaluator:
    """多智能体系统的自动化评估器。"""

    def __init__(self, single_agent_fn, multi_agent_fn):
        """
        Args:
            single_agent_fn: 单智能体方案，输入任务描述，输出结果
            multi_agent_fn: 多智能体方案，输入任务描述，输出 (结果, 指标)
        """
        self.single_agent_fn = single_agent_fn
        self.multi_agent_fn = multi_agent_fn

    def evaluate(self, tasks: list[str]) -> dict:
        """在任务集上评估两个方案。"""
        single_results = []
        multi_results = []

        for task in tasks:
            # 单智能体方案
            start = time.time()
            single_output = self.single_agent_fn(task)
            single_latency = time.time() - start
            single_results.append({
                "task": task,
                "output": single_output,
                "latency": single_latency,
            })

            # 多智能体方案
            start = time.time()
            multi_output, metrics = self.multi_agent_fn(task)
            multi_latency = time.time() - start
            multi_results.append({
                "task": task,
                "output": multi_output,
                "latency": multi_latency,
                "metrics": metrics,
            })

        return self._compare(single_results, multi_results)

    def _compare(self, single_results, multi_results) -> dict:
        """对比两个方案的核心指标。"""
        avg_single_latency = sum(r["latency"] for r in single_results) / len(single_results)
        avg_multi_latency = sum(r["latency"] for r in multi_results) / len(multi_results)
        avg_multi_tokens = sum(
            r["metrics"].total_tokens for r in multi_results
        ) / len(multi_results)

        speed_ratio = avg_single_latency / avg_multi_latency if avg_multi_latency > 0 else 1.0

        return {
            "avg_single_latency": f"{avg_single_latency:.2f}s",
            "avg_multi_latency": f"{avg_multi_latency:.2f}s",
            "avg_multi_tokens": f"{avg_multi_tokens:.0f}",
            "speed_ratio": f"{speed_ratio:.2f}x",
            "recommendation": "使用多智能体" if speed_ratio > 1.0 else "使用单智能体",
        }


# === 演示 ===

if __name__ == "__main__":
    # 模拟单智能体和多智能体方案
    def single_agent(task: str) -> str:
        time.sleep(0.05)
        return f"单智能体完成: {task}"

    def multi_agent(task: str):
        metrics = EvaluationMetrics()
        start = time.time()

        time.sleep(0.03)
        analysis = f"数据分析: {task}"
        metrics.per_agent_tokens["分析师"] = 800
        metrics.agent_outputs.append(analysis)

        time.sleep(0.04)
        report = f"报告撰写: {analysis}"
        metrics.per_agent_tokens["撰写者"] = 1200
        metrics.agent_outputs.append(report)

        metrics.total_tokens = sum(metrics.per_agent_tokens.values())
        metrics.total_latency = time.time() - start
        metrics.task_success = True
        metrics.communication_rounds = 1

        return report, metrics

    evaluator = MultiAgentEvaluator(single_agent, multi_agent)

    tasks = [
        "分析上季度销售数据",
        "生成市场趋势报告",
        "对比竞品优劣势",
    ]

    results = evaluator.evaluate(tasks)

    print("=== 评估结果 ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
