# LLM 评估：LLM-as-Judge + 自动评估流水线


class LLMJudge:
    """LLM 自动评判者。"""
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def evaluate(self, task, response, reference=None):
        prompt = f"评估回答质量（1-10分）:\n任务: {task}\n回答: {response}"
        if reference:
            prompt += f"\n参考: {reference}"
        prompt += "\n评分:"
        score = self.model_fn(prompt)
        return self._parse_score(score)

    def _parse_score(self, text):
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 5


def batch_evaluate(test_cases, judge_fn, generate_fn):
    results = []
    for task, reference in test_cases:
        response = generate_fn(task)
        score = judge_fn(task, response, reference)
        results.append({"task": task, "score": score})
    avg = sum(r["score"] for r in results) / len(results)
    return avg, results


if __name__ == "__main__":
    judge = LLMJudge(lambda x: "8")  # 模拟
    generate = lambda x: f"关于'{x}'的回答内容"
    test_cases = [("Python是什么？", "编程语言"), ("量子力学？", "物理学")]
    avg, results = batch_evaluate(test_cases, judge.evaluate, generate)
    print(f"平均分数: {avg:.1f}")
