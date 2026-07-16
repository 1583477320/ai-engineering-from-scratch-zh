# 智能体评估框架


def evaluate_agent(agent_fn, test_cases, metric_fn):
    """评估智能体。"""
    results = []
    for test in test_cases:
        try:
            result = agent_fn(test["input"])
            passed = metric_fn(result, test["expected"])
            results.append({"task": test["task"], "passed": passed})
        except Exception as e:
            results.append({"task": test["task"], "passed": False, "error": str(e)})
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    return pass_rate, results


def accuracy_metric(result, expected):
    return str(expected).lower() in str(result).lower()


if __name__ == "__main__":
    print("智能体评估演示\n")
    cases = [
        {"task": "加法", "input": "2+3", "expected": "5"},
        {"task": "乘法", "input": "4*5", "expected": "20"},
        {"task": "天气", "input": "北京天气", "expected": "晴天"},
    ]
    rate, results = evaluate_agent(lambda x: x.replace("+"," ").replace("*"," "), cases, accuracy_metric)
    print(f"准确率: {rate:.0%}")
    for r in results:
        print(f"  {r['task']}: {'✓' if r['passed'] else '✗'}")
