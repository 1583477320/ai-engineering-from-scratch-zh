# Web 智能体评估


def evaluate_web_agent(agent_fn, test_cases, metric_fn):
    """评估 Web 智能体。"""
    results = []
    for test in test_cases:
        try:
            result = agent_fn(test["task"], test.get("initial_state", {}))
            passed = metric_fn(result, test.get("expected_state", {}))
            results.append({"task": test["task"], "passed": passed})
        except Exception as e:
            results.append({"task": test["task"], "passed": False, "error": str(e)})
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    return pass_rate, results


if __name__ == "__main__":
    print("Web 智能体评估演示\n")
    cases = [
        {"task": "打开搜索", "expected_state": {"page": "search"}},
        {"task": "登录", "expected_state": {"page": "dashboard"}},
        {"task": "提交表单", "expected_state": {"submitted": True}},
    ]
    rate, results = evaluate_agent(
        lambda task, state: {"page": "search" if "搜索" in task else "ok"},
        cases,
        lambda result, expected: result.get("page") == expected.get("page")
    )
    print(f"通过率: {rate:.0%}")
    for r in results:
        print(f"  {r['task']}: {'✓' if r['passed'] else '✗'}")
