"""多智能体事故分类模拟。"""


def simulate_triage(log_errors, metric_anomaly, runbook_match):
    hypotheses = []
    if log_errors:
        hypotheses.append({"source": "日志", "hypothesis": f"检测到{len(log_errors)}个错误", "confidence": 0.8})
    if metric_anomaly:
        hypotheses.append({"source": "指标", "hypothesis": "CPU/内存异常", "confidence": 0.75})
    if runbook_match:
        hypotheses.append({"source": "手册", "hypothesis": f"匹配: {runbook_match}", "confidence": 0.9})
    consistent = len(set(h["hypothesis"] for h in hypotheses)) == 1 if len(hypotheses) >= 2 else False
    return {"hypotheses": hypotheses, "consistent": consistent,
            "action": "执行" if consistent else "升级给人类"}


if __name__ == "__main__":
    r = simulate_triage(["vLLM OOM", "KV cache spike"], True, "vLLM OOM - 扩容")
    print(f"假设: {len(r['hypotheses'])}  一致: {r['consistent']}  动作: {r['action']}")
    for h in r["hypotheses"]:
        print(f"  {h['source']}: {h['hypothesis']} (置信度: {h['confidence']})")
