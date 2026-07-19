"""AI 控制协议对比模拟。"""
import random


def trusted_editing(U_output, T_threshold=0.6):
    t_score = random.uniform(0, 1)
    if t_score < T_threshold:
        return {"action": "human_audit", "t_score": t_score}
    return {"action": "accept", "t_score": t_score}


def untrusted_monitoring(U_output, collusion_prob=0.0):
    monitor_score = random.uniform(0, 1)
    if random.random() < collusion_prob:
        monitor_score = 1.0
    if monitor_score < 0.5:
        return {"action": "audit", "score": monitor_score}
    return {"action": "accept", "score": monitor_score}


if __name__ == "__main__":
    n = 1000
    te_human = sum(1 for _ in range(n) if trusted_editing("out")["action"] == "human_audit")
    um_audit = sum(1 for _ in range(n) if untrusted_monitoring("out")["action"] == "audit")
    print(f"受信编辑路由到人工: {te_human}/{n} ({te_human/n:.1%})")
    print(f"不受信监控审计: {um_audit}/{n} ({um_audit/n:.1%})")
