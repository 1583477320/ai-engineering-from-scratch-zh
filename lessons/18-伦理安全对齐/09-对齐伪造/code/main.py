"""合规差距模拟。"""


def compliance_gap(monitored=True):
    base_refusal = 0.97
    comply = 0.12 if monitored else 0.03
    gap = comply - (1 - base_refusal)
    return {"comply_rate": comply, "unmonitored": 1 - base_refusal, "gap": gap}


if __name__ == "__main__":
    for label, m in [("付费层(未监控)", False), ("免费层(受监控)", True)]:
        r = compliance_gap(m)
        print(f"{label}  合规率={r['comply_rate']:.1%}  差距={r['gap']:.1%}")
