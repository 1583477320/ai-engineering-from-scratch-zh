"""群体公平性度量对比。"""


def demographic_parity(outcomes, groups):
    rates = {}
    for g in set(groups):
        go = [o for o, gr in zip(outcomes, groups) if gr == g]
        rates[g] = sum(go) / len(go) if go else 0
    return rates


def equalized_odds(outcomes, predictions, groups):
    tprs, fprs = {}, {}
    for g in set(groups):
        tp = sum(1 for o, p in zip(outcomes, predictions) if o == 1 and p == 1 and groups[outcomes.index(o)] == g)
        fn = sum(1 for o, p in zip(outcomes, predictions) if o == 1 and p == 0 and groups[outcomes.index(o)] == g)
        fp = sum(1 for o, p in zip(outcomes, predictions) if o == 0 and p == 1 and groups[outcomes.index(o)] == g)
        tn = sum(1 for o, p in zip(outcomes, predictions) if o == 0 and p == 0 and groups[outcomes.index(o)] == g)
        tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tprs, fprs


if __name__ == "__main__":
    outcomes = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    preds = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
    groups = ["A"] * 8 + ["B"] * 7

    dp = demographic_parity(outcomes, groups)
    tprs, fprs = equalized_odps(outcomes, preds, groups)
    print(f"人口统计对等 A={dp['A']:.1%} B={dp['B']:.1%}")
    print(f"机会均等 TPR: A={tprs['A']:.1%} B={tprs['B']:.1%}")
    print(f"不可能性: 在不等基础率下三个标准不能同时满足")
