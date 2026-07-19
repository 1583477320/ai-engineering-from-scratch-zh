"""W2SG PGR 计算模拟。"""


def compute_pgr(weak_acc, fine_tuned_acc, ceiling_acc):
    gap = ceiling_acc - weak_acc
    if gap <= 0:
        return 0.0
    return (fine_tuned_acc - weak_acc) / gap


if __name__ == "__main__":
    ceiling = 0.95
    print(f"{'弱准确率':>8}  {'PGR':>6}  {'微调准确率':>10}")
    print("-" * 30)
    for w in [0.50, 0.60, 0.70, 0.80]:
        ft = w + 0.5 * (ceiling - w)
        pgr = compute_pgr(w, ft, ceiling)
        print(f"{w:>8.1%}  {pgr:>6.1%}  {ft:>10.1%}")
