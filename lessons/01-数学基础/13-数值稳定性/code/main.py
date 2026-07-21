"""数值稳定性——稳定softmax+log-sum-exp+梯度检查。"""
import math

def softmax_stable(logits):
    mx = max(logits)
    exps = [math.exp(z - mx) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

def cross_entropy_stable(true_class, logits):
    mx = max(logits); shifted = [z - mx for z in logits]
    lse = math.log(sum(math.exp(s) for s in shifted))
    return -(shifted[true_class] - lse)

def numerical_gradient(f, x, h=1e-5):
    return [(f([xi + (h if j == i else 0) for j, xi in enumerate(x)]) -
             f([xi - (h if j == i else 0) for j, xi in enumerate(x)])) / (2 * h)
            for i, x in enumerate(x)]

def check_gradient(analytical, numerical, tol=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        err = abs(a - n) / denom
        status = "OK" if err < tol else "FAIL"
        print(f"  param{i}: a={a:.6f} n={n:.6f} err={err:.2e} [{status}]")

def main():
    print("=== 稳定 softmax ===")
    print(f"安全: {softmax_stable([2.0, 1.0, 0.1])}")
    print(f"危险: {softmax_stable([100.0, 101.0, 102.0])}")
    print("\n=== log-sum-exp ===")
    print(f"稳定: {logsumexp_stable([500.0, 501.0, 502.0]):.6f}")
    print("\n=== 梯度检查 ===")
    f = lambda p: p[0]**2 + 3*p[0]*p[1] + p[1]**3
    g = lambda p: [2*p[0]+3*p[1], 3*p[0]+3*p[1]**2]
    check_gradient(g([2.0, 1.0]), numerical_gradient(f, [2.0, 1.0]))
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
