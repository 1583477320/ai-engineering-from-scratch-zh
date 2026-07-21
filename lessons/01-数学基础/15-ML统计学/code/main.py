"""ML 统计学——描述统计+相关+Bootstrap。"""
import math, random

def mean(x): return sum(x)/len(x)
def variance(x):
    m = mean(x)
    return sum((xi - m)**2 for xi in x) / (len(x) - 1) if len(x) > 1 else 0

def std(x): return math.sqrt(variance(x))

def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx == 0 or sy == 0: return 0
    return sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / (len(x)*sx*sy)

def bootstrap_ci(data, statistic, b=1000, alpha=0.05):
    vals = [statistic(random.choices(data, k=len(data))) for _ in range(b)]
    vals.sort()
    return statistic(data), vals[int(b*alpha/2)], vals[int(b*(1-alpha/2))]

def main():
    x = [1,2,3,4,5,6,7,8,9,10]
    y = [2,4,5,4,5,6,7,8,9,10]
    print(f"均值={mean(x):.1f} 标准差={std(x):.4f}")
    print(f"Pearson={pearson(x,y):.4f}")
    m, lo, hi = bootstrap_ci(x, mean, 500)
    print(f"Bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
