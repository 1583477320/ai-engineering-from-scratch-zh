"""机器学习微积分——数值导数+梯度下降+Hessian。"""
import math

def numerical_derivative(f, x, h=1e-7):
    return (f(x+h) - f(x-h)) / (2*h)

def gradient_descent_2d(f, start, lr=0.1, steps=30):
    pt = list(start)
    for step in range(steps):
        grads = []
        for i in range(len(pt)):
            pp = list(pt); pp[i] += 1e-7
            pm = list(pt); pm[i] -= 1e-7
            grads.append((f(pp) - f(pm)) / 2e-7)
        pt = [p - lr*g for p, g in zip(pt, grads)]
    return pt

def main():
    f = lambda x: x**2
    for x in [-2,0,2]:
        print(f"x={x}: 数值导数={numerical_derivative(f,x):.4f} 解析={2*x}")
    pt = gradient_descent_2d(lambda p: p[0]**2+p[1]**2, [4.0,3.0])
    print(f"\n梯度下降到最小值: ({pt[0]:.4f}, {pt[1]:.4f}) (应接近 (0,0))")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
